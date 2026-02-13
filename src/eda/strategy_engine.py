"""StrategyEngine — BaseStrategy를 이벤트 기반으로 래핑.

기존 벡터화 전략(BaseStrategy)을 bar-by-bar 이벤트 기반으로 실행하는 Adapter입니다.
BaseStrategy 코드를 변경하지 않고 EDA에서 동일한 전략 로직을 재사용합니다.

Rules Applied:
    - Adapter Pattern: 기존 인터페이스를 새로운 컨텍스트에 적용
    - Stateless Strategy: 전략은 시그널만 생성, 상태는 PM이 관리
    - No Signal Dedup: 매 bar마다 SignalEvent 발행 (PM의 should_rebalance가 필터링)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.core.events import AnyEvent, BarEvent, EventType, RiskAlertEvent, SignalEvent
from src.models.types import Direction

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.regime.service import RegimeService
    from src.strategy.base import BaseStrategy


# 연속 실패 임계값 (이 횟수 이상 연속 실패 시 RiskAlertEvent 발행)
_CONSECUTIVE_FAILURE_LIMIT = 3


class StrategyEngine:
    """BaseStrategy를 이벤트 기반으로 래핑하는 Adapter.

    1. BarEvent 수신 -> 내부 버퍼에 OHLCV 누적
    2. warmup 이상 데이터 -> strategy.run() 호출
    3. 매 bar마다 SignalEvent 발행 (PM의 should_rebalance가 불필요한 주문 필터링)

    Args:
        strategy: 벡터화 전략 인스턴스
        warmup_periods: 워밍업 기간 (None이면 자동 감지)
        target_timeframe: 타겟 타임프레임
        incremental: True면 strategy.run_incremental() 호출 (fast_mode용)
        max_buffer_size: 버퍼 최대 크기. 초과 시 오래된 데이터 제거. None이면 무제한.
        precomputed_signals: 사전 계산된 시그널 {symbol: StrategySignals}.
            설정 시 strategy.run() 호출 없이 timestamp lookup으로 시그널 발행.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        warmup_periods: int | None = None,
        target_timeframe: str = "1D",
        *,
        incremental: bool = False,
        max_buffer_size: int | None = None,
        precomputed_signals: dict[str, object] | None = None,
        regime_service: RegimeService | None = None,
    ) -> None:
        self._strategy = strategy
        self._warmup = warmup_periods or self._detect_warmup()
        self._buffers: dict[str, list[dict[str, float]]] = {}
        self._timestamps: dict[str, list[datetime]] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._bus: EventBus | None = None
        self._target_timeframe = target_timeframe
        self._incremental = incremental
        self._max_buffer_size = max_buffer_size
        self._precomputed_signals = precomputed_signals
        self._regime_service = regime_service

    async def register(self, bus: EventBus) -> None:
        """EventBus에 BarEvent 구독 등록.

        Args:
            bus: 이벤트 버스
        """
        self._bus = bus
        bus.subscribe(EventType.BAR, self._on_bar)

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent 핸들러.

        1. OHLCV 버퍼에 누적
        2. warmup 미달 시 스킵
        3. strategy.run() 호출 (또는 precomputed lookup)
        4. 매 bar SignalEvent 발행
        """
        assert isinstance(event, BarEvent)
        bar = event

        # TF 필터: target_timeframe에 해당하는 TF만 처리
        if bar.timeframe != self._target_timeframe:
            return

        symbol = bar.symbol
        bus = self._bus
        assert bus is not None

        # precomputed_signals 모드: timestamp lookup만 수행
        if self._precomputed_signals is not None:
            await self._on_bar_precomputed(bar, bus)
            return

        # 1. 버퍼에 누적
        if symbol not in self._buffers:
            self._buffers[symbol] = []
            self._timestamps[symbol] = []

        self._buffers[symbol].append(
            {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
        self._timestamps[symbol].append(bar.bar_timestamp)

        # 1b. 버퍼 크기 제한 (max_buffer_size 설정 시)
        if self._max_buffer_size is not None and len(self._buffers[symbol]) > self._max_buffer_size:
            trim = len(self._buffers[symbol]) - self._max_buffer_size
            self._buffers[symbol] = self._buffers[symbol][trim:]
            self._timestamps[symbol] = self._timestamps[symbol][trim:]

        # 2. warmup 체크
        buf_len = len(self._buffers[symbol])
        if buf_len < self._warmup:
            return

        # 3. DataFrame 구성 + strategy.run() / run_incremental()
        df = pd.DataFrame(
            self._buffers[symbol],
            index=pd.DatetimeIndex(self._timestamps[symbol], tz=UTC),
        )

        # 3.5 Regime 보강 (RegimeService가 있으면 regime 컬럼 추가)
        if self._regime_service is not None:
            df = self._enrich_with_regime(df, symbol)

        try:
            if self._incremental:
                _, signals = self._strategy.run_incremental(df)
            else:
                _, signals = self._strategy.run(df)
        except (ValueError, TypeError) as exc:
            count = self._consecutive_failures.get(symbol, 0) + 1
            self._consecutive_failures[symbol] = count
            logger.warning(
                "Strategy.run() failed for {}, buffer_size={}, consecutive={}",
                symbol,
                buf_len,
                count,
            )
            if count >= _CONSECUTIVE_FAILURE_LIMIT:
                logger.error(
                    "Strategy.run() failed {} consecutive times for {}: {}",
                    count,
                    symbol,
                    exc,
                )
                alert = RiskAlertEvent(
                    alert_level="WARNING",
                    message=f"Strategy.run() failed {count} consecutive times for {symbol}: {exc}",
                    correlation_id=bar.correlation_id,
                    source="StrategyEngine",
                )
                await bus.publish(alert)
            return

        # 성공 시 연속 실패 카운터 리셋
        self._consecutive_failures[symbol] = 0

        # 4. 최신 시그널 추출 + SignalEvent 발행 (매 bar)
        latest_direction = int(signals.direction.iloc[-1])
        latest_strength = float(signals.strength.iloc[-1])
        signal_event = SignalEvent(
            symbol=symbol,
            strategy_name=self._strategy.name,
            direction=Direction(latest_direction),
            strength=latest_strength,
            bar_timestamp=bar.bar_timestamp,
            correlation_id=bar.correlation_id,
            source="StrategyEngine",
        )
        await bus.publish(signal_event)

    async def _on_bar_precomputed(self, bar: BarEvent, bus: EventBus) -> None:
        """Precomputed signals 모드: timestamp로 사전 계산된 시그널을 조회하여 발행."""
        from src.strategy.types import StrategySignals

        assert self._precomputed_signals is not None
        symbol = bar.symbol
        signals = self._precomputed_signals.get(symbol)
        if signals is None:
            return

        assert isinstance(signals, StrategySignals)

        # bar_timestamp로 시그널 lookup
        ts = bar.bar_timestamp
        if ts not in signals.direction.index:
            return

        direction_val = int(signals.direction.loc[ts])
        strength_val = float(signals.strength.loc[ts])

        signal_event = SignalEvent(
            symbol=symbol,
            strategy_name=self._strategy.name,
            direction=Direction(direction_val),
            strength=strength_val,
            bar_timestamp=ts,
            correlation_id=bar.correlation_id,
            source="StrategyEngine",
        )
        await bus.publish(signal_event)

    def inject_warmup(
        self,
        symbol: str,
        bars: list[dict[str, float]],
        timestamps: list[datetime],
    ) -> None:
        """REST API에서 가져온 과거 데이터를 버퍼에 주입 (라이브 warmup용).

        Args:
            symbol: 거래 심볼
            bars: OHLCV dict 리스트 (open, high, low, close, volume)
            timestamps: bar별 타임스탬프 리스트

        Raises:
            ValueError: bars/timestamps 길이 불일치 또는 이미 데이터가 있는 경우
        """
        if len(bars) != len(timestamps):
            msg = f"bars ({len(bars)}) and timestamps ({len(timestamps)}) length mismatch"
            raise ValueError(msg)

        if symbol in self._buffers and len(self._buffers[symbol]) > 0:
            msg = f"Buffer for {symbol} is not empty, cannot inject warmup"
            raise ValueError(msg)

        self._buffers[symbol] = list(bars)
        self._timestamps[symbol] = list(timestamps)
        logger.info(
            "Injected {} warmup bars for {} (warmup_needed={})",
            len(bars),
            symbol,
            self._warmup,
        )

    def _enrich_with_regime(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """RegimeService로 DataFrame에 regime 컬럼을 추가.

        1. 사전 계산 있으면 timestamp join
        2. 없으면 현재 state를 전체 행에 broadcast (live fallback)
        """
        assert self._regime_service is not None
        df = self._regime_service.enrich_dataframe(df, symbol)

        # Live fallback: 사전 계산 없으면 현재 state broadcast
        if "regime_label" not in df.columns:
            cols = self._regime_service.get_regime_columns(symbol)
            if cols is not None:
                for col_name, col_val in cols.items():
                    df[col_name] = col_val

        return df

    def _detect_warmup(self) -> int:
        """전략 설정에서 warmup 기간 자동 감지.

        config.warmup_periods()가 있으면 우선 사용 (가장 정확).
        없으면 안전한 기본값 50 반환.

        Returns:
            감지된 warmup 기간
        """
        config = self._strategy.config
        if config is not None:
            warmup_fn = getattr(config, "warmup_periods", None)
            if warmup_fn is not None:
                return int(warmup_fn())
        return 50  # 안전한 기본값

    @property
    def warmup_periods(self) -> int:
        """현재 워밍업 기간."""
        return self._warmup
