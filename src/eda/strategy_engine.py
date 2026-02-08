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
    from src.strategy.base import BaseStrategy


# 연속 실패 임계값 (이 횟수 이상 연속 실패 시 RiskAlertEvent 발행)
_CONSECUTIVE_FAILURE_LIMIT = 3


class StrategyEngine:
    """BaseStrategy를 이벤트 기반으로 래핑하는 Adapter.

    1. BarEvent 수신 → 내부 버퍼에 OHLCV 누적
    2. warmup 이상 데이터 → strategy.run() 호출
    3. 매 bar마다 SignalEvent 발행 (PM의 should_rebalance가 불필요한 주문 필터링)

    Args:
        strategy: 벡터화 전략 인스턴스
        warmup_periods: 워밍업 기간 (None이면 자동 감지)
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        warmup_periods: int | None = None,
        target_timeframe: str = "1D",
    ) -> None:
        self._strategy = strategy
        self._warmup = warmup_periods or self._detect_warmup()
        self._buffers: dict[str, list[dict[str, float]]] = {}
        self._timestamps: dict[str, list[datetime]] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._bus: EventBus | None = None
        self._target_timeframe = target_timeframe

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
        3. strategy.run() 호출
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

        # 2. warmup 체크
        buf_len = len(self._buffers[symbol])
        if buf_len < self._warmup:
            return

        # 3. DataFrame 구성 + strategy.run()
        df = pd.DataFrame(
            self._buffers[symbol],
            index=pd.DatetimeIndex(self._timestamps[symbol], tz=UTC),
        )

        try:
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

    def _detect_warmup(self) -> int:
        """전략 설정에서 warmup 기간 자동 감지.

        Returns:
            감지된 warmup 기간
        """
        config = self._strategy.config
        if config is not None:
            config_dict = config.model_dump()
            lookback = config_dict.get("lookback", 0)
            vol_window = config_dict.get("vol_window", 0)
            if lookback > 0:
                return int(max(lookback, vol_window) + 10)
        return 50  # 안전한 기본값

    @property
    def warmup_periods(self) -> int:
        """현재 워밍업 기간."""
        return self._warmup
