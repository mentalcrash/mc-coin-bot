"""StrategyPod — 전략별 독립 실행 단위.

하나의 BaseStrategy를 래핑하여 bar-by-bar 시그널을 생성하고,
Pod 레벨 포지션·수익률을 추적합니다.

Rules Applied:
    - Adapter Pattern: BaseStrategy를 bar-by-bar 컨텍스트에 적용
    - Stateless Strategy: 전략은 시그널만 생성, 상태는 Pod이 관리
    - #10 Python Standards: Modern typing, named constants
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.orchestrator.models import LifecycleState, PodPerformance, PodPosition

if TYPE_CHECKING:
    from src.orchestrator.config import OrchestratorConfig, PodConfig
    from src.strategy.base import BaseStrategy

# ── Constants ─────────────────────────────────────────────────────

_MAX_BUFFER_SIZE = 1000
_MIN_WEIGHT_THRESHOLD = 1e-8
_DEFAULT_WARMUP = 50


# ── StrategyPod ──────────────────────────────────────────────────


class StrategyPod:
    """전략별 독립 실행 단위.

    BaseStrategy를 래핑하여 bar-by-bar로 시그널을 생성하고,
    Pod 내부의 포지션·수익률을 추적합니다.

    Args:
        config: Pod 설정 (PodConfig)
        strategy: BaseStrategy 인스턴스
        capital_fraction: Allocator가 할당한 자본 비중
    """

    def __init__(
        self,
        config: PodConfig,
        strategy: BaseStrategy,
        capital_fraction: float,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._capital_fraction = capital_fraction
        self._state = LifecycleState.INCUBATION

        # StrategyEngine 패턴 미러: 심볼별 OHLCV 버퍼
        self._buffers: dict[str, list[dict[str, float]]] = {}
        self._timestamps: dict[str, list[datetime]] = {}

        # 시그널 가중치 (최신)
        self._target_weights: dict[str, float] = {}

        # Pod 레벨 포지션
        self._positions: dict[str, PodPosition] = {}

        # 성과 추적
        self._performance = PodPerformance(pod_id=config.pod_id)
        self._daily_returns: list[float] = []

        # warmup 감지
        self._warmup = self._detect_warmup()

    # ── Properties ─────────────────────────────────────────────────

    @property
    def pod_id(self) -> str:
        """Pod 고유 식별자."""
        return self._config.pod_id

    @property
    def symbols(self) -> tuple[str, ...]:
        """거래 대상 심볼 목록."""
        return self._config.symbols

    @property
    def state(self) -> LifecycleState:
        """현재 생애주기 상태."""
        return self._state

    @state.setter
    def state(self, value: LifecycleState) -> None:
        self._state = value

    @property
    def capital_fraction(self) -> float:
        """Allocator가 할당한 자본 비중."""
        return self._capital_fraction

    @capital_fraction.setter
    def capital_fraction(self, value: float) -> None:
        self._capital_fraction = value

    @property
    def performance(self) -> PodPerformance:
        """성과 추적 컨테이너."""
        return self._performance

    @property
    def is_active(self) -> bool:
        """RETIRED가 아니면 활성."""
        return self._state != LifecycleState.RETIRED

    @property
    def warmup_periods(self) -> int:
        """워밍업 기간."""
        return self._warmup

    @property
    def daily_returns_series(self) -> pd.Series:
        """Allocator용 일별 수익률 Series."""
        return pd.Series(self._daily_returns, dtype=float)

    @property
    def config(self) -> PodConfig:
        """Pod 설정."""
        return self._config

    # ── Core Methods ────────────────────────────────────────────────

    def compute_signal(
        self,
        symbol: str,
        bar_data: dict[str, float],
        bar_timestamp: datetime,
    ) -> tuple[int, float] | None:
        """Bar 데이터로 시그널을 계산합니다.

        1. 버퍼에 OHLCV 누적
        2. warmup 미달 시 None 반환
        3. strategy.run() 호출 → (direction, strength) 반환

        Args:
            symbol: 거래 심볼
            bar_data: OHLCV dict (open, high, low, close, volume)
            bar_timestamp: 캔들 타임스탬프

        Returns:
            (direction, strength) 튜플 또는 None (warmup 미충족/에러)
        """
        if not self.accepts_symbol(symbol):
            return None

        # 1. 버퍼에 누적
        if symbol not in self._buffers:
            self._buffers[symbol] = []
            self._timestamps[symbol] = []

        self._buffers[symbol].append(bar_data)
        self._timestamps[symbol].append(bar_timestamp)

        # 1b. 버퍼 크기 제한
        if len(self._buffers[symbol]) > _MAX_BUFFER_SIZE:
            trim = len(self._buffers[symbol]) - _MAX_BUFFER_SIZE
            self._buffers[symbol] = self._buffers[symbol][trim:]
            self._timestamps[symbol] = self._timestamps[symbol][trim:]

        # 2. warmup 체크
        if len(self._buffers[symbol]) < self._warmup:
            return None

        # 3. DataFrame 구성 + strategy.run()
        df = pd.DataFrame(
            self._buffers[symbol],
            index=pd.DatetimeIndex(self._timestamps[symbol], tz=UTC),
        )

        try:
            _, signals = self._strategy.run(df)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Pod {}: strategy.run() failed for {}: {}",
                self.pod_id,
                symbol,
                exc,
            )
            return None

        # 4. 최신 시그널 추출
        direction = int(signals.direction.iloc[-1])
        strength = float(signals.strength.iloc[-1])

        # target_weight 저장
        weight = direction * strength
        if abs(weight) < _MIN_WEIGHT_THRESHOLD:
            weight = 0.0
        self._target_weights[symbol] = weight

        return (direction, strength)

    def get_target_weights(self) -> dict[str, float]:
        """Pod 내부 가중치 반환."""
        return dict(self._target_weights)

    def get_global_weights(self) -> dict[str, float]:
        """전체 포트폴리오 기준 가중치 (internal * capital_fraction)."""
        return {
            sym: w * self._capital_fraction
            for sym, w in self._target_weights.items()
        }

    def update_position(
        self,
        symbol: str,
        fill_qty: float,
        fill_price: float,
        fee: float,
        *,
        is_buy: bool,
    ) -> None:
        """체결 정보로 Pod 포지션을 업데이트합니다.

        Args:
            symbol: 거래 심볼
            fill_qty: 체결 수량
            fill_price: 체결 가격
            fee: 수수료
            is_buy: 매수 여부
        """
        if symbol not in self._positions:
            self._positions[symbol] = PodPosition(
                pod_id=self.pod_id,
                symbol=symbol,
            )

        pos = self._positions[symbol]
        signed_qty = fill_qty if is_buy else -fill_qty
        new_notional = pos.notional_usd + signed_qty * fill_price

        # realized PnL: 기존 포지션과 반대 방향 체결 시
        realized_delta = 0.0
        if not is_buy and pos.notional_usd > 0:
            realized_delta = fill_qty * fill_price - fill_qty * (
                pos.notional_usd / max(abs(pos.notional_usd / fill_price), 1e-12)
            )
        realized_delta -= fee

        self._positions[symbol] = PodPosition(
            pod_id=self.pod_id,
            symbol=symbol,
            target_weight=pos.target_weight,
            global_weight=pos.global_weight,
            notional_usd=new_notional,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl + realized_delta,
        )

        self._performance.trade_count += 1

    def record_daily_return(self, daily_return: float) -> None:
        """일별 수익률을 기록합니다.

        Args:
            daily_return: 일별 수익률
        """
        self._daily_returns.append(daily_return)
        self._performance.live_days = len(self._daily_returns)
        self._performance.last_updated = datetime.now(UTC)

    def inject_warmup(
        self,
        symbol: str,
        bars: list[dict[str, float]],
        timestamps: list[datetime],
    ) -> None:
        """과거 데이터를 버퍼에 주입 (라이브 warmup용).

        Args:
            symbol: 거래 심볼
            bars: OHLCV dict 리스트
            timestamps: bar별 타임스탬프 리스트

        Raises:
            ValueError: bars/timestamps 길이 불일치 또는 이미 데이터 존재
        """
        if len(bars) != len(timestamps):
            msg = f"bars ({len(bars)}) and timestamps ({len(timestamps)}) length mismatch"
            raise ValueError(msg)

        if symbol in self._buffers and len(self._buffers[symbol]) > 0:
            msg = f"Buffer for {symbol} is not empty, cannot inject warmup"
            raise ValueError(msg)

        self._buffers[symbol] = list(bars)
        self._timestamps[symbol] = list(timestamps)

    def accepts_symbol(self, symbol: str) -> bool:
        """이 Pod이 해당 심볼을 처리하는지 여부."""
        return symbol in self._config.symbols

    # ── Private ──────────────────────────────────────────────────────

    def _detect_warmup(self) -> int:
        """전략 설정에서 warmup 기간 자동 감지."""
        config = self._strategy.config
        if config is not None:
            warmup_fn = getattr(config, "warmup_periods", None)
            if warmup_fn is not None:
                return int(warmup_fn())
        return _DEFAULT_WARMUP


# ── Factory ──────────────────────────────────────────────────────


def build_pods(config: OrchestratorConfig) -> list[StrategyPod]:
    """OrchestratorConfig → StrategyPod 리스트 생성.

    registry.get_strategy(name) → cls.from_params(**params) 패턴 사용.

    Args:
        config: OrchestratorConfig

    Returns:
        StrategyPod 리스트
    """
    from src.strategy.registry import get_strategy

    pods: list[StrategyPod] = []
    for pod_cfg in config.pods:
        strategy_cls = get_strategy(pod_cfg.strategy_name)
        strategy = strategy_cls.from_params(**pod_cfg.strategy_params)

        pod = StrategyPod(
            config=pod_cfg,
            strategy=strategy,
            capital_fraction=pod_cfg.initial_fraction,
        )
        pods.append(pod)

    return pods
