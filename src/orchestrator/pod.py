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
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from src.orchestrator.asset_allocator import IntraPodAllocator
from src.orchestrator.asset_selector import AssetSelector
from src.orchestrator.models import LifecycleState, PodPerformance, PodPosition

if TYPE_CHECKING:
    from src.orchestrator.config import OrchestratorConfig, PodConfig
    from src.strategy.base import BaseStrategy

# ── Constants ─────────────────────────────────────────────────────

_MAX_BUFFER_SIZE = 1000
_MIN_WEIGHT_THRESHOLD = 1e-8
_DEFAULT_WARMUP = 50
_PERIODS_PER_YEAR = 365
_EPSILON = 1e-12
_MIN_METRICS_SAMPLES = 2
_MIN_RETURN_BARS = 3  # return 계산에 필요한 최소 버퍼 크기
_ROLLING_WINDOW = 30  # rolling Sharpe/DD 계산 윈도우 (일)


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

        # pause 상태
        self._paused: bool = False

        # MTM equity 추적 (daily return 계산용)
        self._base_equity: float = 0.0
        self._prev_equity: float = 0.0

        # Intra-pod asset allocation
        self._asset_allocator: IntraPodAllocator | None = None
        self._asset_returns: dict[str, list[float]] = {}
        self._prev_closes: dict[str, float] = {}
        if config.asset_allocation is not None:
            self._asset_allocator = IntraPodAllocator(
                config=config.asset_allocation,
                symbols=config.symbols,
            )

        # Asset selector (WHO participates)
        self._asset_selector: AssetSelector | None = None
        if config.asset_selector is not None and config.asset_selector.enabled:
            self._asset_selector = AssetSelector(
                config=config.asset_selector,
                symbols=config.symbols,
            )

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
    def timeframe(self) -> str:
        """Pod 타임프레임."""
        return self._config.timeframe

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
    def paused(self) -> bool:
        """일시 중지 상태."""
        return self._paused

    def pause(self) -> None:
        """Pod를 일시 중지합니다."""
        self._paused = True
        logger.info("Pod {} paused", self.pod_id)

    def resume(self) -> None:
        """Pod 일시 중지를 해제합니다."""
        self._paused = False
        logger.info("Pod {} resumed", self.pod_id)

    @property
    def is_active(self) -> bool:
        """RETIRED가 아니고 paused가 아니면 활성."""
        return not self._paused and self._state != LifecycleState.RETIRED

    @property
    def should_emit_signals(self) -> bool:
        """시그널 발행 가능 여부 (active + 에셋 존재)."""
        if not self.is_active:
            return False
        return not (
            self._asset_selector is not None and self._asset_selector.all_excluded
        )

    @property
    def warmup_periods(self) -> int:
        """워밍업 기간."""
        return self._warmup

    @property
    def daily_returns(self) -> list[float]:
        """일별 수익률 리스트 (읽기 전용 뷰)."""
        return self._daily_returns

    @property
    def daily_returns_series(self) -> pd.Series:
        """Allocator용 일별 수익률 Series."""
        return pd.Series(self._daily_returns, dtype=float)

    @property
    def config(self) -> PodConfig:
        """Pod 설정."""
        return self._config

    @property
    def asset_selector(self) -> AssetSelector | None:
        """에셋 선별 FSM (None=비활성)."""
        return self._asset_selector

    @property
    def rolling_sharpe(self) -> float:
        """최근 30일 rolling Sharpe ratio (rf=0, annualized)."""
        window = self._daily_returns[-_ROLLING_WINDOW:]
        n = len(window)
        if n < _MIN_METRICS_SAMPLES:
            return 0.0
        mean_r = sum(window) / n
        var_r = sum((r - mean_r) ** 2 for r in window) / (n - 1)
        vol = var_r**0.5
        annual_vol = vol * (_PERIODS_PER_YEAR**0.5)
        if annual_vol < _EPSILON:
            return 0.0
        return (mean_r * _PERIODS_PER_YEAR) / annual_vol

    @property
    def rolling_drawdown(self) -> float:
        """최근 30일 rolling max drawdown (양수 표현)."""
        window = self._daily_returns[-_ROLLING_WINDOW:]
        if not window:
            return 0.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in window:
            equity *= 1.0 + r
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

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

        # 3. DataFrame 구성 + strategy.run_incremental()
        df = pd.DataFrame(
            self._buffers[symbol],
            index=pd.DatetimeIndex(self._timestamps[symbol], tz=UTC),
        )

        try:
            _, signals = self._strategy.run_incremental(df)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Pod {}: strategy.run_incremental() failed for {}: {}",
                self.pod_id,
                symbol,
                exc,
            )
            return None

        # 4. 최신 시그널 추출
        direction = int(signals.direction.iloc[-1])
        strength = float(signals.strength.iloc[-1])

        # 4b. Layer 1 leverage cap: per-symbol strength 상한
        strength = min(strength, self._config.max_leverage)

        # 5. Intra-pod asset allocation
        self._update_asset_returns(symbol, self._buffers[symbol])
        self._maybe_rebalance_assets(symbol, strength)
        adjusted_strength = self._apply_asset_weight(symbol, strength)

        # target_weight 저장
        weight = direction * adjusted_strength
        if abs(weight) < _MIN_WEIGHT_THRESHOLD:
            weight = 0.0
        self._target_weights[symbol] = weight

        return (direction, adjusted_strength)

    def get_target_weights(self) -> dict[str, float]:
        """Pod 내부 가중치 반환."""
        return dict(self._target_weights)

    def get_global_weights(self) -> dict[str, float]:
        """전체 포트폴리오 기준 가중치 (internal * capital_fraction)."""
        return {sym: w * self._capital_fraction for sym, w in self._target_weights.items()}

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

        Signed quantity 기반 가중 평균 진입가 추적으로 정확한 realized PnL 계산.

        Args:
            symbol: 거래 심볼
            fill_qty: 체결 수량 (양수)
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
        old_qty = pos.quantity
        signed_fill = fill_qty if is_buy else -fill_qty

        # Realized PnL: 기존 포지션과 반대 방향 체결 시
        realized_delta = self._compute_realized_pnl(
            old_qty, signed_fill, fill_price, pos.avg_entry_price
        )
        realized_delta -= fee

        # 새 quantity / avg_entry_price 계산
        new_qty = old_qty + signed_fill
        new_avg = self._compute_new_avg_entry(
            old_qty,
            pos.avg_entry_price,
            signed_fill,
            fill_price,
            new_qty,
        )
        new_notional = abs(new_qty) * fill_price

        self._positions[symbol] = PodPosition(
            pod_id=self.pod_id,
            symbol=symbol,
            target_weight=pos.target_weight,
            global_weight=pos.global_weight,
            notional_usd=new_notional,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl + realized_delta,
            avg_entry_price=new_avg,
            quantity=new_qty,
        )

        self._performance.trade_count += 1

    @staticmethod
    def _compute_realized_pnl(
        old_qty: float,
        signed_fill: float,
        fill_price: float,
        avg_entry: float,
    ) -> float:
        """기존 포지션과 반대 방향 체결 시 realized PnL 계산."""
        if abs(old_qty) < _EPSILON or avg_entry < _EPSILON:
            return 0.0

        # 같은 방향 추가 → realized 없음
        if (old_qty > 0 and signed_fill > 0) or (old_qty < 0 and signed_fill < 0):
            return 0.0

        close_qty = min(abs(signed_fill), abs(old_qty))
        if old_qty > 0:
            # Long close: (fill_price - avg_entry) * close_qty
            return (fill_price - avg_entry) * close_qty
        # Short cover: (avg_entry - fill_price) * close_qty
        return (avg_entry - fill_price) * close_qty

    @staticmethod
    def _compute_new_avg_entry(
        old_qty: float,
        old_avg: float,
        signed_fill: float,
        fill_price: float,
        new_qty: float,
    ) -> float:
        """새로운 가중 평균 진입가 계산."""
        # 완전 청산
        if abs(new_qty) < _EPSILON:
            return 0.0

        # 신규 진입 (기존 포지션 없음)
        if abs(old_qty) < _EPSILON:
            return fill_price

        # 방향 전환 (flip)
        if (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
            return fill_price

        # 같은 방향 추가: 가중 평균
        if (old_qty > 0 and signed_fill > 0) or (old_qty < 0 and signed_fill < 0):
            old_cost = abs(old_qty) * old_avg
            new_cost = abs(signed_fill) * fill_price
            return (old_cost + new_cost) / abs(new_qty)

        # 부분 청산: avg_entry 유지
        return old_avg

    def record_daily_return(self, daily_return: float) -> None:
        """일별 수익률을 기록합니다.

        Args:
            daily_return: 일별 수익률
        """
        self._daily_returns.append(daily_return)
        self._performance.live_days = len(self._daily_returns)
        self._performance.last_updated = datetime.now(UTC)
        self._compute_metrics()

    # ── MTM Equity Methods ─────────────────────────────────────────

    def set_base_equity(self, base: float) -> None:
        """초기 equity를 설정합니다 (initial_capital * capital_fraction).

        이미 복원된 base_equity가 있으면 스킵합니다.

        Args:
            base: 초기 equity
        """
        if self._base_equity > _EPSILON:
            return  # 이미 복원됨
        self._base_equity = base
        self._prev_equity = base

    def _compute_total_pnl(self, close_prices: dict[str, float]) -> float:
        """전 포지션의 MTM PnL을 합산합니다.

        Args:
            close_prices: 심볼별 최신 close price

        Returns:
            realized + unrealized PnL 합계
        """
        total = 0.0
        for symbol, pos in self._positions.items():
            price = close_prices.get(symbol)
            if price is None or abs(pos.quantity) < _EPSILON:
                total += pos.realized_pnl
                continue
            unrealized = (price - pos.avg_entry_price) * pos.quantity
            total += pos.realized_pnl + unrealized
        return total

    def compute_mtm_equity(self, close_prices: dict[str, float]) -> float:
        """Mark-to-Market equity를 계산합니다.

        Args:
            close_prices: 심볼별 최신 close price

        Returns:
            base_equity + total_pnl
        """
        return self._base_equity + self._compute_total_pnl(close_prices)

    def record_daily_return_mtm(self, close_prices: dict[str, float]) -> None:
        """MTM equity 변화로 일별 수익률을 계산·기록합니다.

        prev_equity가 0에 가까우면 스킵 (미초기화 상태).

        Args:
            close_prices: 심볼별 전일 close price
        """
        if self._prev_equity < _EPSILON:
            return  # 미초기화 상태

        current_equity = self.compute_mtm_equity(close_prices)
        daily_return = (current_equity - self._prev_equity) / self._prev_equity
        self.record_daily_return(daily_return)
        self._prev_equity = current_equity

    def adjust_base_equity_on_rebalance(
        self,
        new_base: float,
        close_prices: dict[str, float],
    ) -> None:
        """리밸런스 시 equity 연속성을 보장합니다.

        base_equity를 변경하되, PnL을 반영하여 prev_equity를 조정합니다.
        이를 통해 리밸런스가 daily return에 spike를 만들지 않습니다.

        Args:
            new_base: 새 base equity (initial_capital * new_weight)
            close_prices: 심볼별 최신 close price
        """
        pnl = self._compute_total_pnl(close_prices)
        self._base_equity = new_base
        self._prev_equity = new_base + pnl

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

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize Pod state for persistence.

        Excludes _buffers, _timestamps (REST warmup), and _daily_returns
        (stored separately in orchestrator_daily_returns key).
        """
        positions_data: dict[str, dict[str, float]] = {}
        for symbol, pos in self._positions.items():
            positions_data[symbol] = {
                "notional_usd": pos.notional_usd,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "target_weight": pos.target_weight,
                "global_weight": pos.global_weight,
                "avg_entry_price": pos.avg_entry_price,
                "quantity": pos.quantity,
            }

        perf = self._performance
        performance_data: dict[str, object] = {
            "total_return": perf.total_return,
            "sharpe_ratio": perf.sharpe_ratio,
            "max_drawdown": perf.max_drawdown,
            "calmar_ratio": perf.calmar_ratio,
            "win_rate": perf.win_rate,
            "trade_count": perf.trade_count,
            "live_days": perf.live_days,
            "rolling_volatility": perf.rolling_volatility,
            "peak_equity": perf.peak_equity,
            "current_equity": perf.current_equity,
            "current_drawdown": perf.current_drawdown,
            "last_updated": perf.last_updated.isoformat(),
        }

        # Asset allocator state
        asset_allocator_data: dict[str, object] | None = None
        if self._asset_allocator is not None:
            asset_allocator_data = self._asset_allocator.to_dict()

        # Asset selector state
        asset_selector_data: dict[str, object] | None = None
        if self._asset_selector is not None:
            asset_selector_data = self._asset_selector.to_dict()

        return {
            "state": self._state.value,
            "capital_fraction": self._capital_fraction,
            "target_weights": dict(self._target_weights),
            "positions": positions_data,
            "performance": performance_data,
            "paused": self._paused,
            "base_equity": self._base_equity,
            "prev_equity": self._prev_equity,
            "asset_allocator": asset_allocator_data,
            "asset_selector": asset_selector_data,
            "asset_returns": {s: list(r) for s, r in self._asset_returns.items()},
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore Pod state from persisted dict.

        Uses defensive .get() with defaults for forward-compatibility.
        """
        # State
        state_val = data.get("state")
        if isinstance(state_val, str):
            self._state = LifecycleState(state_val)

        # Capital fraction
        fraction_val = data.get("capital_fraction")
        if isinstance(fraction_val, int | float):
            self._capital_fraction = float(fraction_val)

        # Paused
        paused_val = data.get("paused")
        if isinstance(paused_val, bool):
            self._paused = paused_val

        # MTM equity
        base_eq = data.get("base_equity")
        if isinstance(base_eq, int | float):
            self._base_equity = float(base_eq)
        prev_eq = data.get("prev_equity")
        if isinstance(prev_eq, int | float):
            self._prev_equity = float(prev_eq)

        # Target weights
        weights_val = data.get("target_weights")
        if isinstance(weights_val, dict):
            self._target_weights = {str(k): float(v) for k, v in weights_val.items()}

        # Positions
        positions_val = data.get("positions")
        if isinstance(positions_val, dict):
            self._positions = {}
            for symbol, pos_data in positions_val.items():
                if isinstance(pos_data, dict):
                    self._positions[str(symbol)] = PodPosition(
                        pod_id=self.pod_id,
                        symbol=str(symbol),
                        notional_usd=float(pos_data.get("notional_usd", 0.0)),
                        unrealized_pnl=float(pos_data.get("unrealized_pnl", 0.0)),
                        realized_pnl=float(pos_data.get("realized_pnl", 0.0)),
                        target_weight=float(pos_data.get("target_weight", 0.0)),
                        global_weight=float(pos_data.get("global_weight", 0.0)),
                        avg_entry_price=float(pos_data.get("avg_entry_price", 0.0)),
                        quantity=float(pos_data.get("quantity", 0.0)),
                    )

        # Performance
        perf_val = data.get("performance")
        if isinstance(perf_val, dict):
            self._restore_performance(perf_val)

        self._restore_allocators_and_returns(data)

    def _restore_allocators_and_returns(self, data: dict[str, object]) -> None:
        """Restore asset allocator, selector, and returns (PLR0912 sub-method)."""
        # Asset allocator
        alloc_val = data.get("asset_allocator")
        if isinstance(alloc_val, dict) and self._asset_allocator is not None:
            self._asset_allocator.restore_from_dict(alloc_val)

        # Asset selector
        selector_val = data.get("asset_selector")
        if isinstance(selector_val, dict) and self._asset_selector is not None:
            self._asset_selector.restore_from_dict(selector_val)

        # Asset returns
        returns_val = data.get("asset_returns")
        if isinstance(returns_val, dict):
            self._asset_returns = {
                str(k): [float(x) for x in v] for k, v in returns_val.items() if isinstance(v, list)
            }

    def restore_daily_returns(self, returns: list[float]) -> None:
        """Restore daily_returns and sync live_days (persistence용)."""
        self._daily_returns = list(returns)
        self._performance.live_days = len(self._daily_returns)

    def _restore_performance(self, data: dict[str, Any]) -> None:
        """Restore PodPerformance fields (PLR0912 sub-method)."""
        perf = self._performance
        perf.total_return = float(data.get("total_return", 0.0))
        perf.sharpe_ratio = float(data.get("sharpe_ratio", 0.0))
        perf.max_drawdown = float(data.get("max_drawdown", 0.0))
        perf.calmar_ratio = float(data.get("calmar_ratio", 0.0))
        perf.win_rate = float(data.get("win_rate", 0.0))
        perf.trade_count = int(data.get("trade_count", 0))
        perf.live_days = int(data.get("live_days", 0))
        perf.rolling_volatility = float(data.get("rolling_volatility", 0.0))
        perf.peak_equity = float(data.get("peak_equity", 0.0))
        perf.current_equity = float(data.get("current_equity", 0.0))
        perf.current_drawdown = float(data.get("current_drawdown", 0.0))
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            perf.last_updated = datetime.fromisoformat(last_updated)

    # ── Private ──────────────────────────────────────────────────────

    def _compute_metrics(self) -> None:
        """daily_returns 기반으로 PodPerformance 메트릭을 재계산합니다.

        순수 Python 구현 (의존성 최소화).
        n < 2이면 early return (분산 계산 불가).
        """
        returns = self._daily_returns
        n = len(returns)
        perf = self._performance

        if n < _MIN_METRICS_SAMPLES:
            if n >= 1:
                perf.total_return = returns[0]
            return

        # total_return: prod(1+r) - 1
        cum = 1.0
        for r in returns:
            cum *= 1.0 + r
        perf.total_return = cum - 1.0

        # equity curve → peak / drawdown
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= 1.0 + r
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        perf.current_equity = equity
        perf.peak_equity = peak
        perf.current_drawdown = (peak - equity) / peak if peak > 0 else 0.0
        perf.max_drawdown = max_dd

        # mean / variance
        mean_r = sum(returns) / n
        var_r = sum((r - mean_r) ** 2 for r in returns) / (n - 1)

        vol = var_r**0.5
        annual_vol = vol * (_PERIODS_PER_YEAR**0.5)
        perf.rolling_volatility = annual_vol

        # sharpe_ratio (rf=0, annualized)
        if annual_vol > _EPSILON:
            perf.sharpe_ratio = (mean_r * _PERIODS_PER_YEAR) / annual_vol
        else:
            perf.sharpe_ratio = 0.0

        # calmar_ratio
        if max_dd > _EPSILON:
            perf.calmar_ratio = (mean_r * _PERIODS_PER_YEAR) / max_dd
        else:
            perf.calmar_ratio = 0.0

        # win_rate
        positive = sum(1 for r in returns if r > 0)
        perf.win_rate = positive / n

    def _update_asset_returns(
        self,
        symbol: str,
        buf: list[dict[str, float]],
    ) -> None:
        """수익률 히스토리 업데이트 (look-ahead bias 방지).

        buf[-1] = 현재 bar, buf[-2] = 이전 bar.
        return = buf[-2].close / buf[-3].close - 1 (현재 bar의 close 미사용).
        """
        if len(buf) < _MIN_RETURN_BARS:
            return
        prev_close = buf[-3].get("close", 0.0)
        curr_close = buf[-2].get("close", 0.0)
        if prev_close > _EPSILON:
            ret = curr_close / prev_close - 1.0
            if symbol not in self._asset_returns:
                self._asset_returns[symbol] = []
            self._asset_returns[symbol].append(ret)

    def _maybe_rebalance_assets(
        self,
        symbol: str,
        strength: float,
    ) -> None:
        """AssetSelector 평가 후 allocator를 호출하여 에셋 비중 재계산."""
        # AssetSelector 평가 (매 bar, allocator보다 먼저)
        if self._asset_selector is not None:
            close_prices = {s: buf[-1]["close"] for s, buf in self._buffers.items() if buf}
            self._asset_selector.on_bar(
                returns=self._asset_returns,
                close_prices=close_prices,
            )

        if self._asset_allocator is None:
            return

        # Active symbols만 allocator에 전달
        if self._asset_selector is not None:
            active_set = set(self._asset_selector.active_symbols)
            filtered_returns = {s: r for s, r in self._asset_returns.items() if s in active_set}
        else:
            filtered_returns = self._asset_returns

        strengths = dict.fromkeys(self._config.symbols, 0.0)
        strengths[symbol] = strength

        self._asset_allocator.on_bar(
            returns=filtered_returns,
            strengths=strengths,
        )

    def _apply_asset_weight(self, symbol: str, strength: float) -> float:
        """에셋별 비중을 strength에 적용 (1/N 정규화 포함).

        Pod 내 심볼 수(N)로 나누어 전체 합이 strength를 초과하지 않도록 보장.
        regular EDA의 asset_weights=1/N 패턴과 동일한 결과.

        (1) AssetSelector multiplier (WHO) — 제외된 에셋은 0
        (2) 1/N 정규화 또는 IntraPodAllocator weight (HOW MUCH)
        """
        n = len(self._config.symbols)

        # (1) AssetSelector multiplier
        selector_mult = 1.0
        if self._asset_selector is not None:
            selector_mult = self._asset_selector.multipliers.get(symbol, 1.0)
        if selector_mult < _MIN_WEIGHT_THRESHOLD:
            return 0.0  # 제외된 에셋

        # (2) No allocator → equal weight (1/N)
        if self._asset_allocator is None:
            return strength * selector_mult / n

        # (3) With allocator → allocator weights (sum~=1.0, no *N)
        weights = self._asset_allocator.weights
        asset_w = weights.get(symbol, 1.0 / n)
        return strength * asset_w * selector_mult

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
