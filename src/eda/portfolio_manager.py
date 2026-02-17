"""이벤트 기반 PortfolioManager.

SignalEvent → OrderRequestEvent 변환, FillEvent → 포지션/잔고 업데이트를 담당합니다.
기존 PortfolioManagerConfig의 clamp_leverage(), should_rebalance() 등을 재사용합니다.

Rules Applied:
    - Stateful Execution: 포지션/잔고 상태를 관리
    - PortfolioManagerConfig 재사용: 기존 리스크 가드레일 활용
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    EventType,
    FillEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    SignalEvent,
)
from src.logging.tracing import component_span_with_context
from src.models.types import Direction

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from src.core.event_bus import EventBus
    from src.portfolio.config import PortfolioManagerConfig

# 포지션 크기 0 판정 임계값 (부동소수점 오차 대응)
_POSITION_ZERO_THRESHOLD = 1e-10

# ATR 계산 기간 (14-period SMA of True Range)
_ATR_PERIOD = 14

# Cash safety margin: 총 자본 대비 최소 유지 비율 (음수 방지)
_MIN_CASH_RATIO = 0.001  # 0.1% of initial capital


@dataclass
class Position:
    """심볼별 포지션 상태.

    Attributes:
        symbol: 거래 심볼
        direction: 포지션 방향
        size: 포지션 수량 (절대값)
        avg_entry_price: 평균 진입가
        realized_pnl: 실현 손익 (누적)
        unrealized_pnl: 미실현 손익
        current_weight: 현재 포트폴리오 비중
    """

    symbol: str
    direction: Direction = Direction.NEUTRAL
    size: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    current_weight: float = 0.0
    last_price: float = 0.0
    peak_price_since_entry: float = 0.0
    trough_price_since_entry: float = 0.0
    atr_values: list[float] = field(default_factory=list)

    @property
    def notional(self) -> float:
        """포지션 명목 가치 (USD)."""
        return self.size * self.last_price

    @property
    def is_open(self) -> bool:
        """포지션 존재 여부."""
        return self.size > 0.0


class EDAPortfolioManager:
    """이벤트 기반 포트폴리오 매니저.

    Subscribes to: SignalEvent, FillEvent, BarEvent
    Publishes: OrderRequestEvent, PositionUpdateEvent, BalanceUpdateEvent

    Args:
        config: PortfolioManagerConfig (기존 설정 재사용)
        initial_capital: 초기 자본 (USD)
        asset_weights: 에셋별 가중치 (None이면 균등분배)
    """

    def __init__(
        self,
        config: PortfolioManagerConfig,
        initial_capital: float,
        asset_weights: dict[str, float] | None = None,
        target_timeframe: str = "1D",
    ) -> None:
        self._config = config
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._asset_weights = asset_weights or {}
        self._order_counter = 0
        self._bus: EventBus | None = None
        # Per-bar rebalancing state
        self._last_target_weights: dict[str, float] = {}
        self._stopped_this_bar: set[str] = set()
        self._current_bar_ts: datetime | None = None
        self._target_timeframe = target_timeframe

        # Batch processing (멀티에셋 전용)
        self._batch_mode = len(self._asset_weights) > 1
        self._pending_signals: dict[str, float] = {}  # symbol → target_weight
        self._batch_ts: datetime | None = None
        self._last_executed_targets: dict[str, float] = {}  # VBT parity: 마지막 실행 target
        self._deferred_close_targets: dict[str, float] = {}  # SL/TS deferred close (batch mode)

        # SL/TS pending close: close fill 수신 전까지 해당 심볼 주문 차단
        self._pending_close: set[str] = set()

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록."""
        self._bus = bus
        bus.subscribe(EventType.SIGNAL, self._on_signal)
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.BAR, self._on_bar)

    # =========================================================================
    # Properties (RM에서 읽기 전용 접근)
    # =========================================================================
    @property
    def total_equity(self) -> float:
        """총 자산가치 = 현금 + LONG 명목가치 - SHORT 명목가치.

        LONG: cash에서 진입금액이 빠졌고, 현재 notional이 포지션 가치.
        SHORT: cash에 진입금액이 더해졌고, 청산 시 현재 notional을 반환해야 함.
        """
        long_notional = sum(
            p.notional
            for p in self._positions.values()
            if p.is_open and p.direction == Direction.LONG
        )
        short_notional = sum(
            p.notional
            for p in self._positions.values()
            if p.is_open and p.direction == Direction.SHORT
        )
        return self._cash + long_notional - short_notional

    @property
    def available_cash(self) -> float:
        """가용 현금."""
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """현재 포지션 (읽기 전용)."""
        return self._positions

    @property
    def aggregate_leverage(self) -> float:
        """합산 레버리지 = 총 포지션 / equity."""
        equity = self.total_equity
        if equity <= 0:
            return 0.0
        total_abs_notional = sum(p.notional for p in self._positions.values() if p.is_open)
        return total_abs_notional / equity

    @property
    def open_position_count(self) -> int:
        """오픈 포지션 수."""
        return sum(1 for p in self._positions.values() if p.is_open)

    @property
    def order_counter(self) -> int:
        """주문 카운터 (StateManager 접근용)."""
        return self._order_counter

    @property
    def last_target_weights(self) -> dict[str, float]:
        """마지막 target weights (StateManager 접근용)."""
        return self._last_target_weights

    @property
    def last_executed_targets(self) -> dict[str, float]:
        """마지막 실행된 targets (StateManager 접근용)."""
        return self._last_executed_targets

    def restore_state(self, state: dict[str, object]) -> None:
        """저장된 상태를 복원.

        Args:
            state: StateManager에서 로드한 상태 dict.
                keys: positions, cash, order_counter,
                      last_target_weights, last_executed_targets
        """
        # positions 복원
        positions_data = state.get("positions", {})
        assert isinstance(positions_data, dict)
        self._positions.clear()
        for symbol, pos_data in positions_data.items():
            assert isinstance(pos_data, dict)
            pos = Position(
                symbol=symbol,
                direction=Direction(pos_data["direction"]),
                size=float(pos_data["size"]),
                avg_entry_price=float(pos_data["avg_entry_price"]),
                realized_pnl=float(pos_data.get("realized_pnl", 0.0)),
                unrealized_pnl=float(pos_data.get("unrealized_pnl", 0.0)),
                current_weight=float(pos_data.get("current_weight", 0.0)),
                last_price=float(pos_data.get("last_price", 0.0)),
                peak_price_since_entry=float(pos_data.get("peak_price_since_entry", 0.0)),
                trough_price_since_entry=float(pos_data.get("trough_price_since_entry", 0.0)),
                atr_values=[float(v) for v in pos_data.get("atr_values", [])],
            )
            self._positions[symbol] = pos

        # cash, order_counter 복원
        if "cash" in state:
            self._cash = float(state["cash"])  # type: ignore[arg-type]
        if "order_counter" in state:
            self._order_counter = int(state["order_counter"])  # type: ignore[arg-type]

        # target weights 복원
        ltw = state.get("last_target_weights", {})
        assert isinstance(ltw, dict)
        self._last_target_weights = {k: float(v) for k, v in ltw.items()}

        let = state.get("last_executed_targets", {})
        assert isinstance(let, dict)
        self._last_executed_targets = {k: float(v) for k, v in let.items()}

    def reconcile_with_exchange(
        self,
        exchange_positions: dict[str, tuple[float, Direction]],
    ) -> list[str]:
        """거래소 실제 포지션과 PM 상태를 비교하여 phantom position 제거.

        거래소에 없지만 PM에만 남아있는 포지션(phantom)을 정리합니다.
        거래소에만 있는 포지션은 PM에 추가하지 않습니다 (경고 로그만).

        Args:
            exchange_positions: {symbol: (size, Direction)} 거래소 포지션 맵.
                size=0이면 해당 심볼에 포지션 없음을 의미.

        Returns:
            제거된 심볼 리스트
        """
        removed: list[str] = []

        for symbol, pos in list(self._positions.items()):
            if not pos.is_open:
                continue

            ex_size, _ = exchange_positions.get(symbol, (0.0, Direction.NEUTRAL))
            if ex_size > 0:
                continue

            # Phantom position: 거래소에 없으므로 PM에서 제거
            logger.warning(
                "Reconcile: removing phantom position {} (size={:.6f}, dir={})",
                symbol,
                pos.size,
                pos.direction.name,
            )
            pos.size = 0.0
            pos.direction = Direction.NEUTRAL
            pos.avg_entry_price = 0.0
            pos.unrealized_pnl = 0.0
            pos.current_weight = 0.0
            pos.last_price = 0.0
            pos.peak_price_since_entry = 0.0
            pos.trough_price_since_entry = 0.0
            pos.atr_values = []

            # 부가 상태 정리
            self._last_target_weights.pop(symbol, None)
            self._last_executed_targets.pop(symbol, None)
            self._pending_signals.pop(symbol, None)
            self._pending_close.discard(symbol)
            self._deferred_close_targets.pop(symbol, None)

            removed.append(symbol)

        # 거래소에만 있는 포지션 경고
        pm_symbols = {s for s, p in self._positions.items() if p.is_open}
        for symbol, (ex_size, ex_dir) in exchange_positions.items():
            if ex_size > 0 and symbol not in pm_symbols:
                logger.warning(
                    "Reconcile: exchange-only position {} (size={:.6f}, dir={}) — not added to PM",
                    symbol,
                    ex_size,
                    ex_dir.name,
                )

        if removed:
            logger.info("Reconcile: removed {} phantom positions: {}", len(removed), removed)

        return removed

    def sync_capital(self, exchange_equity: float) -> None:
        """LIVE 모드: 거래소 잔고 기준으로 capital 동기화.

        State 복원 후 PM cash가 이전 세션(paper 등) 값으로 남는 문제 방지.
        포지션 notional을 고려하여 cash를 역산합니다.

        Args:
            exchange_equity: 거래소 총 equity (USDT)
        """
        long_notional = sum(
            p.notional
            for p in self._positions.values()
            if p.is_open and p.direction == Direction.LONG
        )
        short_notional = sum(
            p.notional
            for p in self._positions.values()
            if p.is_open and p.direction == Direction.SHORT
        )
        # equity = cash + long - short → cash = equity - long + short
        self._cash = exchange_equity - long_notional + short_notional
        self._initial_capital = exchange_equity

    # =========================================================================
    # Cash Sufficiency Check
    # =========================================================================
    def _has_sufficient_cash(self, notional: float, side: str) -> bool:
        """주문 실행에 충분한 현금이 있는지 확인.

        레버리지 거래에서 cash는 정상적으로 음수가 될 수 있음
        (BUY notional > initial cash). max_leverage_cap을 고려하여
        cash가 레버리지 한도 이상으로 음수가 되는 것만 방지.

        Args:
            notional: 주문 명목가치 (USD)
            side: "BUY" 또는 "SELL"

        Returns:
            True면 주문 가능
        """
        # 청산(SELL) 방향 주문은 현금 반환 → 항상 허용
        if side == "SELL":
            return True

        # 레버리지 허용 범위 내 현금 하한 계산
        # max_leverage_cap=2 일 때 initial=10000이면 notional 최대 20000
        # → cash 최소 10000 - 20000 = -10000 까지 허용
        leverage_cap = self._config.max_leverage_cap
        max_allowed_negative = self._initial_capital * (leverage_cap - 1.0)
        buffer = self._initial_capital * _MIN_CASH_RATIO
        cash_floor = -(max_allowed_negative + buffer)

        projected_cash = self._cash - notional
        if projected_cash < cash_floor:
            logger.warning(
                "Cash sufficiency check failed: cash={:.2f}, notional={:.2f}, projected={:.2f}, floor={:.2f}",
                self._cash,
                notional,
                projected_cash,
                cash_floor,
            )
            return False
        return True

    # =========================================================================
    # Event Handlers
    # =========================================================================
    async def _on_signal(self, event: AnyEvent) -> None:
        """SignalEvent → target weight 저장 + 리밸런스 평가."""
        assert isinstance(event, SignalEvent)
        signal = event
        corr_id = str(signal.correlation_id) if signal.correlation_id else None
        with component_span_with_context("pm.process_signal", corr_id, {"symbol": signal.symbol}):
            await self._on_signal_inner(signal)

    async def _on_signal_inner(self, signal: SignalEvent) -> None:
        """_on_signal 본체 (tracing span 내부)."""
        symbol = signal.symbol

        # 1. 에셋 가중치 적용
        asset_weight = self._asset_weights.get(symbol, 1.0)
        raw_target = signal.strength * asset_weight

        # 2. 방향 적용 + 레버리지 클램핑
        if signal.direction == Direction.NEUTRAL:
            target_weight = 0.0
        else:
            direction_sign = 1.0 if signal.direction == Direction.LONG else -1.0
            clamped = self._config.clamp_leverage(abs(raw_target))
            target_weight = direction_sign * clamped

        # 3. target weight 저장 (per-bar rebalancing에서도 사용)
        self._last_target_weights[symbol] = target_weight

        # 4. 배치 모드 vs 즉시 모드
        if self._batch_mode:
            # Deferred SL/TS close: target=0.0으로 수집 (fill at next bar's open)
            if symbol in self._deferred_close_targets:
                target_weight = self._deferred_close_targets.pop(symbol)
            elif symbol in self._stopped_this_bar:
                # SL bar의 일반 signal은 수집하지 않음 (VBT parity: SL bar에서 weight=0)
                return
            # 새 timestamp → 이전 배치 flush
            if self._batch_ts is not None and signal.bar_timestamp != self._batch_ts:
                await self._flush_signal_batch()
            self._batch_ts = signal.bar_timestamp
            self._pending_signals[symbol] = target_weight
        else:
            await self._evaluate_rebalance(
                symbol, signal.correlation_id, source_strategy=signal.strategy_name
            )

    async def _flush_signal_batch(self) -> None:
        """수집된 signal을 동일 equity snapshot으로 일괄 주문 생성.

        VBT parity: threshold 비교는 last_executed_target vs new_target
        (VBT의 apply_rebalance_threshold_numba와 동일한 로직).
        """
        if not self._pending_signals:
            return

        bus = self._bus
        assert bus is not None

        # Equity snapshot 1회
        equity = self.total_equity
        if equity <= 0:
            self._pending_signals.clear()
            return
        threshold = self._config.rebalance_threshold

        for symbol, target_weight in self._pending_signals.items():
            # stop-loss 후 재진입 방지 (deferred close target=0.0은 통과)
            if symbol in self._stopped_this_bar and target_weight != 0.0:
                continue

            # VBT parity: last_executed_target vs new_target 비교
            last_executed = self._last_executed_targets.get(symbol, 0.0)
            change = abs(target_weight - last_executed)

            # VBT 동일 로직: threshold 초과 또는 첫 진입 (0→non-zero)
            if change < threshold and not (last_executed == 0.0 and target_weight != 0.0):
                continue

            # 주문 생성 (동일 equity 사용)
            pos = self._positions.get(symbol)
            current_weight = pos.current_weight if pos else 0.0
            self._order_counter += 1
            delta_weight = target_weight - current_weight
            notional = abs(delta_weight) * equity
            side: str = "BUY" if delta_weight > 0 else "SELL"

            # Cash sufficiency pre-trade guard (음수 방지)
            if not self._has_sufficient_cash(notional, side):
                continue

            order = OrderRequestEvent(
                client_order_id=f"batch-{symbol}-{self._order_counter}",
                symbol=symbol,
                side=side,  # type: ignore[arg-type]
                target_weight=target_weight,
                notional_usd=notional,
                correlation_id=None,
                source="PortfolioManager",
            )
            await bus.publish(order)

            # 실행된 target 기록
            self._last_executed_targets[symbol] = target_weight

        self._pending_signals.clear()

    async def flush_pending_signals(self) -> None:
        """마지막 timestamp의 미처리 signal 일괄 처리 (Runner에서 호출)."""
        if self._batch_mode:
            # Remaining deferred closes를 pending에 추가
            for sym, tw in self._deferred_close_targets.items():
                self._pending_signals[sym] = tw
            self._deferred_close_targets.clear()
            await self._flush_signal_batch()

    async def _evaluate_rebalance(
        self,
        symbol: str,
        correlation_id: UUID | None,
        source_strategy: str = "rebalance",
    ) -> None:
        """저장된 target weight와 last executed 비교 → 리밸런스 주문 생성.

        VBT parity: target vs last_executed_target 비교 (weight drift 무시).
        VBT의 apply_rebalance_threshold_numba와 동일한 로직으로,
        신호 변화가 threshold 이상일 때만 리밸런싱합니다.
        """
        bus = self._bus
        assert bus is not None

        # stop-loss 후 같은 bar에서 재진입 방지 / SL/TS close 대기 중 → 재진입 방지
        if symbol in self._stopped_this_bar or symbol in self._pending_close:
            return

        target_weight = self._last_target_weights.get(symbol)
        if target_weight is None:
            return

        # VBT parity: target vs last_executed (not current_weight vs target)
        # VBT의 apply_rebalance_threshold_numba와 동일: 신호 변화 < threshold면 무시
        last_executed = self._last_executed_targets.get(symbol, 0.0)
        change = abs(target_weight - last_executed)
        if change < self._config.rebalance_threshold:
            return

        pos = self._positions.get(symbol)
        current_weight = pos.current_weight if pos else 0.0

        # 주문 생성
        equity = self.total_equity
        if equity <= 0:
            return

        self._order_counter += 1
        client_order_id = f"{source_strategy}-{symbol}-{self._order_counter}"

        delta_weight = target_weight - current_weight
        notional = abs(delta_weight) * equity

        # 최소 주문 크기 필터 (overflow fix로 생긴 tiny position 정리용)
        if notional < 1.0:
            self._last_executed_targets[symbol] = target_weight
            return

        side: str = "BUY" if delta_weight > 0 else "SELL"

        # Cash sufficiency pre-trade guard (음수 방지)
        if not self._has_sufficient_cash(notional, side):
            return

        order = OrderRequestEvent(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            target_weight=target_weight,
            notional_usd=notional,
            correlation_id=correlation_id,
            source="PortfolioManager",
        )
        await bus.publish(order)

        # 실행된 target 기록 (VBT parity)
        self._last_executed_targets[symbol] = target_weight

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent → 포지션/잔고 업데이트."""
        assert isinstance(event, FillEvent)
        fill = event
        bus = self._bus
        assert bus is not None

        symbol = fill.symbol
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        pos = self._positions[symbol]

        fill_notional = fill.fill_price * fill.fill_qty
        is_buy = fill.side == "BUY"

        # 포지션 업데이트 (로직 분리)
        self._apply_fill_to_position(pos, fill, fill_notional, is_buy=is_buy)

        # Close fill → _pending_close 해제 (포지션 종료 확인)
        if not pos.is_open and symbol in self._pending_close:
            self._pending_close.discard(symbol)

        # 현금 업데이트
        if is_buy:
            self._cash -= fill_notional + fill.fee
        else:
            self._cash += fill_notional - fill.fee

        # 가중치 업데이트
        equity = self.total_equity
        if equity > 0:
            direction_sign = 1.0 if pos.direction == Direction.LONG else -1.0
            pos.current_weight = direction_sign * pos.notional / equity if pos.is_open else 0.0

        # PositionUpdateEvent 발행
        pos_event = PositionUpdateEvent(
            symbol=symbol,
            direction=pos.direction,
            size=pos.size,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl,
            correlation_id=fill.correlation_id,
            source="PortfolioManager",
        )
        await bus.publish(pos_event)

        # BalanceUpdateEvent 발행 (fill의 bar timestamp 사용)
        await self._publish_balance_update(fill.correlation_id, timestamp=fill.fill_timestamp)

        logger.debug(
            "Fill processed: {} {} {} qty={:.6f} price={:.2f} fee={:.4f}",
            symbol,
            fill.side,
            pos.direction.name,
            fill.fill_qty,
            fill.fill_price,
            fill.fee,
        )

    # =========================================================================
    # Position Logic Helpers
    # =========================================================================
    def _apply_fill_to_position(
        self,
        pos: Position,
        fill: FillEvent,
        fill_notional: float,
        *,
        is_buy: bool,
    ) -> None:
        """Fill을 포지션에 반영 (진입/청산/반전 처리)."""
        if is_buy:
            self._apply_buy(pos, fill, fill_notional)
        else:
            self._apply_sell(pos, fill, fill_notional)

    def _apply_buy(self, pos: Position, fill: FillEvent, fill_notional: float) -> None:
        """BUY fill 처리: 롱 진입/추가 또는 숏 청산(+flip)."""
        if pos.direction == Direction.SHORT:
            # 숏 포지션 청산/축소 (close_qty = min(fill, size)로 overflow 방지)
            close_qty = min(fill.fill_qty, pos.size)
            realized = (pos.avg_entry_price - fill.fill_price) * close_qty
            pos.realized_pnl += realized
            pos.size -= close_qty
            if pos.size < _POSITION_ZERO_THRESHOLD:
                pos.size = 0.0
                pos.direction = Direction.NEUTRAL
                pos.avg_entry_price = 0.0
                pos.unrealized_pnl = 0.0
                pos.peak_price_since_entry = 0.0
                pos.trough_price_since_entry = 0.0
                # Overflow → 롱 포지션 신규 진입 (close/open 가격 차이로 발생)
                remaining = fill.fill_qty - close_qty
                if remaining > _POSITION_ZERO_THRESHOLD:
                    pos.size = remaining
                    pos.direction = Direction.LONG
                    pos.avg_entry_price = fill.fill_price
                    pos.peak_price_since_entry = fill.fill_price
                    pos.trough_price_since_entry = 0.0
                    pos.atr_values = []
        else:
            # 롱 포지션 신규 진입 또는 추가
            is_new = pos.size <= _POSITION_ZERO_THRESHOLD
            if not is_new:
                # 기존 롱에 추가 (가중 평균 진입가)
                total_cost = pos.avg_entry_price * pos.size + fill_notional
                pos.size += fill.fill_qty
                pos.avg_entry_price = total_cost / pos.size
            else:
                pos.size = fill.fill_qty
                pos.avg_entry_price = fill.fill_price
                pos.peak_price_since_entry = fill.fill_price
                pos.trough_price_since_entry = 0.0
                pos.atr_values = []
            pos.direction = Direction.LONG

        pos.last_price = fill.fill_price

    def _apply_sell(self, pos: Position, fill: FillEvent, fill_notional: float) -> None:
        """SELL fill 처리: 숏 진입/추가 또는 롱 청산(+flip)."""
        if pos.direction == Direction.LONG:
            # 롱 포지션 청산/축소 (close_qty = min(fill, size)로 overflow 방지)
            close_qty = min(fill.fill_qty, pos.size)
            realized = (fill.fill_price - pos.avg_entry_price) * close_qty
            pos.realized_pnl += realized
            pos.size -= close_qty
            if pos.size < _POSITION_ZERO_THRESHOLD:
                pos.size = 0.0
                pos.direction = Direction.NEUTRAL
                pos.avg_entry_price = 0.0
                pos.unrealized_pnl = 0.0
                pos.peak_price_since_entry = 0.0
                pos.trough_price_since_entry = 0.0
                # Overflow → 숏 포지션 신규 진입 (close/open 가격 차이로 발생)
                remaining = fill.fill_qty - close_qty
                if remaining > _POSITION_ZERO_THRESHOLD:
                    pos.size = remaining
                    pos.direction = Direction.SHORT
                    pos.avg_entry_price = fill.fill_price
                    pos.trough_price_since_entry = fill.fill_price
                    pos.peak_price_since_entry = 0.0
                    pos.atr_values = []
        else:
            # 숏 포지션 신규 진입 또는 추가
            is_new = pos.size <= _POSITION_ZERO_THRESHOLD
            if not is_new:
                # 기존 숏에 추가 (가중 평균 진입가)
                total_cost = pos.avg_entry_price * pos.size + fill_notional
                pos.size += fill.fill_qty
                pos.avg_entry_price = total_cost / pos.size
            else:
                pos.size = fill.fill_qty
                pos.avg_entry_price = fill.fill_price
                pos.trough_price_since_entry = fill.fill_price
                pos.peak_price_since_entry = 0.0
                pos.atr_values = []
            pos.direction = Direction.SHORT

        pos.last_price = fill.fill_price

    # =========================================================================
    # Bar Handler
    # =========================================================================
    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent → mark-to-market, stop-loss, trailing stop, per-bar rebalancing."""
        assert isinstance(event, BarEvent)
        bar = event

        # 0. TF bar일 때만 stopped set 리셋 (1m bar에서는 리셋 안 함)
        is_tf_bar = bar.timeframe == self._target_timeframe
        if is_tf_bar and self._current_bar_ts != bar.bar_timestamp:
            self._current_bar_ts = bar.bar_timestamp
            self._stopped_this_bar.clear()

        # 1m bar (intrabar): SL/TS 체크만 수행, rebalancing 생략
        if not is_tf_bar:
            await self._on_intrabar(bar)
            return

        # === TF bar 로직 (기존과 동일) ===
        pos = self._positions.get(bar.symbol)
        if pos is None or not pos.is_open:
            # 포지션 없어도 per-bar rebalancing 체크 (단일에셋 모드만)
            if not self._batch_mode and bar.symbol in self._last_target_weights:
                await self._evaluate_rebalance(bar.symbol, bar.correlation_id)
            return

        # 1. ATR 업데이트 (last_price 변경 전에 prev_close 사용)
        atr = self._update_atr(pos, bar)

        # 2. Mark-to-market
        pos.last_price = bar.close
        if pos.direction == Direction.LONG:
            pos.unrealized_pnl = (bar.close - pos.avg_entry_price) * pos.size
        elif pos.direction == Direction.SHORT:
            pos.unrealized_pnl = (pos.avg_entry_price - bar.close) * pos.size

        # Close 대기 중이면 SL/TS/rebalance 스킵 (mark-to-market만 수행)
        if bar.symbol in self._pending_close:
            await self._publish_balance_update(bar.correlation_id, timestamp=bar.bar_timestamp)
            return

        # 3. Position stop-loss 체크
        if self._config.system_stop_loss is not None and self._check_position_stop_loss(pos, bar):
            await self._handle_stop_trigger(pos, bar, "stop-loss")
            return

        # 4. Trailing stop 체크
        if self._config.use_trailing_stop:
            self._update_peak_trough(pos, bar)
            if atr is not None and self._check_trailing_stop(pos, bar, atr):
                await self._handle_stop_trigger(pos, bar, "trailing-stop")
                return

        # 5. Weight 업데이트
        equity = self.total_equity
        if equity > 0:
            direction_sign = 1.0 if pos.direction == Direction.LONG else -1.0
            pos.current_weight = direction_sign * pos.notional / equity

        # 6. BalanceUpdateEvent 발행 (RM 실시간 drawdown 추적)
        await self._publish_balance_update(bar.correlation_id, timestamp=bar.bar_timestamp)

        # 7. Per-bar rebalancing (단일에셋 모드만 — 배치 모드는 signal에서 처리)
        if not self._batch_mode and bar.symbol in self._last_target_weights:
            await self._evaluate_rebalance(bar.symbol, bar.correlation_id)

    # =========================================================================
    # Intrabar Handler (1m bar → SL/TS only)
    # =========================================================================
    async def _on_intrabar(self, bar: BarEvent) -> None:
        """1m bar에서 intrabar SL/TS만 체크 (rebalancing 생략)."""
        pos = self._positions.get(bar.symbol)
        if pos is None or not pos.is_open:
            return
        # 이미 이번 TF bar에서 SL/TS 발동 → 중복 close 방지
        if bar.symbol in self._stopped_this_bar:
            return
        atr = self._update_atr(pos, bar)
        pos.last_price = bar.close
        if pos.direction == Direction.LONG:
            pos.unrealized_pnl = (bar.close - pos.avg_entry_price) * pos.size
        elif pos.direction == Direction.SHORT:
            pos.unrealized_pnl = (pos.avg_entry_price - bar.close) * pos.size
        # Stop-loss
        if self._config.system_stop_loss is not None and self._check_position_stop_loss(pos, bar):
            self._stopped_this_bar.add(bar.symbol)
            self._last_executed_targets[bar.symbol] = 0.0
            await self._emit_close_order(pos, bar.correlation_id, "stop-loss", fill_price=bar.close)
            return
        # Trailing stop
        if self._config.use_trailing_stop:
            self._update_peak_trough(pos, bar)
            if atr is not None and self._check_trailing_stop(pos, bar, atr):
                self._stopped_this_bar.add(bar.symbol)
                self._last_executed_targets[bar.symbol] = 0.0
                await self._emit_close_order(
                    pos, bar.correlation_id, "trailing-stop", fill_price=bar.close
                )

    # =========================================================================
    # Stop Trigger Handler (batch/non-batch 분기)
    # =========================================================================
    async def _handle_stop_trigger(self, pos: Position, bar: BarEvent, reason: str) -> None:
        """SL/TS 발동 처리.

        Batch mode: deferred close (next bar's open에서 체결, VBT parity).
        Non-batch mode: 즉시 close (bar.close에서 체결).
        """
        self._stopped_this_bar.add(bar.symbol)
        if self._batch_mode:
            # Stale signal 제거 + deferred close 등록
            self._pending_signals.pop(bar.symbol, None)
            self._deferred_close_targets[bar.symbol] = 0.0
            self._last_target_weights[bar.symbol] = 0.0
            logger.info(
                "{} triggered for {} (deferred close): entry={:.2f} last={:.2f}",
                reason,
                pos.symbol,
                pos.avg_entry_price,
                pos.last_price,
            )
        else:
            await self._emit_close_order(pos, bar.correlation_id, reason, fill_price=bar.close)
            self._last_executed_targets[bar.symbol] = 0.0
            await self._publish_balance_update(bar.correlation_id, timestamp=bar.bar_timestamp)

    # =========================================================================
    # Stop-Loss / Trailing Stop Helpers
    # =========================================================================
    def _update_atr(self, pos: Position, bar: BarEvent) -> float | None:
        """True Range를 누적하고 ATR(14) SMA를 반환. 데이터 부족 시 None."""
        prev_close = pos.last_price
        if prev_close <= 0:
            return None
        true_range = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close),
        )
        pos.atr_values.append(true_range)
        if len(pos.atr_values) > _ATR_PERIOD:
            pos.atr_values = pos.atr_values[-_ATR_PERIOD:]
        if len(pos.atr_values) < _ATR_PERIOD:
            return None
        return sum(pos.atr_values) / _ATR_PERIOD

    def _check_position_stop_loss(self, pos: Position, bar: BarEvent) -> bool:
        """포지션 레벨 손절 조건 체크.

        Returns:
            True면 손절 발동 (청산 필요)
        """
        sl = self._config.system_stop_loss
        if sl is None:
            return False

        use_intrabar = self._config.use_intrabar_stop
        if pos.direction == Direction.LONG:
            check_price = bar.low if use_intrabar else bar.close
            return check_price < pos.avg_entry_price * (1 - sl)
        if pos.direction == Direction.SHORT:
            check_price = bar.high if use_intrabar else bar.close
            return check_price > pos.avg_entry_price * (1 + sl)
        return False

    def _update_peak_trough(self, pos: Position, bar: BarEvent) -> None:
        """진입 후 최고가/최저가 갱신."""
        if pos.direction == Direction.LONG:
            pos.peak_price_since_entry = max(pos.peak_price_since_entry, bar.high)
        elif pos.direction == Direction.SHORT:
            if pos.trough_price_since_entry <= 0:
                pos.trough_price_since_entry = bar.low
            else:
                pos.trough_price_since_entry = min(pos.trough_price_since_entry, bar.low)

    def _check_trailing_stop(self, pos: Position, bar: BarEvent, atr: float) -> bool:
        """Trailing stop 조건 체크.

        Returns:
            True면 trailing stop 발동 (청산 필요)
        """
        mult = self._config.trailing_stop_atr_multiplier
        trailing_distance = atr * mult

        if pos.direction == Direction.LONG:
            return bar.low < pos.peak_price_since_entry - trailing_distance
        if pos.direction == Direction.SHORT:
            return bar.high > pos.trough_price_since_entry + trailing_distance
        return False

    async def _emit_close_order(
        self,
        pos: Position,
        correlation_id: UUID | None,
        reason: str,
        fill_price: float | None = None,
    ) -> None:
        """포지션 청산 주문 발행.

        Args:
            pos: 청산할 포지션
            correlation_id: 이벤트 추적 ID
            reason: 청산 사유 ("stop-loss", "trailing-stop")
            fill_price: 체결 가격 (SL/TS: bar.close, None이면 executor 기본값)
        """
        bus = self._bus
        assert bus is not None

        self._order_counter += 1
        side = "SELL" if pos.direction == Direction.LONG else "BUY"
        client_order_id = f"{reason}-{pos.symbol}-{self._order_counter}"

        order = OrderRequestEvent(
            client_order_id=client_order_id,
            symbol=pos.symbol,
            side=side,  # type: ignore[arg-type]
            target_weight=0.0,
            notional_usd=pos.notional,
            price=fill_price,
            correlation_id=correlation_id,
            source="PortfolioManager",
        )
        await bus.publish(order)

        # Close fill 수신 전까지 해당 심볼 주문 차단
        self._pending_close.add(pos.symbol)

        # Batch mode: 이전 batch의 stale signal 제거 + last_executed 리셋
        if self._batch_mode:
            self._pending_signals.pop(pos.symbol, None)
            self._last_executed_targets[pos.symbol] = 0.0
        logger.info(
            "{} triggered for {} @ {}: entry={:.2f} last={:.2f}",
            reason,
            pos.symbol,
            pos.direction.name,
            pos.avg_entry_price,
            pos.last_price,
        )

    async def _publish_balance_update(
        self, correlation_id: UUID | None, timestamp: datetime | None = None
    ) -> None:
        """BalanceUpdateEvent 발행 (공통 헬퍼).

        Args:
            correlation_id: 이벤트 추적 ID
            timestamp: bar/fill 시각 (None이면 현재 시각 사용)
        """
        bus = self._bus
        assert bus is not None

        margin_used = sum(p.notional for p in self._positions.values() if p.is_open)
        kwargs: dict[str, object] = {
            "total_equity": self.total_equity,
            "available_cash": self._cash,
            "total_margin_used": margin_used,
            "correlation_id": correlation_id,
            "source": "PortfolioManager",
        }
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        bal_event = BalanceUpdateEvent(**kwargs)  # type: ignore[arg-type]
        await bus.publish(bal_event)
