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
from src.models.types import Direction

if TYPE_CHECKING:
    from uuid import UUID

    from src.core.event_bus import EventBus
    from src.portfolio.config import PortfolioManagerConfig

# 포지션 크기 0 판정 임계값 (부동소수점 오차 대응)
_POSITION_ZERO_THRESHOLD = 1e-10

# ATR 계산 기간 (14-period SMA of True Range)
_ATR_PERIOD = 14


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
    ) -> None:
        self._config = config
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._asset_weights = asset_weights or {}
        self._order_counter = 0
        self._bus: EventBus | None = None

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

    # =========================================================================
    # Event Handlers
    # =========================================================================
    async def _on_signal(self, event: AnyEvent) -> None:
        """SignalEvent → OrderRequestEvent 변환."""
        assert isinstance(event, SignalEvent)
        signal = event
        bus = self._bus
        assert bus is not None

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

        # 3. 리밸런스 임계값 확인
        pos = self._positions.get(symbol)
        current_weight = pos.current_weight if pos else 0.0

        if not self._config.should_rebalance(current_weight, target_weight):
            return

        # 4. 주문 생성
        self._order_counter += 1
        client_order_id = f"{signal.strategy_name}-{symbol}-{self._order_counter}"

        equity = self.total_equity
        delta_weight = target_weight - current_weight
        notional = abs(delta_weight) * equity
        side: str = "BUY" if delta_weight > 0 else "SELL"

        order = OrderRequestEvent(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            target_weight=target_weight,
            notional_usd=notional,
            correlation_id=signal.correlation_id,
            source="PortfolioManager",
        )
        await bus.publish(order)

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

        # BalanceUpdateEvent 발행
        await self._publish_balance_update(fill.correlation_id)

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
        """BUY fill 처리: 롱 진입/추가 또는 숏 청산."""
        if pos.direction == Direction.SHORT:
            # 숏 포지션 청산/축소
            realized = (pos.avg_entry_price - fill.fill_price) * fill.fill_qty
            pos.realized_pnl += realized
            pos.size -= fill.fill_qty
            if pos.size < _POSITION_ZERO_THRESHOLD:
                pos.size = 0.0
                pos.direction = Direction.NEUTRAL
                pos.avg_entry_price = 0.0
                pos.unrealized_pnl = 0.0
                pos.peak_price_since_entry = 0.0
                pos.trough_price_since_entry = 0.0
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
        """SELL fill 처리: 숏 진입/추가 또는 롱 청산."""
        if pos.direction == Direction.LONG:
            # 롱 포지션 청산/축소
            realized = (fill.fill_price - pos.avg_entry_price) * fill.fill_qty
            pos.realized_pnl += realized
            pos.size -= fill.fill_qty
            if pos.size < _POSITION_ZERO_THRESHOLD:
                pos.size = 0.0
                pos.direction = Direction.NEUTRAL
                pos.avg_entry_price = 0.0
                pos.unrealized_pnl = 0.0
                pos.peak_price_since_entry = 0.0
                pos.trough_price_since_entry = 0.0
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
        """BarEvent → mark-to-market, stop-loss, trailing stop, balance update."""
        assert isinstance(event, BarEvent)
        bar = event
        pos = self._positions.get(bar.symbol)
        if pos is None or not pos.is_open:
            return

        # 1. ATR 업데이트 (last_price 변경 전에 prev_close 사용)
        atr = self._update_atr(pos, bar)

        # 2. Mark-to-market
        pos.last_price = bar.close
        if pos.direction == Direction.LONG:
            pos.unrealized_pnl = (bar.close - pos.avg_entry_price) * pos.size
        elif pos.direction == Direction.SHORT:
            pos.unrealized_pnl = (pos.avg_entry_price - bar.close) * pos.size

        # 3. Position stop-loss 체크
        if self._config.system_stop_loss is not None and self._check_position_stop_loss(pos, bar):
            await self._emit_close_order(pos, bar.correlation_id, "stop-loss")
            await self._publish_balance_update(bar.correlation_id)
            return

        # 4. Trailing stop 체크
        if self._config.use_trailing_stop:
            self._update_peak_trough(pos, bar)
            if atr is not None and self._check_trailing_stop(pos, bar, atr):
                await self._emit_close_order(pos, bar.correlation_id, "trailing-stop")
                await self._publish_balance_update(bar.correlation_id)
                return

        # 5. Weight 업데이트
        equity = self.total_equity
        if equity > 0:
            direction_sign = 1.0 if pos.direction == Direction.LONG else -1.0
            pos.current_weight = direction_sign * pos.notional / equity

        # 6. BalanceUpdateEvent 발행 (RM 실시간 drawdown 추적)
        await self._publish_balance_update(bar.correlation_id)

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
            return bar.close < pos.peak_price_since_entry - trailing_distance
        if pos.direction == Direction.SHORT:
            return bar.close > pos.trough_price_since_entry + trailing_distance
        return False

    async def _emit_close_order(
        self, pos: Position, correlation_id: UUID | None, reason: str
    ) -> None:
        """포지션 청산 주문 발행."""
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
            correlation_id=correlation_id,
            source="PortfolioManager",
        )
        await bus.publish(order)
        logger.info(
            "{} triggered for {} @ {}: entry={:.2f} last={:.2f}",
            reason,
            pos.symbol,
            pos.direction.name,
            pos.avg_entry_price,
            pos.last_price,
        )

    async def _publish_balance_update(self, correlation_id: UUID | None) -> None:
        """BalanceUpdateEvent 발행 (공통 헬퍼)."""
        bus = self._bus
        assert bus is not None

        margin_used = sum(p.notional for p in self._positions.values() if p.is_open)
        bal_event = BalanceUpdateEvent(
            total_equity=self.total_equity,
            available_cash=self._cash,
            total_margin_used=margin_used,
            correlation_id=correlation_id,
            source="PortfolioManager",
        )
        await bus.publish(bal_event)
