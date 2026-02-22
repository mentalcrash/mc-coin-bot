"""EDA RiskManager.

OrderRequestEvent 사전 검증을 수행하고,
시스템 손절 조건 시 CircuitBreakerEvent를 발행합니다.

흐름: PM → OrderRequest(validated=False) → RM 검증 → OrderRequest(validated=True) / OrderRejected

Rules Applied:
    - Pre-trade Risk Check: 레버리지, 포지션 수, 주문 크기 제한
    - System Stop-Loss: peak equity 대비 drawdown 감시
    - PortfolioManagerConfig 재사용: 기존 리스크 파라미터 활용
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    CircuitBreakerEvent,
    EventType,
    OrderRejectedEvent,
    OrderRequestEvent,
)
from src.logging.tracing import component_span_with_context
from src.models.types import Direction

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.portfolio.config import PortfolioManagerConfig

# 기본 제한값 상수
DEFAULT_MAX_OPEN_POSITIONS = 8
DEFAULT_MAX_ORDER_SIZE_USD = 100_000.0


class EDARiskManager:
    """EDA 리스크 매니저.

    Subscribes to: OrderRequestEvent(validated=False), BalanceUpdateEvent
    Publishes: OrderRequestEvent(validated=True), OrderRejectedEvent, CircuitBreakerEvent

    Args:
        config: PortfolioManagerConfig (레버리지, 손절 설정)
        portfolio_manager: PM 참조 (레버리지, 포지션 수 조회)
        max_open_positions: 최대 오픈 포지션 수
        max_order_size_usd: 단일 주문 최대 금액 (USD)
    """

    def __init__(
        self,
        config: PortfolioManagerConfig,
        portfolio_manager: EDAPortfolioManager,
        max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS,
        max_order_size_usd: float = DEFAULT_MAX_ORDER_SIZE_USD,
        *,
        enable_circuit_breaker: bool = True,
    ) -> None:
        self._config = config
        self._pm = portfolio_manager
        self._max_open_positions = max_open_positions
        self._static_max_order_size_usd = max_order_size_usd
        self._use_dynamic_max_order_size = False
        self._bus: EventBus | None = None
        self._enable_circuit_breaker = enable_circuit_breaker

        # Drawdown 추적
        self._peak_equity: float = portfolio_manager.total_equity
        self._circuit_breaker_triggered = False

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록."""
        self._bus = bus
        bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance_update)

    @property
    def peak_equity(self) -> float:
        """Peak equity (고점)."""
        return self._peak_equity

    @property
    def current_drawdown(self) -> float:
        """현재 drawdown 비율 (0.0 ~ 1.0)."""
        return self._compute_drawdown(self._pm.total_equity)

    @property
    def max_order_size_usd(self) -> float:
        """단일 주문 최대 금액 (USD).

        동적 모드: equity * max_leverage_cap. 정적 모드: 생성자 값.
        """
        if self._use_dynamic_max_order_size:
            return self._pm.total_equity * self._config.max_leverage_cap
        return self._static_max_order_size_usd

    def enable_dynamic_max_order_size(self) -> None:
        """동적 max_order_size 모드 활성화.

        equity 변동에 따라 자동 조정. LIVE 모드에서 호출.
        """
        self._use_dynamic_max_order_size = True

    @property
    def is_circuit_breaker_active(self) -> bool:
        """서킷 브레이커 발동 여부."""
        return self._circuit_breaker_triggered

    def sync_peak_equity(self, exchange_equity: float) -> None:
        """LIVE 모드: 거래소 잔고 기준으로 peak_equity 리셋.

        State 복원 후 peak_equity가 이전 세션 값(paper 등)으로 남아
        가짜 drawdown이 발생하는 문제를 방지합니다.

        Args:
            exchange_equity: 거래소 총 equity (USDT)
        """
        self._peak_equity = exchange_equity
        self._circuit_breaker_triggered = False

    def restore_state(self, state: dict[str, object]) -> None:
        """저장된 상태를 복원.

        Args:
            state: StateManager에서 로드한 상태 dict.
                keys: peak_equity, circuit_breaker_triggered
        """
        if "peak_equity" in state:
            self._peak_equity = float(state["peak_equity"])  # type: ignore[arg-type]
        if "circuit_breaker_triggered" in state:
            self._circuit_breaker_triggered = bool(state["circuit_breaker_triggered"])

    async def sync_exchange_equity(self, exchange_equity: float) -> None:
        """거래소 실제 equity로 peak/drawdown 동기화.

        Reconciler가 주기적으로 호출. exchange_equity 기준으로
        peak 갱신 + drawdown 체크 → 즉시 CircuitBreaker 발동.

        Args:
            exchange_equity: 거래소 조회 총 equity (USDT)
        """
        if not self._enable_circuit_breaker or self._circuit_breaker_triggered:
            return
        if self._config.system_stop_loss is None:
            self._peak_equity = max(self._peak_equity, exchange_equity)
            return

        drawdown = self._compute_drawdown(exchange_equity)
        self._peak_equity = max(self._peak_equity, exchange_equity)

        if drawdown >= self._config.system_stop_loss:
            self._circuit_breaker_triggered = True
            logger.critical(
                "EXCHANGE EQUITY CB: drawdown {:.1%} >= stop-loss {:.1%} (exchange_equity=${:.0f})",
                drawdown,
                self._config.system_stop_loss,
                exchange_equity,
            )
            bus = self._bus
            if bus is not None:
                cb_event = CircuitBreakerEvent(
                    reason=(
                        f"Exchange equity drawdown {drawdown:.1%} "
                        f">= stop-loss {self._config.system_stop_loss:.1%}"
                    ),
                    close_all_positions=True,
                    source="RiskManager",
                )
                await bus.publish(cb_event)

    async def _on_order_request(self, event: AnyEvent) -> None:
        """OrderRequestEvent 검증."""
        assert isinstance(event, OrderRequestEvent)
        order = event
        bus = self._bus
        assert bus is not None

        # already validated인 경우 무시 (무한루프 방지)
        if order.validated:
            return

        corr_id = str(order.correlation_id) if order.correlation_id else None
        with component_span_with_context("rm.pre_trade_check", corr_id, {"symbol": order.symbol}):
            await self._on_order_request_inner(order, bus)

    async def _on_order_request_inner(self, order: OrderRequestEvent, bus: EventBus) -> None:
        """_on_order_request 본체 (tracing span 내부)."""
        # 서킷 브레이커 발동 상태에서는 모든 주문 거부
        if self._circuit_breaker_triggered:
            reject = OrderRejectedEvent(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                reason="Circuit breaker active",
                correlation_id=order.correlation_id,
                source="RiskManager",
            )
            await bus.publish(reject)
            return

        # Pre-trade checks
        rejection_reason = self._validate_order(order)
        if rejection_reason is not None:
            reject = OrderRejectedEvent(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                reason=rejection_reason,
                correlation_id=order.correlation_id,
                source="RiskManager",
            )
            await bus.publish(reject)
            logger.warning(
                "Order rejected: {} {} - {}",
                order.symbol,
                order.client_order_id,
                rejection_reason,
            )
            return

        # 검증 통과 → validated=True로 재발행
        logger.debug(
            "Order validated: {} {} notional=${:.2f}",
            order.symbol,
            order.client_order_id,
            order.notional_usd,
        )
        validated_order = OrderRequestEvent(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            target_weight=order.target_weight,
            notional_usd=order.notional_usd,
            price=order.price,
            validated=True,
            correlation_id=order.correlation_id,
            source="RiskManager",
        )
        await bus.publish(validated_order)

    def _validate_order(self, order: OrderRequestEvent) -> str | None:
        """주문 사전 검증.

        Returns:
            거부 사유 (None이면 통과)
        """
        # 1. Aggregate leverage check
        current_leverage = self._pm.aggregate_leverage
        if current_leverage >= self._config.max_leverage_cap and not self._is_reducing_position(
            order
        ):
            return (
                f"Aggregate leverage {current_leverage:.2f} "
                f">= cap {self._config.max_leverage_cap:.2f}"
            )

        # 2. Max open positions check
        open_count = self._pm.open_position_count
        is_new_position = (
            order.symbol not in self._pm.positions or not self._pm.positions[order.symbol].is_open
        )
        if is_new_position and open_count >= self._max_open_positions:
            return f"Max open positions reached ({open_count}/{self._max_open_positions})"

        # 3. Single order size check (포지션 축소 주문은 제외)
        max_size = self.max_order_size_usd
        if order.notional_usd > max_size and not self._is_reducing_position(order):
            return f"Order size ${order.notional_usd:,.0f} > max ${max_size:,.0f}"

        return None

    def _is_reducing_position(self, order: OrderRequestEvent) -> bool:
        """포지션을 축소하는 주문인지 확인."""
        pos = self._pm.positions.get(order.symbol)
        if pos is None or not pos.is_open:
            return False
        # LONG 포지션의 SELL, SHORT 포지션의 BUY → 축소
        return (pos.direction == Direction.LONG and order.side == "SELL") or (
            pos.direction == Direction.SHORT and order.side == "BUY"
        )

    def _compute_drawdown(self, current_equity: float) -> float:
        """주어진 equity 기준 drawdown 계산."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - current_equity / self._peak_equity)

    async def _on_balance_update(self, event: AnyEvent) -> None:
        """BalanceUpdateEvent → drawdown 감시 + stop-loss 체크."""
        assert isinstance(event, BalanceUpdateEvent)
        balance = event
        equity = balance.total_equity

        # Circuit breaker 비활성화 시 peak만 추적
        if not self._enable_circuit_breaker:
            self._peak_equity = max(self._peak_equity, equity)
            return

        # System stop-loss 체크 (peak 업데이트 전에 drawdown 계산)
        if self._config.system_stop_loss is None:
            self._peak_equity = max(self._peak_equity, equity)
            return
        if self._circuit_breaker_triggered:
            return

        drawdown = self._compute_drawdown(equity)

        # Peak equity 업데이트 (drawdown 계산 후)
        self._peak_equity = max(self._peak_equity, equity)

        if drawdown >= self._config.system_stop_loss:
            self._circuit_breaker_triggered = True
            bus = self._bus
            assert bus is not None

            cb_event = CircuitBreakerEvent(
                reason=(
                    f"System stop-loss triggered: "
                    f"drawdown {drawdown:.1%} >= {self._config.system_stop_loss:.1%}"
                ),
                close_all_positions=True,
                correlation_id=event.correlation_id,
                source="RiskManager",
            )
            await bus.publish(cb_event)
            logger.critical(
                "CIRCUIT BREAKER: drawdown {:.1%} >= stop-loss {:.1%}",
                drawdown,
                self._config.system_stop_loss,
            )
