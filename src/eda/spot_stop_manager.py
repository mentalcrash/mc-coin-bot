"""SpotStopManager — Spot STOP_LOSS_LIMIT 안전망 관리.

Spot에서는 STOP_MARKET이 없으므로 STOP_LOSS_LIMIT를 사용합니다.
Futures ExchangeStopManager와 동일한 ratchet 로직이지만:
  - Long-Only (sell side만)
  - 명시적 base_amount (closePosition 없음)
  - limit_price = stop_price * (1 - LIMIT_SLIP_PCT)

Rules Applied:
    - 단일 책임: PM 로직 변경 없음
    - 멱등성: client_order_id prefix "spot-stop-"
    - Ratchet: stop이 올라가는 방향만 허용 (절대 하향 불가)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import AnyEvent, BarEvent, EventType, FillEvent

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.eda.portfolio_manager import EDAPortfolioManager, Position
    from src.exchange.binance_spot_client import BinanceSpotClient
    from src.notification.queue import NotificationQueue
    from src.portfolio.config import PortfolioManagerConfig

# 연속 실패 시 CRITICAL 로그 임계값
_MAX_PLACEMENT_FAILURES = 5

# client_order_id prefix
SPOT_STOP_PREFIX = "spot-stop-"


@dataclass
class SpotStopOrderState:
    """심볼별 Spot stop 주문 상태."""

    symbol: str
    exchange_order_id: str | None
    client_order_id: str
    stop_price: float
    limit_price: float
    base_amount: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    placement_failures: int = 0


class SpotStopManager:
    """Spot STOP_LOSS_LIMIT 주문 lifecycle 관리.

    EventBus를 통해 FILL(진입/청산)과 BAR(가격 업데이트)를 구독하여
    Spot 안전망 stop을 자동 관리합니다.

    Args:
        config: PortfolioManagerConfig (safety net 설정 포함)
        spot_client: BinanceSpotClient (주문 실행)
        pm: EDAPortfolioManager (포지션 조회)
    """

    # stop→limit 슬리피지 허용 비율 (0.5%)
    LIMIT_SLIP_PCT = 0.005

    def __init__(
        self,
        config: PortfolioManagerConfig,
        spot_client: BinanceSpotClient,
        pm: EDAPortfolioManager,
        notification_queue: NotificationQueue | None = None,
    ) -> None:
        self._config = config
        self._client = spot_client
        self._pm = pm
        self._notification_queue: NotificationQueue | None = notification_queue
        self._stops: dict[str, SpotStopOrderState] = {}

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록 (PM/OMS 이후 호출)."""
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.BAR, self._on_bar)
        logger.info(
            "SpotStopManager registered (margin={:.1%}, ts_margin={:.1%})",
            self._config.exchange_safety_margin,
            self._config.exchange_trailing_safety_margin,
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_fill(self, event: AnyEvent) -> None:
        """Fill 이벤트 처리: 진입 → stop 배치, 청산 → stop 취소."""
        assert isinstance(event, FillEvent)
        fill = event
        symbol = fill.symbol
        pos = self._pm.positions.get(symbol)

        if pos is not None and pos.is_open:
            # 포지션이 열려 있음 → stop 배치/갱신
            if symbol not in self._stops:
                await self._place_safety_stop(symbol, pos)
            # 수량 변경 시 stop 재설정 (base_amount 갱신)
            elif self._stops[symbol].base_amount != pos.size:
                await self._cancel_safety_stop(symbol)
                await self._place_safety_stop(symbol, pos)
        elif symbol in self._stops:
            # 포지션 청산됨 → stop 취소
            await self._cancel_safety_stop(symbol)

    async def _on_bar(self, event: AnyEvent) -> None:
        """Bar 이벤트: trailing stop price 업데이트 (ratchet)."""
        assert isinstance(event, BarEvent)
        symbol = event.symbol

        if symbol not in self._stops:
            return

        pos = self._pm.positions.get(symbol)
        if pos is None or not pos.is_open:
            await self._cancel_safety_stop(symbol)
            return

        new_stop_price = self._calculate_stop_price(pos)
        if new_stop_price is None:
            return

        await self._update_stop_if_needed(symbol, new_stop_price, pos.size)

    # =========================================================================
    # Exchange API Operations
    # =========================================================================

    async def _place_safety_stop(self, symbol: str, pos: Position) -> None:
        """포지션에 대한 Spot 안전망 stop 배치."""
        stop_price = self._calculate_stop_price(pos)
        if stop_price is None:
            logger.debug("No safety stop for {} (system_stop_loss=None)", symbol)
            return

        limit_price = stop_price * (1.0 - self.LIMIT_SLIP_PCT)
        base_amount = pos.size
        safe_id = symbol.replace("/", "-")
        client_order_id = f"{SPOT_STOP_PREFIX}{safe_id}_{uuid.uuid4().hex[:8]}"

        try:
            result = await self._client.create_stop_limit_sell(
                symbol=symbol,
                base_amount=base_amount,
                stop_price=stop_price,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
            exchange_order_id = result.get("id")
            self._stops[symbol] = SpotStopOrderState(
                symbol=symbol,
                exchange_order_id=exchange_order_id,
                client_order_id=client_order_id,
                stop_price=stop_price,
                limit_price=limit_price,
                base_amount=base_amount,
            )
            logger.info(
                "Spot safety stop placed: {} sell {:.6f} @ stop={:.2f} limit={:.2f} (order={})",
                symbol,
                base_amount,
                stop_price,
                limit_price,
                exchange_order_id,
            )
        except Exception:
            logger.exception("Failed to place spot safety stop for {}", symbol)
            if symbol in self._stops:
                self._stops[symbol].placement_failures += 1
                await self._check_failure_threshold(symbol)

    async def _update_stop_if_needed(
        self, symbol: str, new_stop_price: float, current_size: float
    ) -> None:
        """0.5%+ 변동 시에만 cancel+create로 업데이트. Ratchet 적용."""
        state = self._stops.get(symbol)
        if state is None:
            return

        current_price = state.stop_price
        threshold = self._config.exchange_stop_update_threshold

        # 변동률 체크
        if current_price > 0:
            change_pct = abs(new_stop_price - current_price) / current_price
            if change_pct < threshold:
                return

        # Ratchet: Long-Only → 올림만 허용
        if new_stop_price < current_price:
            return

        # Cancel + Create
        new_limit = new_stop_price * (1.0 - self.LIMIT_SLIP_PCT)
        try:
            # 기존 stop 취소
            if state.exchange_order_id:
                try:
                    await self._client.cancel_order(state.exchange_order_id, symbol)
                except Exception:
                    logger.warning(
                        "Failed to cancel old spot safety stop for {} (may already be cancelled)",
                        symbol,
                    )
                state.exchange_order_id = None

            # 새 stop 배치
            safe_id = symbol.replace("/", "-")
            client_order_id = f"{SPOT_STOP_PREFIX}{safe_id}_{uuid.uuid4().hex[:8]}"
            result = await self._client.create_stop_limit_sell(
                symbol=symbol,
                base_amount=current_size,
                stop_price=new_stop_price,
                limit_price=new_limit,
                client_order_id=client_order_id,
            )
            state.exchange_order_id = result.get("id")
            state.client_order_id = client_order_id
            state.stop_price = new_stop_price
            state.limit_price = new_limit
            state.base_amount = current_size
            state.last_updated = datetime.now(UTC)
            state.placement_failures = 0

            logger.info(
                "Spot safety stop ratcheted: {} @ {:.2f} → {:.2f}",
                symbol,
                current_price,
                new_stop_price,
            )
            await self._notify_ratchet(symbol, current_price, new_stop_price)
        except Exception:
            logger.exception("Failed to update spot safety stop for {}", symbol)
            state.placement_failures += 1
            await self._check_failure_threshold(symbol)

    async def _cancel_safety_stop(self, symbol: str) -> None:
        """심볼의 안전망 stop 취소."""
        state = self._stops.pop(symbol, None)
        if state is None:
            return

        if state.exchange_order_id:
            try:
                await self._client.cancel_order(state.exchange_order_id, symbol)
                logger.info(
                    "Spot safety stop cancelled: {} (order={})", symbol, state.exchange_order_id
                )
            except Exception:
                logger.warning(
                    "Failed to cancel spot safety stop for {} (order={}) — may already be filled/cancelled",
                    symbol,
                    state.exchange_order_id,
                )

    async def cancel_all_stops(self) -> None:
        """모든 안전망 stop 취소."""
        symbols = list(self._stops.keys())
        for symbol in symbols:
            await self._cancel_safety_stop(symbol)
        logger.info("All spot safety stops cancelled ({} symbols)", len(symbols))

    # =========================================================================
    # Stop Price Calculation
    # =========================================================================

    def _calculate_stop_price(self, pos: Position) -> float | None:
        """포지션에 대한 안전망 stop price 계산.

        SL stop과 TS stop을 각각 계산하고, 더 넓은 쪽(min)을 선택.
        Spot은 Long-Only이므로 항상 LONG 방향.

        Returns:
            안전망 stop price. system_stop_loss=None이면 None.
        """
        sl_stop = self._calculate_sl_stop(pos)
        ts_stop = self._calculate_ts_stop(pos)

        if sl_stop is None and ts_stop is None:
            return None
        if sl_stop is not None and ts_stop is None:
            return sl_stop
        if sl_stop is None and ts_stop is not None:
            return ts_stop

        # 둘 다 있으면: LONG → min (더 넓은 쪽)
        assert sl_stop is not None and ts_stop is not None
        return min(sl_stop, ts_stop)

    def _calculate_sl_stop(self, pos: Position) -> float | None:
        """System Stop Loss 기반 안전망 stop price.

        LONG: entry * (1 - sl - margin)
        """
        sl = self._config.system_stop_loss
        if sl is None:
            return None

        margin = self._config.exchange_safety_margin
        entry = pos.avg_entry_price
        if entry <= 0:
            return None

        return entry * (1.0 - sl - margin)

    def _calculate_ts_stop(self, pos: Position) -> float | None:
        """Trailing Stop 기반 안전망 stop price.

        ATR 미성숙 (14봉 미달) 시 None 반환.
        LONG: (peak - atr * mult) * (1 - ts_margin)
        """
        _atr_period = 14
        if not self._config.use_trailing_stop or len(pos.atr_values) < _atr_period:
            return None

        atr = pos.atr_values[-1]
        if atr <= 0:
            return None

        mult = self._config.trailing_stop_atr_multiplier
        ts_margin = self._config.exchange_trailing_safety_margin

        anchor = pos.peak_price_since_entry
        if anchor <= 0:
            return None

        return (anchor - atr * mult) * (1.0 - ts_margin)

    # =========================================================================
    # State Management
    # =========================================================================

    def get_state(self) -> dict[str, object]:
        """현재 상태를 직렬화 가능한 dict로 반환."""
        stops_data: dict[str, dict[str, object]] = {}
        for symbol, state in self._stops.items():
            stops_data[symbol] = {
                "symbol": state.symbol,
                "exchange_order_id": state.exchange_order_id,
                "client_order_id": state.client_order_id,
                "stop_price": state.stop_price,
                "limit_price": state.limit_price,
                "base_amount": state.base_amount,
                "last_updated": state.last_updated.isoformat(),
                "placement_failures": state.placement_failures,
            }
        return {"stops": stops_data}

    def restore_state(self, state: dict[str, object]) -> None:
        """저장된 상태를 복원."""
        stops_data = state.get("stops", {})
        assert isinstance(stops_data, dict)
        self._stops.clear()
        for symbol, data in stops_data.items():
            assert isinstance(data, dict)
            self._stops[symbol] = SpotStopOrderState(
                symbol=str(data["symbol"]),
                exchange_order_id=data.get("exchange_order_id"),  # type: ignore[arg-type]
                client_order_id=str(data["client_order_id"]),
                stop_price=float(data["stop_price"]),  # type: ignore[arg-type]
                limit_price=float(data["limit_price"]),  # type: ignore[arg-type]
                base_amount=float(data["base_amount"]),  # type: ignore[arg-type]
                last_updated=datetime.fromisoformat(str(data["last_updated"])),
                placement_failures=int(data.get("placement_failures", 0)),  # type: ignore[arg-type]
            )
        if self._stops:
            logger.info("SpotStopManager state restored ({} stops)", len(self._stops))

    @property
    def active_stops(self) -> dict[str, SpotStopOrderState]:
        """현재 활성 stop 주문 (읽기 전용)."""
        return dict(self._stops)

    # =========================================================================
    # Recovery
    # =========================================================================

    async def verify_exchange_stops(self) -> None:
        """재시작 후 거래소 실제 주문 존재 여부 검증.

        restore_state() 후 호출하여 각 심볼의 exchange_order_id가
        거래소에 실제 존재하는지 확인합니다.
        """
        if not self._stops:
            return

        keys_to_remove: list[str] = []
        for symbol, state in self._stops.items():
            if state.exchange_order_id is None:
                keys_to_remove.append(symbol)
                logger.warning(
                    "Spot safety stop for {} has no exchange_order_id — removing stale state",
                    symbol,
                )
                continue

            try:
                open_orders = await self._client.fetch_open_orders(symbol)
                order_ids = {str(o.get("id", "")) for o in open_orders}
                if state.exchange_order_id not in order_ids:
                    keys_to_remove.append(symbol)
                    logger.warning(
                        "Spot safety stop for {} (order={}) not found on exchange — removing",
                        symbol,
                        state.exchange_order_id,
                    )
                else:
                    logger.info(
                        "Spot safety stop for {} verified on exchange (order={})",
                        symbol,
                        state.exchange_order_id,
                    )
            except Exception:
                logger.warning(
                    "Failed to verify spot safety stop for {} — retaining state conservatively",
                    symbol,
                )

        for symbol in keys_to_remove:
            self._stops.pop(symbol, None)

        if keys_to_remove:
            logger.info(
                "Spot safety stop verification: {} stale, {} retained",
                len(keys_to_remove),
                len(self._stops),
            )

    async def place_missing_stops(self) -> int:
        """PM에 열린 포지션이 있지만 stop이 없는 심볼에 안전망 stop 배치.

        Returns:
            배치된 stop 수
        """
        placed = 0
        for pos_key, pos in self._pm.positions.items():
            if not pos.is_open:
                continue
            if pos_key in self._stops:
                continue
            await self._place_safety_stop(pos_key, pos)
            if pos_key in self._stops:
                placed += 1
        if placed:
            logger.info("Placed {} missing spot safety stops", placed)
        return placed

    # =========================================================================
    # Utilities
    # =========================================================================

    async def _notify_ratchet(self, symbol: str, old_stop: float, new_stop: float) -> None:
        """Ratchet 성공 시 INFO 알림 enqueue."""
        if self._notification_queue is None:
            return

        from src.notification.formatters import format_stop_ratchet_embed
        from src.notification.models import ChannelRoute, NotificationItem, Severity

        embed = format_stop_ratchet_embed(symbol, old_stop, new_stop)
        await self._notification_queue.enqueue(
            NotificationItem(
                severity=Severity.INFO,
                channel=ChannelRoute.ALERTS,
                embed=embed,
                spam_key=f"stop_ratchet:{symbol}",
            )
        )

    async def _check_failure_threshold(self, symbol: str) -> None:
        """연속 실패 횟수 임계값 체크 + Discord CRITICAL 알림."""
        state = self._stops.get(symbol)
        if state is not None and state.placement_failures >= _MAX_PLACEMENT_FAILURES:
            logger.critical(
                "SPOT SAFETY STOP FAILURE: {} consecutive failures for {} — safety net may be inactive!",
                state.placement_failures,
                symbol,
            )
            if self._notification_queue is not None:
                from src.notification.formatters import format_safety_stop_failure_embed
                from src.notification.models import ChannelRoute, NotificationItem, Severity

                embed = format_safety_stop_failure_embed(symbol, state.placement_failures)
                await self._notification_queue.enqueue(
                    NotificationItem(
                        severity=Severity.CRITICAL,
                        channel=ChannelRoute.ALERTS,
                        embed=embed,
                    )
                )

    def is_safety_stop_order(self, client_order_id: str) -> bool:
        """client_order_id가 Spot 안전망 stop 주문인지 확인."""
        return client_order_id.startswith(SPOT_STOP_PREFIX)
