"""ExchangeStopManager — 거래소 STOP_MARKET 안전망 관리.

소프트웨어 SL/TS를 유지하면서, 거래소 STOP_MARKET 주문을 안전망으로 추가합니다.
봇 장애 시에만 거래소 주문이 발동하도록 SW SL보다 약간 넓게 설정합니다.

정상: 1m bar → SW SL 발동 → Market close → Exchange stop 취소
장애: 봇 다운 → 가격 급락 → Exchange STOP_MARKET 발동 → 보호

Rules Applied:
    - 단일 책임: PM 로직 변경 없음
    - 멱등성: client_order_id prefix "safety-stop-"
    - Ratchet: 안전망이 좁아지는 방향만 허용 (LONG up, SHORT down)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import AnyEvent, BarEvent, EventType, FillEvent
from src.models.types import Direction

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.eda.portfolio_manager import EDAPortfolioManager, Position
    from src.exchange.binance_futures_client import BinanceFuturesClient
    from src.notification.queue import NotificationQueue
    from src.portfolio.config import PortfolioManagerConfig

# 연속 실패 시 CRITICAL 로그 임계값
_MAX_PLACEMENT_FAILURES = 5

# client_order_id prefix
SAFETY_STOP_PREFIX = "safety-stop-"


@dataclass
class StopOrderState:
    """심볼별 거래소 stop 주문 상태."""

    symbol: str
    exchange_order_id: str | None
    client_order_id: str
    stop_price: float
    position_side: str  # "LONG" / "SHORT"
    close_side: str  # "sell" (LONG), "buy" (SHORT)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    placement_failures: int = 0


class ExchangeStopManager:
    """거래소 STOP_MARKET 주문 lifecycle 관리.

    EventBus를 통해 FILL(진입/청산)과 BAR(가격 업데이트)를 구독하여
    거래소 안전망 stop을 자동 관리합니다.

    Args:
        config: PortfolioManagerConfig (safety net 설정 포함)
        futures_client: BinanceFuturesClient (주문 실행)
        pm: EDAPortfolioManager (포지션 조회)
    """

    def __init__(
        self,
        config: PortfolioManagerConfig,
        futures_client: BinanceFuturesClient,
        pm: EDAPortfolioManager,
        notification_queue: NotificationQueue | None = None,
        *,
        hedge_mode: bool = False,
    ) -> None:
        self._config = config
        self._client = futures_client
        self._pm = pm
        self._notification_queue: NotificationQueue | None = notification_queue
        self._hedge_mode = hedge_mode
        # key: symbol (one-way) or composite key (hedge)
        self._stops: dict[str, StopOrderState] = {}

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록 (PM/OMS 이후 호출)."""
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.BAR, self._on_bar)
        logger.info(
            "ExchangeStopManager registered (margin={:.1%}, ts_margin={:.1%})",
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

        # Hedge mode: composite key, One-way: plain symbol
        stop_key = self._pm._pos_key(fill.symbol, fill.pod_id) if self._hedge_mode else fill.symbol  # pyright: ignore[reportPrivateUsage]
        pos = self._pm.positions.get(stop_key)

        if pos is not None and pos.is_open:
            # 포지션이 열려 있음 → 진입 또는 추가 진입
            existing = self._stops.get(stop_key)
            if existing is not None:
                # Direction flip 감지: 기존 stop과 현재 포지션 방향이 다르면 교체
                expected_side = "LONG" if pos.direction == Direction.LONG else "SHORT"
                if existing.position_side != expected_side:
                    await self._cancel_safety_stop(stop_key)
                    await self._place_safety_stop(stop_key, pos)
            else:
                await self._place_safety_stop(stop_key, pos)
        # 포지션 청산됨 → stop 취소
        elif stop_key in self._stops:
            await self._cancel_safety_stop(stop_key)

    async def _on_bar(self, event: AnyEvent) -> None:
        """Bar 이벤트: trailing stop price 업데이트 (throttled)."""
        assert isinstance(event, BarEvent)
        symbol = event.symbol

        # 해당 심볼에 매칭되는 모든 stop key 수집
        stop_keys = self._stop_keys_for_symbol(symbol)
        if not stop_keys:
            return

        for stop_key in stop_keys:
            pos = self._pm.positions.get(stop_key)
            if pos is None or not pos.is_open:
                # 포지션이 사라졌으면 stop 취소
                await self._cancel_safety_stop(stop_key)
                continue

            new_stop_price = self._calculate_stop_price(pos)
            if new_stop_price is None:
                continue

            await self._update_stop_if_needed(stop_key, new_stop_price)

    # =========================================================================
    # Key Helpers
    # =========================================================================

    def _stop_keys_for_symbol(self, symbol: str) -> list[str]:
        """symbol에 매칭되는 모든 stop key 반환."""
        if not self._hedge_mode:
            return [symbol] if symbol in self._stops else []
        return [k for k in self._stops if k == symbol or k.endswith(f"|{symbol}")]

    @staticmethod
    def _real_symbol(stop_key: str) -> str:
        """stop_key에서 실제 거래소 심볼 추출."""
        return stop_key.split("|", 1)[1] if "|" in stop_key else stop_key

    # =========================================================================
    # Exchange API Operations
    # =========================================================================

    async def _place_safety_stop(self, stop_key: str, pos: Position) -> None:
        """포지션에 대한 거래소 안전망 stop 배치."""
        stop_price = self._calculate_stop_price(pos)
        if stop_price is None:
            logger.debug("No safety stop for {} (system_stop_loss=None)", stop_key)
            return

        position_side = "LONG" if pos.direction == Direction.LONG else "SHORT"
        close_side = "sell" if pos.direction == Direction.LONG else "buy"
        safe_id = stop_key.replace("/", "-").replace("|", "_")
        client_order_id = f"{SAFETY_STOP_PREFIX}{safe_id}_{uuid.uuid4().hex[:8]}"

        real_symbol = self._real_symbol(stop_key)
        futures_symbol = self._client.to_futures_symbol(real_symbol)

        # Hedge mode → positionSide 전달
        ps_param = position_side if self._hedge_mode else None

        try:
            result = await self._client.create_stop_market_order(
                symbol=futures_symbol,
                side=close_side,
                stop_price=stop_price,
                client_order_id=client_order_id,
                position_side=ps_param,
            )
            exchange_order_id = result.get("id")
            self._stops[stop_key] = StopOrderState(
                symbol=real_symbol,
                exchange_order_id=exchange_order_id,
                client_order_id=client_order_id,
                stop_price=stop_price,
                position_side=position_side,
                close_side=close_side,
            )
            logger.info(
                "Safety stop placed: {} {} @ {:.2f} (order={})",
                stop_key,
                close_side,
                stop_price,
                exchange_order_id,
            )
        except Exception:
            logger.exception("Failed to place safety stop for {}", stop_key)
            # 기존 state가 있으면 failure 카운트 증가
            if stop_key in self._stops:
                self._stops[stop_key].placement_failures += 1
                await self._check_failure_threshold(stop_key)

    async def _update_stop_if_needed(self, stop_key: str, new_stop_price: float) -> None:
        """0.5%+ 변동 시에만 cancel+create로 업데이트. Ratchet 적용."""
        state = self._stops.get(stop_key)
        if state is None:
            return

        current_price = state.stop_price
        threshold = self._config.exchange_stop_update_threshold

        # 변동률 체크
        if current_price > 0:
            change_pct = abs(new_stop_price - current_price) / current_price
            if change_pct < threshold:
                return

        # Ratchet: LONG은 올림만, SHORT은 내림만
        if state.position_side == "LONG" and new_stop_price < current_price:
            return
        if state.position_side == "SHORT" and new_stop_price > current_price:
            return

        # Cancel + Create
        real_symbol = self._real_symbol(stop_key)
        futures_symbol = self._client.to_futures_symbol(real_symbol)
        ps_param = state.position_side if self._hedge_mode else None
        try:
            # 기존 stop 취소
            if state.exchange_order_id:
                try:
                    await self._client.cancel_order(state.exchange_order_id, futures_symbol)
                except Exception:
                    logger.warning(
                        "Failed to cancel old safety stop for {} (may already be cancelled)",
                        stop_key,
                    )
                # Cancel 후 즉시 ID 초기화 — Create 실패 시에도 stale ID 방지
                state.exchange_order_id = None

            # 새 stop 배치
            safe_id = stop_key.replace("/", "-").replace("|", "_")
            client_order_id = f"{SAFETY_STOP_PREFIX}{safe_id}_{uuid.uuid4().hex[:8]}"
            result = await self._client.create_stop_market_order(
                symbol=futures_symbol,
                side=state.close_side,
                stop_price=new_stop_price,
                client_order_id=client_order_id,
                position_side=ps_param,
            )
            state.exchange_order_id = result.get("id")
            state.client_order_id = client_order_id
            state.stop_price = new_stop_price
            state.last_updated = datetime.now(UTC)
            state.placement_failures = 0

            logger.debug(
                "Safety stop updated: {} @ {:.2f} → {:.2f}",
                stop_key,
                current_price,
                new_stop_price,
            )
        except Exception:
            logger.exception("Failed to update safety stop for {}", stop_key)
            state.placement_failures += 1
            await self._check_failure_threshold(stop_key)

    async def _cancel_safety_stop(self, stop_key: str) -> None:
        """심볼/키의 안전망 stop 취소."""
        state = self._stops.pop(stop_key, None)
        if state is None:
            return

        if state.exchange_order_id:
            real_symbol = self._real_symbol(stop_key)
            futures_symbol = self._client.to_futures_symbol(real_symbol)
            try:
                await self._client.cancel_order(state.exchange_order_id, futures_symbol)
                logger.info(
                    "Safety stop cancelled: {} (order={})", stop_key, state.exchange_order_id
                )
            except Exception:
                logger.warning(
                    "Failed to cancel safety stop for {} (order={}) — may already be filled/cancelled",
                    stop_key,
                    state.exchange_order_id,
                )

    async def cancel_all_stops(self) -> None:
        """모든 안전망 stop 취소."""
        symbols = list(self._stops.keys())
        for symbol in symbols:
            await self._cancel_safety_stop(symbol)
        logger.info("All safety stops cancelled ({} symbols)", len(symbols))

    # =========================================================================
    # Stop Price Calculation
    # =========================================================================

    def _calculate_stop_price(self, pos: Position) -> float | None:
        """포지션에 대한 안전망 stop price 계산.

        SL stop과 TS stop을 각각 계산하고, LONG은 min(넓은 쪽), SHORT은 max(넓은 쪽)를 선택.

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

        # 둘 다 있으면: 더 넓은 쪽 (LONG→min, SHORT→max)
        assert sl_stop is not None and ts_stop is not None
        if pos.direction == Direction.LONG:
            return min(sl_stop, ts_stop)
        return max(sl_stop, ts_stop)

    def _calculate_sl_stop(self, pos: Position) -> float | None:
        """System Stop Loss 기반 안전망 stop price.

        LONG: entry * (1 - sl - margin)
        SHORT: entry * (1 + sl + margin)
        """
        sl = self._config.system_stop_loss
        if sl is None:
            return None

        margin = self._config.exchange_safety_margin
        entry = pos.avg_entry_price
        if entry <= 0:
            return None

        if pos.direction == Direction.LONG:
            return entry * (1.0 - sl - margin)
        if pos.direction == Direction.SHORT:
            return entry * (1.0 + sl + margin)
        return None

    def _calculate_ts_stop(self, pos: Position) -> float | None:
        """Trailing Stop 기반 안전망 stop price.

        ATR 미성숙 (14봉 미달) 시 None 반환.
        LONG: (peak - atr * mult) * (1 - ts_margin)
        SHORT: (trough + atr * mult) * (1 + ts_margin)
        """
        _atr_period = 14
        if not self._config.use_trailing_stop or len(pos.atr_values) < _atr_period:
            return None

        atr = pos.atr_values[-1]
        if atr <= 0:
            return None

        mult = self._config.trailing_stop_atr_multiplier
        ts_margin = self._config.exchange_trailing_safety_margin
        anchor, sign_mult, margin_dir = 0.0, 0.0, 0.0

        if pos.direction == Direction.LONG:
            anchor = pos.peak_price_since_entry
            sign_mult = -1.0  # peak - atr*mult
            margin_dir = -ts_margin  # * (1 - ts_margin)
        elif pos.direction == Direction.SHORT:
            anchor = pos.trough_price_since_entry
            sign_mult = 1.0  # trough + atr*mult
            margin_dir = ts_margin  # * (1 + ts_margin)
        else:
            return None

        if anchor <= 0:
            return None
        return (anchor + sign_mult * atr * mult) * (1.0 + margin_dir)

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
                "position_side": state.position_side,
                "close_side": state.close_side,
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
            self._stops[symbol] = StopOrderState(
                symbol=str(data["symbol"]),
                exchange_order_id=data.get("exchange_order_id"),  # type: ignore[arg-type]
                client_order_id=str(data["client_order_id"]),
                stop_price=float(data["stop_price"]),  # type: ignore[arg-type]
                position_side=str(data["position_side"]),
                close_side=str(data["close_side"]),
                last_updated=datetime.fromisoformat(str(data["last_updated"])),
                placement_failures=int(data.get("placement_failures", 0)),  # type: ignore[arg-type]
            )
        if self._stops:
            logger.info("ExchangeStopManager state restored ({} stops)", len(self._stops))

    @property
    def active_stops(self) -> dict[str, StopOrderState]:
        """현재 활성 stop 주문 (읽기 전용)."""
        return dict(self._stops)

    # =========================================================================
    # Utilities
    # =========================================================================

    async def _check_failure_threshold(self, stop_key: str) -> None:
        """연속 실패 횟수 임계값 체크 + Discord CRITICAL 알림."""
        state = self._stops.get(stop_key)
        if state is not None and state.placement_failures >= _MAX_PLACEMENT_FAILURES:
            logger.critical(
                "SAFETY STOP FAILURE: {} consecutive failures for {} — safety net may be inactive!",
                state.placement_failures,
                stop_key,
            )
            if self._notification_queue is not None:
                from src.notification.formatters import format_safety_stop_failure_embed
                from src.notification.models import ChannelRoute, NotificationItem, Severity

                embed = format_safety_stop_failure_embed(stop_key, state.placement_failures)
                await self._notification_queue.enqueue(
                    NotificationItem(
                        severity=Severity.CRITICAL,
                        channel=ChannelRoute.ALERTS,
                        embed=embed,
                    )
                )

    async def verify_exchange_stops(self) -> None:
        """재시작 후 거래소 실제 주문 존재 여부 검증.

        restore_state() 후 호출하여 각 심볼의 exchange_order_id가
        거래소에 실제 존재하는지 확인합니다.

        - exchange_order_id가 None이면 stale로 제거
        - 거래소에 없으면 stale로 제거 (다음 bar에서 자동 재배치)
        - API 실패 시 보수적으로 상태 유지
        """
        if not self._stops:
            return

        keys_to_remove: list[str] = []
        for stop_key, state in self._stops.items():
            if state.exchange_order_id is None:
                keys_to_remove.append(stop_key)
                logger.warning(
                    "Safety stop for {} has no exchange_order_id — removing stale state", stop_key
                )
                continue

            real_symbol = self._real_symbol(stop_key)
            futures_symbol = self._client.to_futures_symbol(real_symbol)
            try:
                open_orders = await self._client.fetch_open_orders(futures_symbol)
                order_ids = {str(o.get("id", "")) for o in open_orders}
                if state.exchange_order_id not in order_ids:
                    keys_to_remove.append(stop_key)
                    logger.warning(
                        "Safety stop for {} (order={}) not found on exchange — removing stale state",
                        stop_key,
                        state.exchange_order_id,
                    )
                else:
                    logger.info(
                        "Safety stop for {} verified on exchange (order={})",
                        stop_key,
                        state.exchange_order_id,
                    )
            except Exception:
                logger.warning(
                    "Failed to verify safety stop for {} — retaining state conservatively",
                    stop_key,
                )

        for stop_key in keys_to_remove:
            self._stops.pop(stop_key, None)
            # Discord WARNING 알림 (queue가 있을 때만)
            if self._notification_queue is not None:
                from src.notification.formatters import format_safety_stop_stale_embed
                from src.notification.models import ChannelRoute, NotificationItem, Severity

                embed = format_safety_stop_stale_embed(stop_key)
                await self._notification_queue.enqueue(
                    NotificationItem(
                        severity=Severity.WARNING,
                        channel=ChannelRoute.ALERTS,
                        embed=embed,
                    )
                )

        if keys_to_remove:
            logger.info(
                "Safety stop verification: {} stale, {} retained",
                len(keys_to_remove),
                len(self._stops),
            )

    async def place_missing_stops(self) -> int:
        """PM에 열린 포지션이 있지만 stop이 없는 키에 안전망 stop 배치.

        재시작 후 verify_exchange_stops()에서 stale stop이 제거된 경우,
        또는 reconciliation 후 새로운 포지션이 확인된 경우 호출합니다.

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
            logger.info("Placed {} missing safety stops", placed)
        return placed

    def is_safety_stop_order(self, client_order_id: str) -> bool:
        """client_order_id가 안전망 stop 주문인지 확인."""
        return client_order_id.startswith(SAFETY_STOP_PREFIX)
