"""SmartExecutor — Limit Order 우선 실행기.

Decorator 패턴으로 LiveExecutor를 래핑합니다.
일반 주문은 Limit order를 우선 시도하고, 미체결 시 Market fallback합니다.
긴급 주문(SL/TS/close/direction-flip)은 즉시 Market order로 실행합니다.

ExecutorPort Protocol을 만족합니다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.events import FillEvent, OrderRequestEvent
from src.exchange.binance_futures_client import BinanceFuturesClient

if TYPE_CHECKING:
    from src.eda.executors import LiveExecutor
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.eda.smart_executor_config import SmartExecutorConfig
    from src.monitoring.metrics import SmartExecutorMetrics


class Urgency(StrEnum):
    """주문 긴급도."""

    URGENT = "urgent"
    NORMAL = "normal"


@dataclass
class LimitOrderState:
    """진행 중인 Limit 주문 상태."""

    order: OrderRequestEvent
    exchange_order_id: str
    futures_symbol: str
    limit_price: float
    reference_price: float  # 배치 시점 mid price (deviation 체크용)
    placed_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    filled_qty: float = 0.0
    filled_notional: float = 0.0


class SmartExecutor:
    """Limit Order 우선 실행기.

    Decorator 패턴으로 LiveExecutor(inner)를 래핑합니다.
    ExecutorPort Protocol을 만족합니다.

    Args:
        inner: LiveExecutor 인스턴스 (market fallback용)
        config: SmartExecutor 설정
        futures_client: BinanceFuturesClient (limit order 배치/조회용)
    """

    def __init__(
        self,
        inner: LiveExecutor,
        config: SmartExecutorConfig,
        futures_client: BinanceFuturesClient,
    ) -> None:
        self._inner = inner
        self._config = config
        self._client = futures_client
        self._metrics: SmartExecutorMetrics | None = None
        self._active_limit_count = 0

    @property
    def inner(self) -> LiveExecutor:
        """래핑된 LiveExecutor 접근자."""
        return self._inner

    def set_pm(self, pm: EDAPortfolioManager) -> None:
        """PortfolioManager 참조 설정 (inner에 위임)."""
        self._inner.set_pm(pm)

    def set_metrics(self, metrics: Any) -> None:
        """메트릭 콜백 설정.

        SmartExecutorMetrics가 있으면 자체 보관,
        LiveExecutorMetrics는 inner에 위임합니다.
        """
        from src.monitoring.metrics import SmartExecutorMetrics as SmartMetrics

        if isinstance(metrics, SmartMetrics):
            self._metrics = metrics
        else:
            self._inner.set_metrics(metrics)

    def set_smart_metrics(self, metrics: SmartExecutorMetrics) -> None:
        """SmartExecutor 전용 메트릭 설정."""
        self._metrics = metrics

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 실행 — 긴급/일반 분류 후 적절한 방식으로 실행.

        Args:
            order: 검증된 주문 요청

        Returns:
            체결 결과 (에러 시 None)
        """
        urgency = self._classify_urgency(order)

        if urgency == Urgency.URGENT:
            return await self._inner.execute(order)

        return await self._execute_limit(order)

    def _classify_urgency(self, order: OrderRequestEvent) -> Urgency:
        """주문 긴급도 분류.

        Args:
            order: 주문 요청

        Returns:
            URGENT 또는 NORMAL
        """
        is_urgent = (
            not self._config.enabled
            or order.price is not None  # SL/TS exit
            or order.target_weight == 0  # 포지션 청산
            or not self._client.is_api_healthy
            or self._active_limit_count >= self._config.max_concurrent_limit_orders
            or self._is_direction_flip(order)
        )
        return Urgency.URGENT if is_urgent else Urgency.NORMAL

    def _is_direction_flip(self, order: OrderRequestEvent) -> bool:
        """방향 전환 여부 확인."""
        from src.models.types import Direction

        pm = self._inner._pm  # pyright: ignore[reportPrivateUsage]
        if pm is None:
            return False

        pos = pm.positions.get(order.symbol)
        if pos is None or not pos.is_open:
            return False

        current_dir = pos.direction
        return (
            (order.target_weight > 0 and current_dir == Direction.SHORT)
            or (order.target_weight < 0 and current_dir == Direction.LONG)
        )

    async def _execute_limit(self, order: OrderRequestEvent) -> FillEvent | None:
        """Limit 주문 라이프사이클: 배치 → 모니터 → fallback.

        Args:
            order: 일반 주문 요청

        Returns:
            FillEvent 또는 None
        """
        futures_symbol = BinanceFuturesClient.to_futures_symbol(order.symbol)

        # 1. 현재가 조회
        try:
            ticker = await self._client.fetch_ticker(futures_symbol)
        except Exception:
            logger.warning("SmartExecutor: fetch_ticker failed for {}, falling back to market", order.symbol)
            return await self._inner.execute(order)

        bid = float(ticker.get("bid", 0) or 0)
        ask = float(ticker.get("ask", 0) or 0)

        if bid <= 0 or ask <= 0:
            logger.warning("SmartExecutor: invalid bid/ask for {}, falling back to market", order.symbol)
            return await self._inner.execute(order)

        limit_price = self._compute_limit_price(order.side, bid, ask)
        mid_price = (bid + ask) / 2.0

        # 2. 수량 계산
        amount = order.notional_usd / limit_price
        if amount <= 0:
            return await self._inner.execute(order)

        # 3. Limit 주문 배치
        try:
            result = await self._client.create_order(
                symbol=futures_symbol,
                side=order.side.lower(),
                amount=amount,
                price=limit_price,
                client_order_id=order.client_order_id,
            )
        except Exception:
            logger.exception("SmartExecutor: limit order placement failed for {}", order.symbol)
            return await self._inner.execute(order)

        exchange_order_id = str(result.get("id", ""))
        if not exchange_order_id:
            logger.warning("SmartExecutor: no order ID returned, falling back to market")
            return await self._inner.execute(order)

        state = LimitOrderState(
            order=order,
            exchange_order_id=exchange_order_id,
            futures_symbol=futures_symbol,
            limit_price=limit_price,
            reference_price=mid_price,
        )

        self._active_limit_count += 1
        if self._metrics is not None:
            self._metrics.on_limit_placed(order.symbol)

        logger.info(
            "SmartExecutor: limit {} {} @ {:.2f} (mid={:.2f}) id={}",
            order.side,
            order.symbol,
            limit_price,
            mid_price,
            exchange_order_id,
        )

        try:
            return await self._monitor_limit_order(state)
        finally:
            self._active_limit_count = max(0, self._active_limit_count - 1)

    async def _monitor_limit_order(self, state: LimitOrderState) -> FillEvent | None:
        """Limit 주문 poll 루프.

        Args:
            state: LimitOrderState

        Returns:
            FillEvent 또는 None
        """
        deadline = state.placed_at + self._config.limit_timeout_seconds

        while True:
            await asyncio.sleep(self._config.poll_interval_seconds)

            # 주문 상태 조회
            try:
                order_status = await self._client.fetch_order(
                    state.exchange_order_id, state.futures_symbol
                )
            except Exception:
                logger.warning(
                    "SmartExecutor: fetch_order failed for {}, cancelling",
                    state.order.symbol,
                )
                return await self._cancel_and_handle_remainder(state, reason="fetch_failed")

            status = str(order_status.get("status", ""))
            filled_qty = float(order_status.get("filled", 0) or 0)
            avg_price = float(order_status.get("average", 0) or order_status.get("price", 0) or 0)

            # 완전 체결
            if status == "closed" and filled_qty > 0 and avg_price > 0:
                if self._metrics is not None:
                    self._metrics.on_limit_filled(state.order.symbol)
                logger.info(
                    "SmartExecutor: limit filled {} {} @ {:.2f} qty={:.6f}",
                    state.order.side,
                    state.order.symbol,
                    avg_price,
                    filled_qty,
                )
                fee_info = order_status.get("fee")
                fee = float(fee_info["cost"]) if fee_info and fee_info.get("cost") else 0.0
                return FillEvent(
                    client_order_id=state.order.client_order_id,
                    symbol=state.order.symbol,
                    side=state.order.side,
                    fill_price=avg_price,
                    fill_qty=filled_qty,
                    fee=fee,
                    fill_timestamp=datetime.now(UTC),
                    correlation_id=state.order.correlation_id,
                    source="SmartExecutor:limit",
                    pod_id=state.order.pod_id,
                )

            # 부분 체결 정보 업데이트
            if filled_qty > 0:
                state.filled_qty = filled_qty
                state.filled_notional = filled_qty * avg_price if avg_price > 0 else 0.0

            # 가격 이탈 체크
            if await self._price_deviated(state):
                return await self._cancel_and_handle_remainder(state, reason="price_deviation")

            # 타임아웃 체크
            now = asyncio.get_event_loop().time()
            if now >= deadline:
                if self._metrics is not None:
                    self._metrics.on_limit_timeout(state.order.symbol)
                return await self._cancel_and_handle_remainder(state, reason="timeout")

    async def _price_deviated(self, state: LimitOrderState) -> bool:
        """현재 가격이 배치 시점 대비 크게 이탈했는지 확인.

        Args:
            state: LimitOrderState

        Returns:
            True면 조기 취소 필요
        """
        try:
            ticker = await self._client.fetch_ticker(state.futures_symbol)
        except Exception:
            return True  # 조회 실패 → 안전하게 취소

        bid = float(ticker.get("bid", 0) or 0)
        ask = float(ticker.get("ask", 0) or 0)
        if bid <= 0 or ask <= 0:
            return True

        current_mid = (bid + ask) / 2.0
        deviation_pct = abs(current_mid - state.reference_price) / state.reference_price * 100

        return deviation_pct > self._config.max_price_deviation_pct

    async def _cancel_and_handle_remainder(
        self,
        state: LimitOrderState,
        *,
        reason: str = "timeout",
    ) -> FillEvent | None:
        """Limit 주문 취소 + 잔량 market fallback.

        Args:
            state: LimitOrderState
            reason: 취소 사유

        Returns:
            FillEvent (merge된 결과) 또는 None
        """
        logger.info(
            "SmartExecutor: cancelling limit order {} (reason={})",
            state.exchange_order_id,
            reason,
        )

        # 취소 시도
        try:
            cancel_result = await self._client.cancel_order(
                state.exchange_order_id, state.futures_symbol
            )
        except Exception:
            logger.warning(
                "SmartExecutor: cancel failed for {} — checking final status",
                state.exchange_order_id,
            )
            cancel_result = None

        # 최종 상태 재조회 (cancel race 대응)
        try:
            final_status = await self._client.fetch_order(
                state.exchange_order_id, state.futures_symbol
            )
        except Exception:
            logger.warning("SmartExecutor: final status fetch failed")
            final_status = cancel_result or {}

        final_filled_qty = float(final_status.get("filled", 0) or 0)
        final_avg_price = float(
            final_status.get("average", 0) or final_status.get("price", 0) or 0
        )
        final_fee_info = final_status.get("fee")
        limit_fee = (
            float(final_fee_info["cost"]) if final_fee_info and final_fee_info.get("cost") else 0.0
        )

        # Cancel race: 이미 완전 체결
        if str(final_status.get("status", "")) == "closed" and final_filled_qty > 0:
            if self._metrics is not None:
                self._metrics.on_limit_filled(state.order.symbol)
            return FillEvent(
                client_order_id=state.order.client_order_id,
                symbol=state.order.symbol,
                side=state.order.side,
                fill_price=final_avg_price,
                fill_qty=final_filled_qty,
                fee=limit_fee,
                fill_timestamp=datetime.now(UTC),
                correlation_id=state.order.correlation_id,
                source="SmartExecutor:limit_race",
                pod_id=state.order.pod_id,
            )

        # 부분 체결 정보
        limit_fill: FillEvent | None = None
        if final_filled_qty > 0 and final_avg_price > 0:
            limit_fill = FillEvent(
                client_order_id=state.order.client_order_id,
                symbol=state.order.symbol,
                side=state.order.side,
                fill_price=final_avg_price,
                fill_qty=final_filled_qty,
                fee=limit_fee,
                fill_timestamp=datetime.now(UTC),
                correlation_id=state.order.correlation_id,
                source="SmartExecutor:limit_partial",
                pod_id=state.order.pod_id,
            )

        # Market fallback 비활성화
        if not self._config.fallback_to_market:
            return limit_fill

        if self._metrics is not None:
            self._metrics.on_market_fallback(state.order.symbol)

        # 잔량 계산 → market fallback
        filled_notional = final_filled_qty * final_avg_price if final_filled_qty > 0 and final_avg_price > 0 else 0.0
        remainder_notional = state.order.notional_usd - filled_notional

        if remainder_notional <= 0:
            # 이미 전량 체결됨
            return limit_fill

        # 잔량 market 주문 생성
        remainder_order = OrderRequestEvent(
            client_order_id=f"{state.order.client_order_id}-mkt",
            symbol=state.order.symbol,
            side=state.order.side,
            target_weight=state.order.target_weight,
            notional_usd=remainder_notional,
            price=None,
            validated=True,
            correlation_id=state.order.correlation_id,
            source="SmartExecutor:market_fallback",
            pod_id=state.order.pod_id,
        )

        market_fill = await self._inner.execute(remainder_order)

        if limit_fill is not None and market_fill is not None:
            if self._metrics is not None:
                self._metrics.on_partial_fill_merged(state.order.symbol)
            return self._merge_fills(limit_fill, market_fill, state.order.client_order_id)
        if limit_fill is not None:
            return limit_fill
        return market_fill

    @staticmethod
    def _merge_fills(
        limit_fill: FillEvent,
        market_fill: FillEvent,
        original_client_order_id: str,
    ) -> FillEvent:
        """Limit + Market fill을 VWAP로 합산.

        Args:
            limit_fill: Limit 체결 결과
            market_fill: Market 체결 결과
            original_client_order_id: 원본 주문 ID

        Returns:
            VWAP 합산된 FillEvent
        """
        total_qty = limit_fill.fill_qty + market_fill.fill_qty
        total_notional = (
            limit_fill.fill_qty * limit_fill.fill_price
            + market_fill.fill_qty * market_fill.fill_price
        )
        vwap = total_notional / total_qty if total_qty > 0 else limit_fill.fill_price
        total_fee = limit_fill.fee + market_fill.fee

        return FillEvent(
            client_order_id=original_client_order_id,
            symbol=limit_fill.symbol,
            side=limit_fill.side,
            fill_price=vwap,
            fill_qty=total_qty,
            fee=total_fee,
            fill_timestamp=market_fill.fill_timestamp,
            correlation_id=limit_fill.correlation_id,
            source="SmartExecutor:merged",
            pod_id=limit_fill.pod_id,
        )

    def _compute_limit_price(self, side: str, bid: float, ask: float) -> float:
        """Limit 가격 계산.

        BUY: bid 쪽으로 offset (spread 안쪽)
        SELL: ask 쪽으로 offset (spread 안쪽)

        Args:
            side: "BUY" 또는 "SELL"
            bid: 현재 최우선 매수호가
            ask: 현재 최우선 매도호가

        Returns:
            Limit 가격
        """
        offset_rate = self._config.price_offset_bps / 10000.0

        if side == "BUY":
            # 매수: ask보다 낮은 가격 (spread 안쪽)
            return ask * (1.0 - offset_rate)
        # 매도: bid보다 높은 가격 (spread 안쪽)
        return bid * (1.0 + offset_rate)

    async def cleanup_stale_orders(self, symbols: list[str]) -> None:
        """시작 시 미체결 limit 주문 정리.

        Args:
            symbols: 정리 대상 심볼 리스트
        """
        for symbol in symbols:
            futures_symbol = BinanceFuturesClient.to_futures_symbol(symbol)
            try:
                open_orders = await self._client.fetch_open_orders(futures_symbol)
                for order_info in open_orders:
                    order_type = str(order_info.get("type", "")).lower()
                    if order_type == "limit":
                        order_id = str(order_info.get("id", ""))
                        if order_id:
                            await self._client.cancel_order(order_id, futures_symbol)
                            logger.info(
                                "SmartExecutor: cancelled stale limit order {} for {}",
                                order_id,
                                symbol,
                            )
            except Exception:
                logger.warning("SmartExecutor: stale order cleanup failed for {}", symbol)
