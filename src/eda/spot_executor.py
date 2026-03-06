"""SpotExecutor — Binance Spot 실주문 실행기.

ExecutorPort Protocol 만족. Long-Only Spot 주문 실행.

Rules Applied:
    - Long-Only: target_weight < 0 차단
    - BUY: quoteOrderQty (USDT 금액 기준)
    - SELL: base_amount (보유 수량 기준)
    - Partial fill 감지 + 실제 체결량 반영
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.events import FillEvent, OrderRequestEvent
from src.logging.tracing import component_span_with_context

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.exchange.binance_spot_client import BinanceSpotClient
    from src.monitoring.metrics import LiveExecutorMetrics


class SpotExecutor:
    """Binance Spot 실주문 실행기.

    ExecutorPort Protocol 만족. Market order → 즉시 체결 → FillEvent 반환.
    Long-Only: target_weight < 0 주문은 차단.

    Args:
        spot_client: BinanceSpotClient 인스턴스
    """

    def __init__(self, spot_client: BinanceSpotClient) -> None:
        self._client = spot_client
        self._pm: EDAPortfolioManager | None = None
        self._metrics: LiveExecutorMetrics | None = None

    def set_pm(self, pm: EDAPortfolioManager) -> None:
        """PortfolioManager 참조 설정 (LiveRunner에서 호출)."""
        self._pm = pm

    def set_metrics(self, metrics: LiveExecutorMetrics) -> None:
        """메트릭 콜백 설정 (LiveRunner에서 호출)."""
        self._metrics = metrics

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 실행.

        Args:
            order: 검증된 주문 요청

        Returns:
            체결 결과 (에러 시 None)
        """
        corr_id = str(order.correlation_id) if order.correlation_id else None
        with component_span_with_context("spot.create_order", corr_id, {"symbol": order.symbol}):
            return await self._execute_inner(order)

    async def _execute_inner(self, order: OrderRequestEvent) -> FillEvent | None:
        """execute 본체."""
        if not self._check_guards(order):
            return None

        try:
            if order.side == "BUY":
                return await self._execute_buy(order)
            if order.side == "SELL":
                return await self._execute_sell(order)
        except Exception:
            logger.exception("SpotExecutor: Failed to execute order {}", order.client_order_id)
            return None
        else:
            logger.warning("SpotExecutor: Unknown side '{}' for {}", order.side, order.symbol)
            return None

    def _check_guards(self, order: OrderRequestEvent) -> bool:
        """주문 전 가드 조건 검사. False면 주문 차단."""
        if self._pm is None:
            logger.error("SpotExecutor: PM not set, cannot execute order {}", order.client_order_id)
            return False

        if not self._client.is_api_healthy:
            logger.critical(
                "SpotExecutor: API unhealthy ({} consecutive failures), blocking order {}",
                self._client.consecutive_failures,
                order.client_order_id,
            )
            if self._metrics is not None:
                self._metrics.on_api_blocked(order.symbol)
            return False

        # Long-Only 방어: short 주문 차단
        if order.target_weight < 0:
            logger.warning(
                "SpotExecutor: SHORT order rejected (Spot is Long-Only) — {} target_weight={:.4f}",
                order.symbol,
                order.target_weight,
            )
            return False

        return True

    async def _execute_buy(self, order: OrderRequestEvent) -> FillEvent | None:
        """BUY 실행 — quoteOrderQty (USDT 금액 기준)."""
        notional = order.notional_usd

        # MIN_NOTIONAL 사전 검증
        if not self._client.validate_min_notional(order.symbol, notional):
            min_notional = self._client.get_min_notional(order.symbol)
            logger.warning(
                "SpotExecutor: notional ${:.2f} < MIN_NOTIONAL ${:.2f} for {}, skipping",
                notional,
                min_notional,
                order.symbol,
            )
            if self._metrics is not None:
                self._metrics.on_min_notional_skip(order.symbol)
            return None

        logger.info(
            "SpotExecutor: BUY {} ${:.2f} ({})",
            order.symbol,
            notional,
            order.client_order_id,
        )

        result = await self._client.create_market_buy(
            symbol=order.symbol,
            quote_amount=notional,
            client_order_id=order.client_order_id,
        )

        result = await self._confirm_order(result, order.symbol)
        return self._parse_fill(order, result, metrics=self._metrics)

    async def _execute_sell(self, order: OrderRequestEvent) -> FillEvent | None:
        """SELL 실행 — base_amount (보유 수량 기준)."""
        assert self._pm is not None

        pos = self._pm.positions.get(order.symbol)

        # 매도 수량 결정
        if order.target_weight == 0 and pos is not None and pos.is_open:
            # 전량 청산
            sell_amount = pos.size
        elif pos is not None and pos.is_open:
            # 부분 매도: notional 기반으로 수량 계산
            price_est = pos.last_price if pos.last_price > 0 else 0.0
            if price_est <= 0:
                logger.warning(
                    "SpotExecutor: No price estimate for {}, cannot calculate sell amount",
                    order.symbol,
                )
                return None
            sell_amount = order.notional_usd / price_est
            # 포지션 이상 매도 방지
            sell_amount = min(sell_amount, pos.size)
        else:
            logger.warning("SpotExecutor: No position to sell for {}", order.symbol)
            return None

        if sell_amount <= 0 or not math.isfinite(sell_amount):
            logger.warning(
                "SpotExecutor: Invalid sell amount {:.8f} for {}", sell_amount, order.symbol
            )
            return None

        logger.info(
            "SpotExecutor: SELL {} {:.6f} ({})",
            order.symbol,
            sell_amount,
            order.client_order_id,
        )

        result = await self._client.create_market_sell(
            symbol=order.symbol,
            base_amount=sell_amount,
            client_order_id=order.client_order_id,
        )

        result = await self._confirm_order(result, order.symbol)
        return self._parse_fill(order, result, requested_amount=sell_amount, metrics=self._metrics)

    async def _confirm_order(self, result: dict[str, Any], symbol: str) -> dict[str, Any]:
        """주문 상태 확인 — closed가 아니면 0.5s 후 재확인."""
        _confirm_delay = 0.5
        status = str(result.get("status", ""))
        if status == "closed":
            return result

        order_id = str(result.get("id", ""))
        if not order_id:
            return result

        await asyncio.sleep(_confirm_delay)
        try:
            confirmed = await self._client.fetch_order(order_id, symbol)
        except Exception:
            logger.warning("SpotExecutor: Order {} confirmation failed, using original", order_id)
            return result
        else:
            logger.info(
                "SpotExecutor: Order {} confirmed — status={}",
                order_id,
                confirmed.get("status"),
            )
            return confirmed

    @staticmethod
    def _parse_fill(
        order: OrderRequestEvent,
        result: dict[str, Any],
        requested_amount: float = 0.0,
        metrics: LiveExecutorMetrics | None = None,
    ) -> FillEvent | None:
        """CCXT 응답에서 FillEvent 생성.

        Args:
            order: 원본 주문 요청
            result: CCXT create_order 응답
            requested_amount: 요청한 수량 (partial fill 감지용)
            metrics: LiveExecutor 메트릭 콜백 (선택)

        Returns:
            FillEvent 또는 파싱 실패 시 None
        """
        avg_price = float(result.get("average", 0) or result.get("price", 0) or 0)
        filled_qty = float(result.get("filled", 0) or result.get("amount", 0) or 0)

        if avg_price <= 0 or filled_qty <= 0:
            logger.warning(
                "SpotExecutor: Fill parsing failed — price={}, qty={}",
                avg_price,
                filled_qty,
            )
            if metrics is not None:
                metrics.on_fill_parse_failure(order.symbol)
            return None

        # Partial fill 감지
        _partial_fill_threshold = 0.99
        if requested_amount > 0 and filled_qty < requested_amount * _partial_fill_threshold:
            logger.warning(
                "SpotExecutor: Partial fill — requested={:.6f}, filled={:.6f} ({:.1%})",
                requested_amount,
                filled_qty,
                filled_qty / requested_amount,
            )
            if metrics is not None:
                metrics.on_partial_fill(order.symbol)

        # 수수료 추출
        fee_info = result.get("fee")
        fee = float(fee_info["cost"]) if fee_info and fee_info.get("cost") else 0.0

        return FillEvent(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=avg_price,
            fill_qty=filled_qty,
            fee=fee,
            fill_timestamp=datetime.now(UTC),
            correlation_id=order.correlation_id,
            source="SpotExecutor",
        )
