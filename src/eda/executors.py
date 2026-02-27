"""EDA Executor 구현체.

BacktestExecutor: Deferred execution — 일반 주문은 다음 Bar open 가격으로 체결
ShadowExecutor: 로깅만 (Phase 5 dry-run용)
LiveExecutor: Binance Futures 실주문 실행기

Rules Applied:
    - Look-Ahead Bias 방지: next-open deferred execution
    - SL/TS 즉시 체결: PM이 설정한 가격으로 즉시 체결
    - CostModel 재사용: 기존 수수료 모델 적용
    - One-way Mode: reduceOnly 매핑 + direction-flip close-only
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.events import (
    BarEvent,
    FillEvent,
    OrderRequestEvent,
)
from src.exchange.binance_futures_client import BinanceFuturesClient
from src.logging.tracing import component_span_with_context

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.monitoring.metrics import LiveExecutorMetrics
    from src.portfolio.cost_model import CostModel


class BacktestExecutor:
    """백테스트 실행기 (Deferred Execution).

    일반 주문(signal 기반)은 다음 TF Bar의 open 가격으로 체결합니다.
    SL/TS 주문(order.price 설정)은 즉시 체결합니다.

    Flow:
        1. Signal from Bar[N] → PM creates order (price=None) → execute() stores as pending
        2. Bar[N+1] arrives → on_bar() updates prices → fill_pending() fills at open[N+1]
        3. drain_fills() → Runner publishes FillEvents to bus

    Args:
        cost_model: 거래 비용 모델
    """

    def __init__(self, cost_model: CostModel, *, smart_execution: bool = False) -> None:
        self._cost_model = cost_model
        self._smart_execution = smart_execution
        # 심볼별 마지막 bar 가격/타임스탬프
        self._last_open: dict[str, float] = {}
        self._last_close: dict[str, float] = {}
        self._last_bar_timestamp: dict[str, datetime] = {}
        # Deferred execution: 대기 주문 + 체결 결과
        self._pending_orders: list[OrderRequestEvent] = []
        self._deferred_fills: list[FillEvent] = []

    def on_bar(self, bar: BarEvent) -> None:
        """Bar 데이터 업데이트 (가격/타임스탬프)."""
        self._last_open[bar.symbol] = bar.open
        self._last_close[bar.symbol] = bar.close
        self._last_bar_timestamp[bar.symbol] = bar.bar_timestamp

    def fill_pending(self, bar: BarEvent) -> None:
        """해당 심볼의 대기 주문을 새 bar의 open 가격으로 체결.

        Args:
            bar: 새로 도착한 TF bar (open 가격으로 체결)
        """
        remaining: list[OrderRequestEvent] = []
        for order in self._pending_orders:
            if order.symbol == bar.symbol:
                fill = self._create_fill(order, bar.open, bar.bar_timestamp)
                if fill is not None:
                    self._deferred_fills.append(fill)
                else:
                    logger.warning(
                        "Deferred fill failed for {}: invalid price {}",
                        order.client_order_id,
                        bar.open,
                    )
            else:
                remaining.append(order)
        self._pending_orders = remaining

    def drain_fills(self) -> list[FillEvent]:
        """대기 체결 결과를 반환하고 비웁니다."""
        fills = self._deferred_fills
        self._deferred_fills = []
        return fills

    @property
    def pending_count(self) -> int:
        """대기 주문 수."""
        return len(self._pending_orders)

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 실행.

        SL/TS 주문 (order.price 설정): 즉시 체결 (PM이 명시한 가격)
        일반 주문 (order.price 없음): 다음 bar의 open에 체결하도록 대기
        """
        # SL/TS exit: 즉시 체결 (PM이 설정한 stop 가격)
        if order.price is not None:
            fill_ts = self._last_bar_timestamp.get(order.symbol, datetime.now(UTC))
            return self._create_fill(order, order.price, fill_ts)

        # 일반 주문: 다음 bar의 open에 체결하도록 대기
        self._pending_orders.append(order)
        return None

    def _create_fill(
        self,
        order: OrderRequestEvent,
        fill_price: float,
        fill_ts: datetime,
    ) -> FillEvent | None:
        """FillEvent 생성 (공통 로직).

        Args:
            order: 주문 요청
            fill_price: 체결 가격
            fill_ts: 체결 시각

        Returns:
            FillEvent 또는 유효하지 않은 경우 None
        """
        if fill_price <= 0 or not math.isfinite(fill_price):
            return None

        notional = order.notional_usd
        if not math.isfinite(notional) or notional <= 0:
            return None

        # smart_execution 모드: 일반 주문(price=None)은 maker fee + 슬리피지 0
        # SL/TS(price!=None)는 긴급 = taker fee + 슬리피지 그대로
        is_smart_normal = self._smart_execution and order.price is None

        # 슬리피지를 가격에 적용 (VBT parity: price deterioration)
        slip = 0.0 if is_smart_normal else self._cost_model.slip_rate
        if order.side == "BUY":
            adjusted_price = fill_price * (1.0 + slip)
        else:
            adjusted_price = fill_price * (1.0 - slip)

        # SL/TS exit (order.price 설정): 수량은 원래 가격 기준 (전량 청산 보장)
        # 일반 entry/rebalance: 수량은 슬리피지 반영 가격 기준
        fill_qty = notional / fill_price if order.price is not None else notional / adjusted_price
        fee_rate = (
            self._cost_model.effective_fee_for_order(is_limit=True)
            if is_smart_normal
            else self._cost_model.effective_fee
        )
        fee = notional * fee_rate

        return FillEvent(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=adjusted_price,
            fill_qty=fill_qty,
            fee=fee,
            fill_timestamp=fill_ts,
            correlation_id=order.correlation_id,
            source="BacktestExecutor",
            pod_id=order.pod_id,
        )


class ShadowExecutor:
    """Shadow 실행기 (Phase 5 dry-run용).

    실제 체결 없이 주문을 로깅합니다.
    """

    def __init__(self) -> None:
        self._order_log: list[OrderRequestEvent] = []

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 로깅만 수행, 체결 없음."""
        self._order_log.append(order)
        logger.info(
            "Shadow order: {} {} {} notional=${:.2f}",
            order.symbol,
            order.side,
            order.client_order_id,
            order.notional_usd,
        )
        return None

    @property
    def order_log(self) -> list[OrderRequestEvent]:
        """기록된 주문 목록."""
        return self._order_log


class LiveExecutor:
    """Binance Futures 실주문 실행기.

    ExecutorPort Protocol 만족. Market order → 즉시 체결 → FillEvent 반환.
    One-way Mode reduceOnly 매핑 + direction-flip 분할 처리.
    Hedge Mode: positionSide(LONG/SHORT) 매핑, reduceOnly 미사용.

    reduceOnly 결정 로직 (One-way):
        1. order.price is not None (SL/TS exit) → reduceOnly=True
        2. order.target_weight == 0 (flat close) → reduceOnly=True
        3. Direction-flip (LONG→SHORT or SHORT→LONG) → reduceOnly close만 실행
        4. 같은 방향 entry/increase → reduceOnly=False

    Hedge mode position_side 결정 로직:
        1. SL/TS exit (order.price) → 현재 방향의 positionSide
        2. Flat close (target_weight == 0) → 현재 방향의 positionSide
        3. Entry/Increase: target_weight > 0 → LONG, < 0 → SHORT

    Args:
        futures_client: BinanceFuturesClient 인스턴스
        hedge_mode: True → Hedge Mode (positionSide 사용)
    """

    def __init__(
        self,
        futures_client: BinanceFuturesClient,
        metrics: LiveExecutorMetrics | None = None,
        *,
        hedge_mode: bool = False,
    ) -> None:
        self._client = futures_client
        self._pm: EDAPortfolioManager | None = None
        self._metrics = metrics
        self._hedge_mode = hedge_mode

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
        with component_span_with_context(
            "exchange.create_order", corr_id, {"symbol": order.symbol}
        ):
            return await self._execute_inner(order)

    async def _execute_inner(self, order: OrderRequestEvent) -> FillEvent | None:
        """execute 본체 (tracing span 내부)."""
        if self._pm is None:
            logger.error("LiveExecutor: PM not set, cannot execute order {}", order.client_order_id)
            return None

        # API 건전성 체크
        if not self._client.is_api_healthy:
            logger.critical(
                "LiveExecutor: API unhealthy ({} consecutive failures), blocking order {}",
                self._client.consecutive_failures,
                order.client_order_id,
            )
            if self._metrics is not None:
                self._metrics.on_api_blocked(order.symbol)
            return None

        futures_symbol = BinanceFuturesClient.to_futures_symbol(order.symbol)

        try:
            if self._hedge_mode:
                position_side = self._resolve_hedge_position_side(order)
                logger.info(
                    "LiveExecutor[hedge]: {} {} {} notional=${:.2f} positionSide={}",
                    order.symbol,
                    order.side,
                    order.client_order_id,
                    order.notional_usd,
                    position_side,
                )
                return await self._execute_single(
                    order=order,
                    futures_symbol=futures_symbol,
                    reduce_only=False,
                    position_side=position_side,
                )
            else:
                reduce_only, is_flip_close = self._resolve_reduce_only(order)
                logger.info(
                    "LiveExecutor: {} {} {} notional=${:.2f} reduceOnly={} flip_close={}",
                    order.symbol,
                    order.side,
                    order.client_order_id,
                    order.notional_usd,
                    reduce_only,
                    is_flip_close,
                )
                return await self._execute_single(
                    order=order,
                    futures_symbol=futures_symbol,
                    reduce_only=reduce_only,
                )
        except Exception:
            logger.exception(
                "LiveExecutor: Failed to execute order {}",
                order.client_order_id,
            )
            return None

    def _resolve_reduce_only(
        self,
        order: OrderRequestEvent,
    ) -> tuple[bool, bool]:
        """reduceOnly / is_flip_close 결정 (One-way mode).

        Returns:
            (reduce_only, is_flip_close) 튜플
        """
        from src.models.types import Direction

        if self._pm is None:
            msg = "LiveExecutor._resolve_reduce_only called without PM set"
            raise RuntimeError(msg)
        pos = self._pm.positions.get(order.symbol)
        current_dir = pos.direction if pos and pos.is_open else Direction.NEUTRAL

        # 1. SL/TS exit (price 설정)
        if order.price is not None:
            return True, False

        # 2. Flat close (target_weight == 0)
        if order.target_weight == 0:
            return True, False

        # 3. Direction-flip: close만 실행
        if order.target_weight > 0 and current_dir == Direction.SHORT:
            return True, True
        if order.target_weight < 0 and current_dir == Direction.LONG:
            return True, True

        # 4. 같은 방향 entry/increase
        return False, False

    def _resolve_hedge_position_side(self, order: OrderRequestEvent) -> str:
        """Hedge mode: positionSide 결정.

        Returns:
            "LONG" or "SHORT"
        """
        from src.models.types import Direction

        if self._pm is None:
            msg = "LiveExecutor._resolve_hedge_position_side called without PM set"
            raise RuntimeError(msg)

        # Hedge mode에서 PM positions는 composite key 사용
        pos_key = self._pm._pos_key(order.symbol, order.pod_id)  # pyright: ignore[reportPrivateUsage]
        pos = self._pm.positions.get(pos_key)
        current_dir = pos.direction if pos and pos.is_open else Direction.NEUTRAL

        # 1. SL/TS exit (price 설정) → 현재 방향의 positionSide
        if order.price is not None:
            if current_dir == Direction.SHORT:
                return "SHORT"
            return "LONG"

        # 2. Flat close (target_weight == 0) → 현재 방향의 positionSide
        if order.target_weight == 0:
            if current_dir == Direction.SHORT:
                return "SHORT"
            return "LONG"

        # 3. Entry/Increase
        if order.target_weight > 0:
            return "LONG"
        return "SHORT"

    async def _fetch_fresh_price(self, futures_symbol: str) -> float | None:
        """거래소에서 실시간 가격 조회.

        실패 시 PM last_price로 fallback.
        """
        try:
            ticker = await self._client.fetch_ticker(futures_symbol)
            last = float(ticker.get("last", 0) or 0)
            if last > 0:
                return last
        except Exception:
            logger.warning(
                "LiveExecutor: fetch_ticker failed for {}, using PM price", futures_symbol
            )
        return None

    async def _execute_single(
        self,
        order: OrderRequestEvent,
        futures_symbol: str,
        reduce_only: bool,
        position_side: str | None = None,
    ) -> FillEvent | None:
        """단일 주문 실행 + FillEvent 생성.

        Args:
            order: 주문 요청
            futures_symbol: "BTC/USDT:USDT" 형태
            reduce_only: 청산 전용 여부
            position_side: Hedge mode "LONG"/"SHORT" (None → One-way)

        Returns:
            FillEvent 또는 실패 시 None
        """
        if self._pm is None:
            msg = "LiveExecutor._execute_single called without PM set"
            raise RuntimeError(msg)

        # 수량 계산: hedge mode에서는 composite key로 포지션 조회
        is_close = reduce_only or (
            position_side is not None and (order.price is not None or order.target_weight == 0)
        )
        if is_close:
            if self._hedge_mode:
                pos_key = self._pm._pos_key(order.symbol, order.pod_id)  # pyright: ignore[reportPrivateUsage]
                pos = self._pm.positions.get(pos_key)
            else:
                pos = self._pm.positions.get(order.symbol)
            amount = pos.size if pos and pos.is_open else order.notional_usd / 50000.0
        else:
            # 실시간 가격 조회 → fallback: PM last_price
            fresh_price = await self._fetch_fresh_price(futures_symbol)
            pos = self._pm.positions.get(order.symbol)
            pm_price = pos.last_price if pos and pos.last_price > 0 else 0.0
            price_est = fresh_price if fresh_price is not None else pm_price
            if price_est <= 0:
                logger.warning(
                    "LiveExecutor: No price estimate for {}, cannot calculate amount", order.symbol
                )
                return None
            amount = order.notional_usd / price_est

            # MIN_NOTIONAL 사전 검증
            if not self._client.validate_min_notional(futures_symbol, order.notional_usd):
                min_notional = self._client.get_min_notional(futures_symbol)
                logger.warning(
                    "LiveExecutor: notional ${:.2f} < MIN_NOTIONAL ${:.2f} for {}, skipping",
                    order.notional_usd,
                    min_notional,
                    order.symbol,
                )
                if self._metrics is not None:
                    self._metrics.on_min_notional_skip(order.symbol)
                return None

        if amount <= 0 or not math.isfinite(amount):
            logger.warning("LiveExecutor: Invalid amount {:.8f} for {}", amount, order.symbol)
            return None

        result = await self._client.create_order(
            symbol=futures_symbol,
            side=order.side.lower(),
            amount=amount,
            reduce_only=reduce_only,
            client_order_id=order.client_order_id,
            position_side=position_side,
        )

        # 주문 확인: status != "closed"면 fetch_order로 재확인
        result = await self._confirm_order(result, futures_symbol)

        return self._parse_fill(order, result, amount, metrics=self._metrics)

    async def _confirm_order(self, result: dict[str, Any], futures_symbol: str) -> dict[str, Any]:
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
            confirmed = await self._client.fetch_order(order_id, futures_symbol)
        except Exception:
            logger.warning("LiveExecutor: Order {} confirmation failed, using original", order_id)
            return result
        else:
            logger.info(
                "LiveExecutor: Order {} confirmed — status={}",
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
                "LiveExecutor: Fill parsing failed — price={}, qty={}",
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
                "LiveExecutor: Partial fill — requested={:.6f}, filled={:.6f} ({:.1%})",
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
            source="LiveExecutor",
            pod_id=order.pod_id,
        )
