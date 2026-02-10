"""EDA Executor 구현체.

BacktestExecutor: Deferred execution — 일반 주문은 다음 Bar open 가격으로 체결
ShadowExecutor: 로깅만 (Phase 5 dry-run용)
LiveExecutor: Binance Futures 실주문 실행기

Rules Applied:
    - Look-Ahead Bias 방지: next-open deferred execution
    - SL/TS 즉시 체결: PM이 설정한 가격으로 즉시 체결
    - CostModel 재사용: 기존 수수료 모델 적용
    - Hedge Mode: positionSide 매핑 + direction-flip close-only
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager
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

    def __init__(self, cost_model: CostModel) -> None:
        self._cost_model = cost_model
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

        fill_qty = notional / fill_price
        fee = notional * self._cost_model.total_fee_rate

        return FillEvent(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee=fee,
            fill_timestamp=fill_ts,
            correlation_id=order.correlation_id,
            source="BacktestExecutor",
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
    Hedge Mode positionSide 매핑 + direction-flip 분할 처리.

    positionSide 결정 로직:
        1. order.price is not None (SL/TS exit) → 현재 포지션 방향으로 reduceOnly close
        2. order.target_weight == 0 (flat close) → 현재 포지션 방향으로 reduceOnly close
        3. Direction-flip (LONG→SHORT or SHORT→LONG) → close만 실행, 다음 bar에서 PM이 open 생성
        4. 같은 방향 entry/increase → 해당 positionSide로 open

    Args:
        futures_client: BinanceFuturesClient 인스턴스
    """

    def __init__(self, futures_client: BinanceFuturesClient) -> None:
        self._client = futures_client
        self._pm: EDAPortfolioManager | None = None

    def set_pm(self, pm: EDAPortfolioManager) -> None:
        """PortfolioManager 참조 설정 (LiveRunner에서 호출)."""
        self._pm = pm

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 실행.

        Args:
            order: 검증된 주문 요청

        Returns:
            체결 결과 (에러 시 None)
        """
        if self._pm is None:
            logger.error("LiveExecutor: PM not set, cannot execute order {}", order.client_order_id)
            return None

        futures_symbol = BinanceFuturesClient.to_futures_symbol(order.symbol)

        try:
            position_side, reduce_only, is_flip_close = self._resolve_position_side(order)

            logger.info(
                "LiveExecutor: {} {} {} notional=${:.2f} positionSide={} reduceOnly={} flip_close={}",
                order.symbol,
                order.side,
                order.client_order_id,
                order.notional_usd,
                position_side,
                reduce_only,
                is_flip_close,
            )

            return await self._execute_single(
                order=order,
                futures_symbol=futures_symbol,
                position_side=position_side,
                reduce_only=reduce_only,
            )
        except Exception:
            logger.exception(
                "LiveExecutor: Failed to execute order {}",
                order.client_order_id,
            )
            return None

    def _resolve_position_side(
        self,
        order: OrderRequestEvent,
    ) -> tuple[str, bool, bool]:
        """positionSide / reduceOnly / is_flip_close 결정.

        Returns:
            (position_side, reduce_only, is_flip_close) 튜플
        """
        from src.models.types import Direction

        assert self._pm is not None
        pos = self._pm.positions.get(order.symbol)
        current_dir = pos.direction if pos and pos.is_open else Direction.NEUTRAL

        # 1. SL/TS exit (price 설정)
        if order.price is not None:
            ps = "LONG" if current_dir == Direction.LONG else "SHORT"
            return ps, True, False

        # 2. Flat close (target_weight == 0)
        if order.target_weight == 0:
            ps = "LONG" if current_dir == Direction.LONG else "SHORT"
            return ps, True, False

        # 3. LONG 방향 주문
        if order.target_weight > 0:
            if current_dir == Direction.SHORT:
                # Direction-flip: close SHORT만 실행
                return "SHORT", True, True
            return "LONG", False, False

        # 4. SHORT 방향 주문
        if current_dir == Direction.LONG:
            # Direction-flip: close LONG만 실행
            return "LONG", True, True
        return "SHORT", False, False

    async def _execute_single(
        self,
        order: OrderRequestEvent,
        futures_symbol: str,
        position_side: str,
        reduce_only: bool,
    ) -> FillEvent | None:
        """단일 주문 실행 + FillEvent 생성.

        Args:
            order: 주문 요청
            futures_symbol: "BTC/USDT:USDT" 형태
            position_side: "LONG" 또는 "SHORT"
            reduce_only: 청산 전용 여부

        Returns:
            FillEvent 또는 실패 시 None
        """
        assert self._pm is not None

        # 수량 계산
        if reduce_only:
            pos = self._pm.positions.get(order.symbol)
            amount = pos.size if pos and pos.is_open else order.notional_usd / 50000.0
        else:
            # 거래소에서 현재가로 수량 계산 (last_price 사용)
            pos = self._pm.positions.get(order.symbol)
            price_est = pos.last_price if pos and pos.last_price > 0 else 0.0
            if price_est <= 0:
                logger.warning(
                    "LiveExecutor: No price estimate for {}, cannot calculate amount", order.symbol
                )
                return None
            amount = order.notional_usd / price_est

        if amount <= 0 or not math.isfinite(amount):
            logger.warning("LiveExecutor: Invalid amount {:.8f} for {}", amount, order.symbol)
            return None

        result = await self._client.create_order(
            symbol=futures_symbol,
            side=order.side.lower(),
            amount=amount,
            position_side=position_side,
            reduce_only=reduce_only,
            client_order_id=order.client_order_id,
        )

        return self._parse_fill(order, result)

    @staticmethod
    def _parse_fill(order: OrderRequestEvent, result: dict[str, Any]) -> FillEvent | None:
        """CCXT 응답에서 FillEvent 생성.

        Args:
            order: 원본 주문 요청
            result: CCXT create_order 응답

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
            return None

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
        )
