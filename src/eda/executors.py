"""EDA Executor 구현체.

BacktestExecutor: Deferred execution — 일반 주문은 다음 Bar open 가격으로 체결
ShadowExecutor: 로깅만 (Phase 5 dry-run용)

Rules Applied:
    - Look-Ahead Bias 방지: next-open deferred execution
    - SL/TS 즉시 체결: PM이 설정한 가격으로 즉시 체결
    - CostModel 재사용: 기존 수수료 모델 적용
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    BarEvent,
    FillEvent,
    OrderRequestEvent,
)

if TYPE_CHECKING:
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
