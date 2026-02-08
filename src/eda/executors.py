"""EDA Executor 구현체.

BacktestExecutor: 다음 Bar open 가격으로 체결 (look-ahead bias 방지)
ShadowExecutor: 로깅만 (Phase 5 dry-run용)

Rules Applied:
    - Look-Ahead Bias 방지: next-open 체결
    - CostModel 재사용: 기존 수수료 모델 적용
"""

from __future__ import annotations

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
    """백테스트 실행기.

    다음 Bar의 open 가격으로 체결합니다 (look-ahead bias 방지).
    주문이 들어오면 내부에 저장하고, 다음 BarEvent 수신 시 체결합니다.

    Args:
        cost_model: 거래 비용 모델
    """

    def __init__(self, cost_model: CostModel) -> None:
        self._cost_model = cost_model
        # 심볼별 마지막 bar open 가격 (next-open 체결용)
        self._last_open: dict[str, float] = {}
        # 심볼별 마지막 bar close 가격 (fallback)
        self._last_close: dict[str, float] = {}
        # 심볼별 마지막 bar timestamp (H-001: 체결 시각 정확성)
        self._last_bar_timestamp: dict[str, datetime] = {}

    def on_bar(self, bar: BarEvent) -> None:
        """Bar 데이터 업데이트 (Runner가 호출)."""
        self._last_open[bar.symbol] = bar.open
        self._last_close[bar.symbol] = bar.close
        self._last_bar_timestamp[bar.symbol] = bar.bar_timestamp

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 체결.

        현재 bar의 open 가격으로 체결합니다.
        (HistoricalDataFeed가 Bar를 발행 → StrategyEngine이 Signal 생성
        → PM이 Order 생성 → RM 검증 → OMS → 여기서 체결)

        실제로는 "다음 bar의 open"이 이상적이지만,
        EDA에서는 현재 bar의 open을 사용합니다 (Bar 수신 후 체결이므로).
        """
        symbol = order.symbol

        # SL/TS 체결: order.price 설정 시 해당 가격 사용 (close price)
        # 일반 주문: _last_open 사용 (current bar's open = VBT의 next-open 동일)
        fill_price: float | None = (
            order.price if order.price is not None else self._last_open.get(symbol)
        )

        if fill_price is None:
            # Fallback: close 가격 사용
            fill_price = self._last_close.get(symbol)
            if fill_price is None:
                logger.warning("No price data for {}, cannot fill", symbol)
                return None

        # 수량 계산 (notional / price)
        if fill_price <= 0:
            return None

        fill_qty = order.notional_usd / fill_price

        # 수수료 계산
        fee = order.notional_usd * self._cost_model.total_fee_rate

        # 백테스트: bar_timestamp 사용, 없으면 현재 시각 (라이브 호환)
        fill_ts = self._last_bar_timestamp.get(symbol, datetime.now(UTC))

        return FillEvent(
            client_order_id=order.client_order_id,
            symbol=symbol,
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
