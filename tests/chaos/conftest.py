"""Chaos test fixtures -- FaultyExecutor, StaleDataFeed.

장애 주입 패턴으로 라이브 edge case를 시뮬레이션합니다.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent, FillEvent, OrderRequestEvent


class FaultyExecutor:
    """Fault injection executor.

    ExecutorPort Protocol 만족. 다양한 장애를 주입합니다.

    Args:
        partial_fill_ratio: 부분 체결 비율 (0.6 = 60%만 체결). None이면 전량 체결.
        timeout_probability: 타임아웃 확률 (0.0~1.0)
        fail_after_n: N번째 주문 이후 에러 발생. None이면 비활성.
        seed: 재현을 위한 random seed
    """

    def __init__(
        self,
        partial_fill_ratio: float | None = None,
        timeout_probability: float = 0.0,
        fail_after_n: int | None = None,
        seed: int = 42,
    ) -> None:
        self._partial_fill_ratio = partial_fill_ratio
        self._timeout_probability = timeout_probability
        self._fail_after_n = fail_after_n
        self._rng = random.Random(seed)
        self._order_count = 0
        self._fills: list[FillEvent] = []
        self._timeouts: list[OrderRequestEvent] = []
        self._errors: list[OrderRequestEvent] = []

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """장애가 주입된 주문 실행.

        Args:
            order: 주문 요청

        Returns:
            FillEvent (정상/부분 체결) 또는 None (타임아웃)

        Raises:
            RuntimeError: fail_after_n 초과 시
        """
        self._order_count += 1

        # fail_after_n: N번째 이후 에러
        if self._fail_after_n is not None and self._order_count > self._fail_after_n:
            self._errors.append(order)
            msg = f"Simulated failure after {self._fail_after_n} orders"
            raise RuntimeError(msg)

        # timeout_probability: 확률적 타임아웃
        if self._rng.random() < self._timeout_probability:
            self._timeouts.append(order)
            return None

        # partial_fill_ratio: 부분 체결
        qty_ratio = self._partial_fill_ratio if self._partial_fill_ratio is not None else 1.0
        fill_price = 50000.0  # 고정 가격 (테스트용)
        notional = order.notional_usd * qty_ratio
        fill_qty = notional / fill_price if fill_price > 0 else 0.0

        fill = FillEvent(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee=notional * 0.0004,
            fill_timestamp=datetime.now(UTC),
            correlation_id=order.correlation_id,
            source="FaultyExecutor",
        )
        self._fills.append(fill)
        return fill

    @property
    def fills(self) -> list[FillEvent]:
        return self._fills

    @property
    def timeouts(self) -> list[OrderRequestEvent]:
        return self._timeouts

    @property
    def errors(self) -> list[OrderRequestEvent]:
        return self._errors

    @property
    def order_count(self) -> int:
        return self._order_count


@pytest.fixture
def bus() -> EventBus:
    return EventBus(queue_size=1000)


@pytest.fixture
def faulty_executor() -> FaultyExecutor:
    return FaultyExecutor()


def make_order_request(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    target_weight: float = 0.5,
    notional_usd: float = 5000.0,
    price: float | None = None,
    client_order_id: str | None = None,
) -> OrderRequestEvent:
    """테스트용 OrderRequestEvent 생성."""
    return OrderRequestEvent(
        client_order_id=client_order_id or f"test-{id(symbol)}-{side}",
        symbol=symbol,
        side=side,
        target_weight=target_weight,
        notional_usd=notional_usd,
        price=price,
        validated=True,
    )


def make_bar_event(
    symbol: str = "BTC/USDT",
    close: float = 50000.0,
    timeframe: str = "1D",
) -> BarEvent:
    """테스트용 BarEvent 생성."""
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=1e6,
        bar_timestamp=datetime.now(UTC),
    )
