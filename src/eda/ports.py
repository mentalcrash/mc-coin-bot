"""EDA Port Protocol 정의.

DataFeed와 Executor의 명시적 인터페이스를 정의합니다.
structural subtyping으로 기존 구현체가 자동으로 만족합니다.

Ports:
    - DataFeedPort: 데이터 피드 인터페이스 (HistoricalDataFeed, AggregatingDataFeed)
    - ExecutorPort: 주문 실행기 인터페이스 (BacktestExecutor, ShadowExecutor, LiveExecutor)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.core.events import FillEvent, OrderRequestEvent


@runtime_checkable
class DataFeedPort(Protocol):
    """데이터 피드 인터페이스.

    HistoricalDataFeed, AggregatingDataFeed, LiveDataFeed 등이 구현합니다.
    """

    async def start(self, bus: EventBus) -> None:
        """데이터 리플레이/스트리밍을 시작합니다."""
        ...

    async def stop(self) -> None:
        """데이터 피드를 중지합니다."""
        ...

    @property
    def bars_emitted(self) -> int:
        """발행된 총 BarEvent 수."""
        ...


@runtime_checkable
class ExecutorPort(Protocol):
    """주문 실행기 인터페이스.

    BacktestExecutor, ShadowExecutor, LiveExecutor 등이 구현합니다.
    """

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문 실행.

        Args:
            order: 검증된 주문 요청

        Returns:
            체결 결과 (None이면 체결 실패)
        """
        ...
