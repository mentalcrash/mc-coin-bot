---
name: eda-component
description: >
  EDA(Event-Driven Architecture) 컴포넌트를 생성하는 가이드.
  새 이벤트 타입, EventBus 핸들러, Executor 어댑터, 엔진 플러그인 추가 시 사용.
  사용 시점: LiveExecutor 구현, 모니터링 엔진 추가, 새 이벤트 타입 정의,
  EventBus 연동 코드 작성, EDA 시스템 확장 시.
argument-hint: <component-type>
---

# EDA Component — 이벤트 기반 컴포넌트 개발 가이드

## 이벤트 체인 개요

```
DataFeed ──BAR──> StrategyEngine ──SIGNAL──> PM ──ORDER_REQUEST──> RM ──ORDER_VALIDATED──> OMS ──FILL──> PM
    │                                        │                                                       │
    │                                   BarEvent                                              AnalyticsEngine
    │                                  (SL/TS 체크)                                          (equity curve)
    │
    └── CandleAggregator (1m → target TF)
```

상세 다이어그램: [references/event-chain.md](references/event-chain.md)

---

## 컴포넌트 유형별 가이드

### 유형 A: 새 이벤트 타입 추가

**파일 수정 순서:**

#### 1. `src/core/events.py` — 이벤트 모델 정의

```python
# EventType enum에 추가
class EventType(str, Enum):
    # ... 기존 타입들
    NEW_TYPE = "NEW_TYPE"

# Flat frozen Pydantic model (상속 금지)
class NewTypeEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: Literal[EventType.NEW_TYPE] = EventType.NEW_TYPE
    timestamp: datetime
    correlation_id: str | None = None
    # ... 이벤트 특화 필드
```

#### 2. `AnyEvent` union 타입에 추가

```python
type AnyEvent = (
    BarEvent
    | SignalEvent
    | OrderRequestEvent
    # ...
    | NewTypeEvent  # 추가
)
```

#### 3. `EVENT_TYPE_MAP` 등록

```python
EVENT_TYPE_MAP: dict[EventType, type[BaseModel]] = {
    # ...
    EventType.NEW_TYPE: NewTypeEvent,
}
```

#### 4. DROPPABLE 분류 결정

```python
# 절대 드롭 불가 (주문, 체결, 밸런스 등 재무 이벤트)
NEVER_DROP_EVENTS: frozenset[EventType] = frozenset({
    EventType.ORDER_REQUEST,
    EventType.FILL,
    EventType.BALANCE_UPDATE,
    # EventType.NEW_TYPE,  # 재무 이벤트면 여기
})

# 큐 포화 시 드롭 가능 (시장 데이터 등)
DROPPABLE_EVENTS: frozenset[EventType] = frozenset({
    EventType.BAR,
    # EventType.NEW_TYPE,  # 시장 데이터면 여기
})
```

**판단 기준:**
- 누락 시 **자금 손실** 가능 → `NEVER_DROP_EVENTS`
- 누락 시 **성능 저하**만 → `DROPPABLE_EVENTS`
- 어느 쪽에도 속하지 않으면 → 기본 동작 (타임아웃 후 드롭)

### 유형 B: EventBus 핸들러 (엔진/매니저)

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.core.events import AnyEvent


class NewEngine:
    """새 EDA 엔진."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def register(self) -> None:
        """EventBus에 핸들러 등록."""
        self._bus.subscribe(EventType.SIGNAL, self._on_signal)
        self._bus.subscribe(EventType.BAR, self._on_bar)

    async def _on_signal(self, event: AnyEvent) -> None:
        """SignalEvent 핸들러."""
        assert event.event_type == EventType.SIGNAL
        # 비즈니스 로직
        # ...

        # 파생 이벤트 발행 (correlation_id 전파)
        await self._bus.publish(NewTypeEvent(
            timestamp=event.timestamp,
            correlation_id=event.correlation_id,
            # ...
        ))

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent 핸들러 — 에러 격리."""
        try:
            # 로직
            pass
        except Exception:
            # 핸들러 에러가 다른 핸들러에 영향 주지 않도록
            # EventBus가 자동으로 격리하지만, 내부 상태 보호 필요
            import loguru
            loguru.logger.exception("Handler error in NewEngine._on_bar")
```

**핸들러 필수 패턴:**
- `register()` 메서드로 구독 등록
- `event.event_type` 검증 (방어적 assert)
- `correlation_id` 전파 (이벤트 추적 체인)
- 내부 상태 보호 (try-except 또는 EventBus 격리에 의존)

### 유형 C: Executor 어댑터

```python
from src.eda.executors import ExecutorPort


class LiveExecutor(ExecutorPort):
    """라이브 거래소 주문 실행기."""

    async def execute(self, order: OrderRequestEvent) -> FillEvent:
        """주문 실행 → 체결 이벤트 반환."""
        # 1. Decimal precision
        # 2. amount_to_precision
        # 3. client_order_id 설정
        # 4. 거래소 API 호출
        # 5. FillEvent 생성
        ...
```

**Executor 필수 패턴:**
- `ExecutorPort` 프로토콜 구현
- `client_order_id` 멱등성 보장
- Decimal precision (float 금지)
- 에러 시 FillEvent에 실패 상태 반영

### 유형 D: Runner 통합

```python
# src/eda/runner.py 의 run() 메서드에 등록
async def run(self) -> PerformanceMetrics:
    bus = EventBus(queue_size=self._queue_size)

    # 1. 엔진 생성 (순서 중요!)
    analytics = AnalyticsEngine(bus, ...)
    pm = EDAPortfolioManager(bus, ...)
    rm = RiskManager(bus, ...)
    oms = OrderManagementSystem(bus, ...)
    strategy_engine = StrategyEngine(bus, ...)
    new_engine = NewEngine(bus)  # 추가

    # 2. 등록 (순서 = 이벤트 처리 순서)
    await analytics.register()
    await pm.register()
    await rm.register()
    await oms.register()
    await strategy_engine.register()
    await new_engine.register()  # 추가

    # 3. 데이터 피드 실행
    await self._feed.replay(bus)

    # 4. 결과 수집
    return analytics.get_metrics()
```

**등록 순서 원칙:**
- Analytics가 가장 먼저 (모든 이벤트 기록)
- PM → RM → OMS 순서 (방어 계층)
- StrategyEngine은 마지막 (BAR 이벤트 소비)

---

## 핵심 Gotchas

### 1. `await bus.flush()` 필수

```python
# ❌ flush 없이 bar 연속 발행
for bar in bars:
    await bus.publish(bar)
# → 모든 파생 이벤트가 마지막 bar 가격으로 체결

# ✅ bar별 flush
for bar in bars:
    await bus.publish(bar)
    await bus.flush()  # 이벤트 체인 완료 대기
```

### 2. DROPPABLE bar 처리

BAR 이벤트는 `put_nowait`으로 발행 → 큐 포화 시 드롭됨.
DataFeed에서 `await bus.flush()` 호출이 이를 방지.

### 3. Batch Mode (_batch_mode)

멀티에셋(asset_weights > 1)에서 PM은 batch mode로 전환:
- Signal 수집 → 동일 equity snapshot으로 일괄 주문
- `flush_pending_signals()` 호출 필수 (Runner에서)
- Effective threshold = `rebalance_threshold * asset_weight`

### 4. `_stopped_this_bar` Guard

PM에서 stop-loss/trailing-stop 발동 후 같은 bar에서 재진입 방지:
```python
if symbol in self._stopped_this_bar:
    return  # 이번 bar에서 이미 손절됨
```

### 5. Equity 계산

```python
# ❌ 이중 계산
total_equity = cash + notional + unrealized  # notional에 unrealized 포함!

# ✅ 정답
total_equity = cash + long_notional - short_notional
```

### 6. ATR Incremental 계산

```python
# ❌ last_price 변경 후 ATR 계산
pos.last_price = current_price
pos.update_atr(current_price)  # prev_close가 이미 변경됨!

# ✅ ATR 먼저 계산
pos.update_atr(current_price)  # prev_close = pos.last_price 사용
pos.last_price = current_price
```

### 7. correlation_id 전파

```python
# ❌ 새 이벤트에 correlation_id 누락
await bus.publish(OrderRequestEvent(
    timestamp=signal.timestamp,
    # correlation_id 빠짐!
))

# ✅ 전파
await bus.publish(OrderRequestEvent(
    timestamp=signal.timestamp,
    correlation_id=signal.correlation_id,  # 체인 유지
))
```

---

## 테스트 패턴

### EventBus 통합 테스트

```python
import asyncio
import pytest
from src.core.event_bus import EventBus
from src.core.events import EventType, BarEvent


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus(queue_size=100)


@pytest.mark.asyncio
async def test_event_chain(event_bus: EventBus) -> None:
    received: list[AnyEvent] = []

    async def handler(event: AnyEvent) -> None:
        received.append(event)

    event_bus.subscribe(EventType.BAR, handler)
    await event_bus.start()

    bar = BarEvent(...)
    await event_bus.publish(bar)
    await event_bus.flush()

    assert len(received) == 1
    assert received[0].event_type == EventType.BAR

    await event_bus.stop()
```

### 핸들러 에러 격리 테스트

```python
@pytest.mark.asyncio
async def test_handler_error_isolation(event_bus: EventBus) -> None:
    """하나의 핸들러 에러가 다른 핸들러에 영향 주지 않음."""
    results: list[str] = []

    async def failing_handler(event: AnyEvent) -> None:
        raise ValueError("intentional error")

    async def working_handler(event: AnyEvent) -> None:
        results.append("ok")

    event_bus.subscribe(EventType.BAR, failing_handler)
    event_bus.subscribe(EventType.BAR, working_handler)
    await event_bus.start()

    await event_bus.publish(BarEvent(...))
    await event_bus.flush()

    assert results == ["ok"]  # working_handler는 정상 동작
```

배포 전 체크리스트: [references/component-checklist.md](references/component-checklist.md)
