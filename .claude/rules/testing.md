---
paths:
  - "tests/**"
  - "tests/conftest.py"
---

# Testing Rules

## No Real Network Calls (Critical)

단위 테스트에서 외부 API 호출 금지. `AsyncMock`으로 CCXT 시뮬레이션:

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.create_order.return_value = {"id": "123456", "status": "closed"}
    exchange.amount_to_precision.return_value = "1.0"
    exchange.price_to_precision.return_value = "50000"
    return exchange
```

## Parallel Execution (pytest-xdist)

기본: `-n auto --dist worksteal --timeout=60`

- 디버깅 시: `uv run pytest -p no:xdist`
- session-scoped fixture는 worker별 독립 생성 (공유 아님)

## Auto Markers (디렉토리 기반)

`tests/conftest.py`의 `pytest_collection_modifyitems`가 경로 기반 마커 자동 부여:

| Directory | Marker |
|-----------|--------|
| `/strategy/` | `strategy` |
| `/eda/` | `eda` |
| `/core/`, `/models/`, `/config/`, `/market/`, `/regime/`, `/monitoring/`, `/catalog/`, `/portfolio/`, `/logging/` | `unit` |
| `/backtest/`, `/orchestrator/`, `/notification/`, `/exchange/`, `/pipeline/`, `/cli/` | `integration` |
| `/chaos/` | `chaos` |
| `/data/` | `data` |
| `/regression/` | `slow` |

## Coverage Target

- **핵심 모듈 (execution, strategy, portfolio):** 90%+
- **전체:** 80%+

## EDA Testing Patterns

EventBus 기반 이벤트 흐름 테스트:

```python
@pytest.fixture
def event_bus():
    return EventBus(max_queue_size=100)

async def test_bar_to_fill_flow(event_bus):
    collected = []
    async def handler(event):
        collected.append(event)
    event_bus.subscribe(EventType.FILL, handler)

    await event_bus.publish(bar_event)
    await event_bus.flush()  # 필수!

    assert len(collected) == 1
```

**핵심 규칙:**

- `flush()` 호출 필수 — bar-by-bar 동기 처리 보장
- 이벤트 순서 검증: `BAR → SIGNAL → ORDER → FILL`
- BacktestExecutor: 일반 주문은 다음 bar open에 체결, SL/TS는 즉시 체결
