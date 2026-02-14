---
paths:
  - "tests/**"
  - "tests/conftest.py"
---

# Testing Rules

## pytest-asyncio Setup

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
```

또는 마커 사용:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_operation()
    assert result is not None
```

## No Real Network Calls (Critical)

> **단위 테스트에서 외부 API 호출 금지**

`AsyncMock`으로 CCXT 시뮬레이션:

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()

    # Async methods
    exchange.create_order.return_value = {
        "id": "123456",
        "status": "closed",
        "symbol": "BTC/USDT"
    }
    exchange.load_markets.return_value = None

    # Sync methods (precision)
    exchange.amount_to_precision.return_value = "1.0"
    exchange.price_to_precision.return_value = "50000"

    return exchange
```

## AAA Pattern

```python
@pytest.mark.asyncio
async def test_place_order_success(mock_exchange):
    # Arrange
    manager = OrderManager(exchange=mock_exchange)

    # Act
    result = await manager.place_buy_order(
        symbol="BTC/USDT",
        amount=Decimal("1.0"),
        price=Decimal("50000")
    )

    # Assert
    assert result["id"] == "123456"
    mock_exchange.create_order.assert_awaited_once()
```

## Parametrize

다양한 시장 조건 테스트:

```python
@pytest.mark.parametrize("market,expected", [
    ("uptrend", 1),
    ("downtrend", -1),
    ("sideways", 0),
])
def test_strategy_signals(market: str, expected: int):
    data = generate_market_data(market)
    signal = strategy.generate_signal(data)
    assert signal == expected
```

## Coverage Target

- **핵심 모듈 (execution, strategy, portfolio):** 90%+
- **전체:** 80%+

```bash
# Coverage with HTML report
uv run pytest --cov=src --cov-report=html --cov-fail-under=80
```

## Integration Test Separation

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_exchange_connection():
    """실제 거래소 연결 (통합 테스트)"""
    ...
```

```bash
# Unit tests only (default)
uv run pytest -m "not integration"

# Include integration
uv run pytest
```

## Fixture Organization

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def exchange_config():
    """전역 설정 (세션 공유)"""
    return {"apiKey": "test", "secret": "test"}

@pytest.fixture(scope="function")
def mock_exchange():
    """테스트별 Mock (매번 새로 생성)"""
    return AsyncMock()
```

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

    # Publish BAR → flush로 동기 처리
    await event_bus.publish(bar_event)
    await event_bus.flush()

    assert len(collected) == 1
```

**핵심 규칙:**
- `flush()` 호출 필수 — bar-by-bar 동기 처리 보장
- 이벤트 순서 검증: `BAR → SIGNAL → ORDER → FILL`
- BacktestExecutor: 일반 주문은 다음 bar open에 체결, SL/TS는 즉시 체결
