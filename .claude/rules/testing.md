---
paths:
  - "tests/**"
  - "conftest.py"
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
# conftest.py
@pytest.fixture(scope="session")
def exchange_config():
    """전역 설정 (세션 공유)"""
    return {"apiKey": "test", "secret": "test"}

@pytest.fixture(scope="function")
def mock_exchange():
    """테스트별 Mock (매번 새로 생성)"""
    return AsyncMock()
```
