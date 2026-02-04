# 🧪 Testing Standards: Pytest & Asyncio

## 1. Async Testing Core

### pytest-asyncio Plugin
- **Plugin:** `pytest-asyncio` 사용
- **Marker:** 모든 비동기 테스트 함수에는 `@pytest.mark.asyncio` 마커 필수
- **Alternative:** `pytest.ini`의 `asyncio_mode=auto` 설정으로 자동 감지 가능

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
```

### Event Loop Scope
- **Fixture Scope:** 비동기 픽스처(Fixture)의 스코프 명확히 정의
- `function` (기본): 각 테스트마다 새 이벤트 루프
- `session`: 전체 테스트 세션에서 하나의 이벤트 루프 공유
- **주의:** 스코프 충돌로 인한 이벤트 루프 오류 방지

---

## 2. Mocking Strategy (Strict)

### No Real Network Calls (필수)
> [!CAUTION]
> **단위 테스트(Unit Test)에서는 외부 API 호출이 엄격히 금지됩니다.**

### CCXT Mocking Pattern
- **AsyncMock 사용:** `unittest.mock.AsyncMock`으로 비동기 함수 시뮬레이션
- **반환값 설정:** `return_value` 또는 `side_effect` 사용

```python
from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.fixture
def mock_exchange():
    """CCXT Exchange Mock"""
    exchange = AsyncMock()

    # 비동기 메서드 (await 필요)
    exchange.create_order.return_value = {
        "id": "123456",
        "status": "closed",
        "symbol": "BTC/USDT"
    }

    # 동기 메서드 (정밀도 함수는 동기)
    exchange.amount_to_precision.return_value = "1.0"
    exchange.price_to_precision.return_value = "50000"

    # load_markets도 비동기
    exchange.load_markets.return_value = None

    return exchange
```

### Pydantic Validation Mocking
```python
from pydantic import ValidationError

def test_invalid_order_validation():
    """잘못된 주문 데이터 검증"""
    with pytest.raises(ValidationError) as exc_info:
        Order(symbol="BTC/USDT", price=-100, amount=1)  # 음수 가격

    assert "price" in str(exc_info.value)
```

---

## 3. Test Coverage & Quality

### Critical Paths (90% 이상)
- **주문 집행 (Execution):** 90% 이상 커버리지 목표
- **시그널 생성 (Strategy):** 90% 이상 커버리지 목표
- **포트폴리오 관리 (Portfolio):** 90% 이상 커버리지 목표

### Parametrization (다양한 시나리오)
- `pytest.mark.parametrize` 적극 활용
- 상승장, 하락장, 횡보장 등 다양한 시장 상황 시뮬레이션

```python
import pytest
from decimal import Decimal

@pytest.mark.parametrize("market_condition,expected_signal", [
    ("uptrend", 1),      # 상승장 → 매수 시그널
    ("downtrend", -1),   # 하락장 → 매도 시그널
    ("sideways", 0),     # 횡보장 → 관망
])
def test_strategy_signals(market_condition: str, expected_signal: int):
    """다양한 시장 조건에서 전략 시그널 테스트"""
    data = generate_market_data(market_condition)
    strategy = TSMOMStrategy(config)
    signal = strategy.generate_signal(data)
    assert signal == expected_signal
```

---

## 4. Fixture Organization

### Scope Hierarchy
```python
import pytest
from decimal import Decimal

@pytest.fixture(scope="session")
def exchange_config():
    """전역 설정 (세션 전체 공유)"""
    return {
        "apiKey": "test_key",
        "secret": "test_secret",
        "enableRateLimit": True
    }

@pytest.fixture(scope="function")
def mock_exchange():
    """개별 테스트용 Mock Exchange (매번 새로 생성)"""
    return AsyncMock()

@pytest.fixture
async def order_manager(mock_exchange):
    """OrderManager 인스턴스 (mock_exchange 의존)"""
    manager = OrderManager(exchange=mock_exchange)
    await manager.initialize()
    yield manager
    await manager.cleanup()
```

---

## 5. Testing Patterns

### ✅ Good (Async + Mocking)
```python
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

from src.execution.order_manager import OrderManager
from src.models import Order

@pytest.mark.asyncio
async def test_place_order_success():
    """주문 성공 케이스"""
    # Arrange
    mock_exchange = AsyncMock()
    mock_exchange.create_order.return_value = {
        "id": "123",
        "status": "closed",
        "filled": 1.0
    }
    mock_exchange.amount_to_precision.return_value = "1.0"
    mock_exchange.price_to_precision.return_value = "50000"

    manager = OrderManager(exchange=mock_exchange)

    # Act
    result = await manager.place_buy_order(
        symbol="BTC/USDT",
        amount=Decimal("1.0"),
        price=Decimal("50000")
    )

    # Assert
    assert result["id"] == "123"
    mock_exchange.create_order.assert_awaited_once()

    # 호출 인자 검증
    call_args = mock_exchange.create_order.call_args
    assert call_args.kwargs["symbol"] == "BTC/USDT"
    assert call_args.kwargs["amount"] == "1.0"  # String 타입 확인

@pytest.mark.asyncio
async def test_place_order_insufficient_funds():
    """잔고 부족 에러 케이스"""
    # Arrange
    mock_exchange = AsyncMock()
    mock_exchange.create_order.side_effect = InsufficientFunds("Insufficient balance")

    manager = OrderManager(exchange=mock_exchange)

    # Act & Assert
    with pytest.raises(InsufficientFunds):
        await manager.place_buy_order("BTC/USDT", Decimal("1.0"), Decimal("50000"))
```

### ❌ Bad (Real API Calls)
```python
import ccxt

def test_real_order():  # ❌ 실제 API 호출
    exchange = ccxt.binance({"apiKey": "real_key", "secret": "real_secret"})
    exchange.load_markets()

    # 실제 주문 생성 (위험!)
    order = exchange.create_order("BTC/USDT", "limit", "buy", 0.001, 50000)
    assert order["status"] == "closed"
```

---

## 6. Integration Testing

### 통합 테스트 분리
- **단위 테스트:** Mock 사용, 외부 의존성 없음
- **통합 테스트:** 실제 API 호출 (별도 마커로 분리)

```python
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_exchange_connection():
    """실제 거래소 연결 테스트 (통합 테스트)"""
    exchange = ccxt.binance({"apiKey": "test", "secret": "test"})
    await exchange.load_markets()
    assert "BTC/USDT" in exchange.markets
```

### 테스트 실행 분리
```bash
# 단위 테스트만 실행 (기본)
uv run pytest -m "not integration"

# 통합 테스트 포함
uv run pytest

# 특정 마커만 실행
uv run pytest -m integration
```

---

## 7. Snapshot Testing (선택사항)

### 복잡한 데이터 검증
- 복잡한 지표 계산 결과나 Pydantic 모델 직렬화 결과
- `pytest-snapshot` 또는 `syrupy` 사용

```python
from syrupy.assertion import SnapshotAssertion

def test_strategy_output_snapshot(snapshot: SnapshotAssertion):
    """전략 출력 스냅샷 테스트"""
    strategy = TSMOMStrategy(config)
    result = strategy.calculate_indicators(sample_data)

    # 첫 실행: 스냅샷 생성
    # 이후 실행: 스냅샷과 비교
    assert result.model_dump() == snapshot
```

---

## 8. Test Organization

### 디렉터리 구조
```
tests/
├── unit/                      # 단위 테스트
│   ├── test_strategy.py
│   ├── test_portfolio.py
│   └── test_execution.py
├── integration/               # 통합 테스트
│   ├── test_exchange_api.py
│   └── test_backtest_engine.py
├── fixtures/                  # 공통 픽스처
│   ├── conftest.py
│   └── mock_data.py
└── conftest.py               # 전역 설정
```

### conftest.py 예시
```python
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

@pytest.fixture
def sample_ohlcv():
    """샘플 OHLCV 데이터"""
    return [
        [1609459200000, 29000, 29500, 28800, 29200, 100],
        [1609545600000, 29200, 30000, 29100, 29800, 150],
        [1609632000000, 29800, 30500, 29500, 30200, 200],
    ]

@pytest.fixture
def mock_exchange():
    """공통 Mock Exchange"""
    exchange = AsyncMock()
    exchange.amount_to_precision.return_value = "1.0"
    exchange.price_to_precision.return_value = "50000"
    return exchange
```

---

## 9. Coverage Configuration

### pytest-cov 설정
```ini
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80"
]
```

### 커버리지 실행
```bash
# 커버리지 리포트 생성
uv run pytest --cov=src --cov-report=html

# HTML 리포트 확인
open htmlcov/index.html

# 특정 모듈만 커버리지 확인
uv run pytest --cov=src.strategy --cov-report=term-missing
```

---

## 10. Best Practices Summary

1. **비동기 테스트:** `@pytest.mark.asyncio` 또는 `asyncio_mode=auto`
2. **Mocking 필수:** 단위 테스트에서 실제 API 호출 금지
3. **Parametrize 활용:** 다양한 시나리오 테스트
4. **커버리지 목표:** 핵심 모듈 90% 이상
5. **통합 테스트 분리:** `@pytest.mark.integration` 마커 사용
6. **Fixture 재사용:** `conftest.py`에 공통 픽스처 정의
7. **Type Safety:** 테스트에서도 타입 힌트 적용
8. **명확한 Assert:** `assert result == expected` 보다 `assert result["status"] == "success"` 선호
