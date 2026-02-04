# 🐍 Python 3.13 Standards & Pydantic V2 Modeling

## 1. Python 3.13 Modern Syntax

### Core Principles
- **Version:** Python 3.13+ 문법을 엄격히 준수
- **Modern Typing:**
    - `Union[X, Y]` 대신 `X | Y` 문법 사용
    - `Optional[X]` 대신 `X | None` 사용
    - 리턴 타입이 자기 자신일 경우 `from typing import Self` 사용하여 `-> Self`로 명시
    - 컬렉션 타입은 `List`, `Dict` 대신 내장 제네릭 `list[]`, `dict[]` 사용

### Async & Concurrency (Critical for Trading)
- **Structured Concurrency:** `asyncio.gather()`보다는 예외 처리가 안전한 **`asyncio.TaskGroup`** (Python 3.11+) 패턴 우선 사용
- **Non-blocking:** I/O 바운드 작업(네트워크, DB)은 반드시 `await` 키워드와 함께 비동기 함수로 작성
- **금지:** `time.sleep()` 절대 금지, `await asyncio.sleep()` 사용

### Quant/Financial Precision
- **No Floats for Money:** 가격(Price), 수량(Amount), 잔고(Balance) 계산에는 절대 `float` 사용 금지
- **반드시 `decimal.Decimal` 사용**
- 나눗셈 연산 시 `getcontext().prec` 확인 또는 양자화(`quantize`) 처리

### Code Style
- **Early Return:** 중첩된 `if/else` 블록을 피하고, Guard Clause(조건 불만족 시 즉시 리턴) 패턴 사용
- **Docstrings:** Google Style Docstring 적용, 모든 Public 함수와 클래스에는 설명, 인자(Args), 반환값(Returns), 발생 예외(Raises) 명시

---

## 2. Pydantic V2 Modeling Standards

### Core Principles (V2 Native)
- **Rust Core Utilization:** Pydantic V2의 성능 이점을 위해 최신 문법 준수
- **No V1 Syntax:** `class Config:` 대신 `model_config = ConfigDict(...)` 사용
- **Methods:**
    - `dict()` (Legacy) ❌ → **`model_dump()`** ✅
    - `parse_obj()` (Legacy) ❌ → **`model_validate()`** ✅
    - `parse_raw()` (Legacy) ❌ → **`model_validate_json()`** ✅

### Immutability & Safety
- **Frozen Models:** 트레이딩 데이터(주문 정보, 체결 내역)는 생성 후 변경되면 안 됨
- 기본적으로 `model_config = ConfigDict(frozen=True)` 적용하여 불변 객체로 생성
- 이는 데이터의 스레드 안전성(Thread-safety)을 높이고 해시 가능(Hashable)하게 만듦

### Configuration Management (pydantic-settings)
- **BaseSettings:** 환경 변수(.env) 관리는 반드시 `pydantic-settings` 패키지의 `BaseSettings` 상속
- **Secrets:** API Key, Secret Key 등은 `str` 대신 `SecretStr` 타입 사용하여 로그 출력 시 자동 마스킹(`**********`)

### Field Validation
- **Field Validators:** 단일 필드 검증은 `@field_validator` 사용
- **Model Validators:** 여러 필드 간의 관계 검증은 `@model_validator(mode='after')` 사용
- **Computed Fields:** 직렬화 시 계산된 값을 포함해야 할 경우 `@property` 대신 `@computed_field` 데코레이터 사용

### Trading Specific
- **Decimal Support:** 금액과 수량은 `float` 대신 `Decimal` 강제
- **Alias Handling:** 거래소 API 응답(camelCase)을 Python 스타일(snake_case)로 매핑 시, `Field(alias="orderId")` 또는 `alias_generator` 활용

---

## 3. Example Patterns

### ✅ Good (Modern & Safe)
```python
from decimal import Decimal
import asyncio
from typing import Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, SecretStr, computed_field

class ExchangeConfig(BaseModel):
    """거래소 연결 설정 모델"""
    model_config = ConfigDict(frozen=True)

    api_key: str
    api_secret: SecretStr

    @field_validator("api_key")
    @classmethod
    def check_key_length(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("API Key seems too short")
        return v

class Order(BaseModel):
    """주문 모델"""
    model_config = ConfigDict(frozen=True)

    symbol: str
    price: Decimal = Field(..., gt=0)
    amount: Decimal = Field(..., gt=0)

    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """총 주문 금액 자동 계산"""
        return self.price * self.amount

async def process_orders(orders: list[Order]) -> None:
    """구조화된 동시성으로 주문 처리"""
    async with asyncio.TaskGroup() as tg:
        for order in orders:
            tg.create_task(execute_order(order))
```

### ❌ Bad (Legacy & Unsafe)
```python
from typing import List, Union, Optional  # Legacy typing
import asyncio

class Order:
    def __init__(self, symbol: str, price: float):  # float 사용 위험
        self.symbol = symbol
        self.price = price  # 변경 가능 (mutable)

    class Config:  # V1 문법
        allow_mutation = False

async def process_orders(orders: List[dict]):  # 타입 불명확
    tasks = []
    for order in orders:
        total = order['price'] * order['amount']  # float 연산 위험
        tasks.append(execute_order(order))
    await asyncio.gather(*tasks)  # 예외 처리 불명확
```
