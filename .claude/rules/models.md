---
paths:
  - "src/models/**"
---

# Pydantic V2 Model Rules

## Core Principles

- **V2 문법만** (`class Config:` 금지 → `model_config = ConfigDict(...)`)
- **불변 객체:** 트레이딩 데이터는 `frozen=True`
- **금융 정밀도:** `Decimal` 필수, `float` 금지

## ConfigDict

```python
from pydantic import BaseModel, ConfigDict, Field
from decimal import Decimal

class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    price: Decimal = Field(..., gt=0)
    amount: Decimal = Field(..., gt=0)
```

## Validators

```python
from pydantic import field_validator, model_validator

class Config(BaseModel):
    @field_validator("api_key")
    @classmethod
    def check_key_length(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("API Key too short")
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        return self
```

## Computed Fields

```python
from pydantic import computed_field

class Order(BaseModel):
    price: Decimal
    amount: Decimal

    @computed_field
    @property
    def notional_value(self) -> Decimal:
        return self.price * self.amount
```

## Secrets (API Keys)

```python
from pydantic import SecretStr

class ExchangeConfig(BaseModel):
    api_key: str
    api_secret: SecretStr  # 로그 출력 시 마스킹
```
