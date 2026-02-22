---
paths:
  - "src/exchange/**"
---

# Exchange Integration Rules (CCXT)

## Async Lifecycle (Critical)

**반드시** `async with` 컨텍스트 매니저 사용:

```python
async with ccxt.binance(config) as exchange:
    await exchange.load_markets()  # 필수! Precision 정보 로드
    # ... operations
```

`load_markets()` 누락 → Precision 없음 → 주문 실패.

## String Protocol (Critical)

> CCXT API에 가격/수량 전달 시 반드시 `str` 타입. `float` 사용 금지.

```python
amount = exchange.amount_to_precision(symbol, Decimal("0.001"))
price = exchange.price_to_precision(symbol, Decimal("50000"))
await exchange.create_order(symbol, "limit", "buy", amount, price)
```

Type flow: `Decimal` → `amount_to_precision()` → `str` → CCXT API

## CCXT Exception Hierarchy (Critical)

`RateLimitExceeded`는 `NetworkError`의 서브클래스 — except 순서 중요:

```
BaseError
├── ExchangeError
│   ├── InsufficientFunds
│   └── InvalidOrder
└── NetworkError
    ├── RateLimitExceeded  ← 반드시 먼저 catch
    ├── DDoSProtection
    └── RequestTimeout
```

## Error Handling

| Exception | Strategy |
|-----------|----------|
| `RateLimitExceeded` | Wait + exponential backoff |
| `NetworkError` | Retry with backoff |
| `InsufficientFunds` | Abort + notify |
| `ExchangeError` | Abort + log |
| `InvalidOrder` | Abort + validate params |

## Idempotency (Client Order ID)

```python
client_id = f"{strategy}_{symbol.replace('/', '_')}_{timestamp}_{nonce}"
await exchange.create_order(
    symbol, "limit", "buy", amount, price,
    params={"clientOrderId": client_id}
)
```
