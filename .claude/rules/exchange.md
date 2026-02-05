---
paths:
  - "src/exchange/**"
---

# Exchange Integration Rules (CCXT)

## CCXT Pro Architecture

- **Import:** `import ccxt.pro as ccxt`
- **Market Data:** WebSocket (`watch_ticker`, `watch_order_book`)
- **Order Execution:** REST API (`create_order`, `cancel_order`)

## Async Lifecycle (Critical)

**반드시** `async with` 컨텍스트 매니저 사용:

```python
async with ccxt.binance(config) as exchange:
    await exchange.load_markets()  # 필수!
    # ... operations
# 자동으로 close() 호출
```

**`load_markets()` 누락 시** Precision 정보가 없어 주문 실패합니다.

## String Protocol (Critical)

> **CCXT API에 가격/수량 전달 시 반드시 `str` 타입**

`float` 사용 금지 (부동소수점 오차):

```python
# ❌ Bad (float precision error)
await exchange.create_order("BTC/USDT", "limit", "buy", 0.001, 50000.5)

# ✅ Good (string with precision)
amount = exchange.amount_to_precision(symbol, Decimal("0.001"))
price = exchange.price_to_precision(symbol, Decimal("50000"))
await exchange.create_order(symbol, "limit", "buy", amount, price)
```

## Type Flow

```
Decimal (business logic)
    ↓
amount_to_precision() / price_to_precision()
    ↓
str (CCXT API)
```

## Error Handling

| Exception | Cause | Strategy |
|-----------|-------|----------|
| `NetworkError` | Temporary network issue | **Retry** with backoff |
| `DDoSProtection` | Rate limit reached | **Wait** then retry |
| `InsufficientFunds` | Balance shortage | **Abort** + notify |
| `ExchangeError` | Logic error | **Abort** + log |
| `InvalidOrder` | Parameter error | **Abort** + validate |

## Idempotency (Client Order ID)

중복 주문 방지를 위해 Client Order ID 사용:

```python
client_id = f"{strategy}_{symbol.replace('/', '_')}_{timestamp}_{nonce}"

await exchange.create_order(
    symbol=symbol,
    type="limit",
    side="buy",
    amount=safe_amount,
    price=safe_price,
    params={"clientOrderId": client_id}
)
```

## WebSocket User Data Stream

실시간 주문/잔고 동기화:

```python
async def watch_order_updates(exchange, symbol: str):
    while True:
        orders = await exchange.watch_orders(symbol)
        for order in orders:
            logger.info(f"Order: {order['id']} - {order['status']}")
```
