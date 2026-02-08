# Binance 특화 안전 패턴

## API 설정

### 권한 관리

```
✅ 필수 권한:
- Enable Reading (잔고, 포지션, 주문 조회)
- Enable Spot & Margin Trading (주문 생성/취소)

❌ 절대 부여 금지:
- Enable Withdrawals (출금)
- Enable Internal Transfer (내부 전송)
- Enable Universal Transfer
```

### IP 화이트리스트

```
✅ 반드시 설정:
- 봇 실행 서버의 고정 IP만 등록
- VPN 사용 시 VPN 서버 IP 등록

❌ 금지:
- "Unrestricted" 설정 (모든 IP 허용)
```

---

## Rate Limit

### Binance Futures

| 엔드포인트 | 제한 | 비고 |
|----------|------|------|
| REST API | 1200 req/min | IP 기준 |
| Order | 300 req/10sec | UID 기준 |
| WebSocket | 5 msg/sec (inbound) | 연결당 |
| WebSocket Streams | 200 streams/connection | |

### 안전 마진

```python
# ❌ 제한까지 사용
rate_limit = 1200  # requests per minute

# ✅ 80% 규칙 (20% 마진)
rate_limit = 960   # 1200 * 0.8

# ✅ ccxt의 rateLimit 준수
for symbol in symbols:
    data = exchange.fetch_ohlcv(symbol, "1h")
    await asyncio.sleep(exchange.rateLimit / 1000)  # ms → sec
```

### Rate Limit 초과 시

```
1차 경고: HTTP 429 응답
2차 경고: 일시적 IP 밴 (2-10분)
3차: 장기 IP 밴 (최대 24시간)

대응:
1. 즉시 요청 중단
2. 지수 백오프 (1s → 2s → 4s → 8s)
3. 최대 3회 재시도 후 실패 처리
```

---

## 거래소 점검 시간

### Binance 정기 점검

```
- 시간: 보통 UTC 00:00-04:00 (KST 09:00-13:00)
- 빈도: 월 1-2회 (사전 공지)
- 영향: API 전체 중단, WebSocket 끊김
- 대응: 점검 전 주문 취소, 포지션 정리 또는 하드 SL 설정
```

### 비정기 장애

```
- Flash crash 시 API 응답 지연 (2-30초)
- 유동성 고갈 시 큰 스프레드 (평소의 10-100배)
- WebSocket 끊김 후 재연결 필요
```

---

## 주문 안전 패턴

### Decimal Precision

```python
# ❌ float 직접 사용
amount = equity / price  # 0.12345678901234...

# ✅ 거래소 규격 준수
import ccxt

exchange = ccxt.binance()
exchange.load_markets()

# 수량 정밀도
amount = exchange.amount_to_precision(symbol, raw_amount)
# 가격 정밀도
price = exchange.price_to_precision(symbol, raw_price)
```

### 최소 주문 금액

```python
# Binance Futures 최소 주문: 약 $5 (심볼에 따라 다름)
market = exchange.markets[symbol]
min_notional = float(market.get("limits", {}).get("cost", {}).get("min", 5.0))

if order_notional < min_notional:
    logger.warning(f"Order too small: {order_notional} < {min_notional}")
    return  # 주문 스킵
```

### client_order_id 전략

```python
import uuid
from datetime import datetime, timezone

def generate_client_order_id(strategy: str, symbol: str) -> str:
    """
    멱등성 보장 + 추적 가능한 주문 ID.
    형식: {strategy}_{symbol}_{timestamp}_{uuid4_short}
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    uid = uuid.uuid4().hex[:8]
    clean_symbol = symbol.replace("/", "")
    return f"{strategy}_{clean_symbol}_{ts}_{uid}"

# 예: "tsmom_BTCUSDT_20260208120000_a1b2c3d4"
```

### 주문 상태 확인

```python
# ✅ 주문 후 반드시 상태 확인
async def place_and_verify(exchange, symbol, side, amount, max_retries=3):
    order = await exchange.create_market_order(symbol, side, amount)

    for attempt in range(max_retries):
        await asyncio.sleep(1)
        status = await exchange.fetch_order(order["id"], symbol)

        if status["status"] == "closed":
            return status  # 정상 체결
        elif status["status"] == "canceled":
            raise OrderCancelledError(f"Order {order['id']} was cancelled")
        elif status["status"] == "expired":
            raise OrderExpiredError(f"Order {order['id']} expired")

    raise OrderTimeoutError(f"Order {order['id']} not filled after {max_retries} checks")
```

---

## WebSocket 안정성

### 재연결 패턴

```python
# ✅ 자동 재연결 + 상태 복구
class WebSocketManager:
    async def connect(self):
        while True:
            try:
                async with websockets.connect(url) as ws:
                    self._connected = True
                    await self._on_connect()
                    async for msg in ws:
                        await self._on_message(msg)
            except Exception as e:
                self._connected = False
                logger.error(f"WebSocket disconnected: {e}")
                await asyncio.sleep(5)  # 재연결 대기
                logger.info("Reconnecting...")
```

### Heartbeat

```
- Binance WebSocket: 매 3분 ping 필수
- 10분 무응답 시 서버가 연결 종료
- ccxt Pro가 자동 처리하지만, 직접 구현 시 주의
```

---

## 비상 정지 스크립트

```python
#!/usr/bin/env python3
"""emergency_stop.py — 비상 전량 청산 스크립트."""

import asyncio
import ccxt.async_support as ccxt


async def emergency_stop():
    exchange = ccxt.binance({
        "apiKey": os.environ["BINANCE_API_KEY"],
        "secret": os.environ["BINANCE_SECRET"],
        "options": {"defaultType": "future"},
    })

    try:
        # 1. 열린 주문 전량 취소
        open_orders = await exchange.fetch_open_orders()
        for order in open_orders:
            await exchange.cancel_order(order["id"], order["symbol"])
            print(f"Cancelled: {order['id']}")

        # 2. 포지션 전량 청산 (시장가)
        positions = await exchange.fetch_positions()
        for pos in positions:
            if float(pos["contracts"]) > 0:
                side = "sell" if pos["side"] == "long" else "buy"
                await exchange.create_market_order(
                    pos["symbol"], side, abs(float(pos["contracts"]))
                )
                print(f"Closed: {pos['symbol']} {pos['side']}")

        print("EMERGENCY STOP COMPLETE")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(emergency_stop())
```
