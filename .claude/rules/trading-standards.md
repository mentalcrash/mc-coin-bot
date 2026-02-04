# 🏦 Trading Standards: CCXT Integration

## 1. CCXT Pro Architecture (WebSocket First)

### Library Policy (2026)
- **기본:** 무료화된 **CCXT Pro** 사용
- **Import:** `import ccxt.pro as ccxt`
- **REST API:** 백업용으로만 사용 (`ccxt.async_support`)

### Hybrid Strategy
- **Market Data (실시간):** WebSocket 메서드 사용
  - `watch_ticker`, `watch_order_book`, `watch_trades`
  - 지연 시간 최소화

- **Order Execution:** REST API 권장 (체결 확실성)
  - `create_order`, `cancel_order`, `fetch_order`
  - 초단타(HFT)의 경우에만 `ws_create_order` 허용

---

## 2. Async Lifecycle Management

### Context Managers (필수)
- Exchange 인스턴스는 **반드시** `async with` 블록 안에서 생성 및 관리
- 연결 누수(Connection Leak) 방지 및 `close()` 호출 보장

```python
async with ccxt.binance(config) as exchange:
    await exchange.load_markets()  # 필수!
    # ... 작업 수행
# 자동으로 close() 호출됨
```

### Initialization (필수)
- 인스턴스 생성 직후 `await exchange.load_markets()` 호출 필수
- 최신 정밀도(Precision) 정보 및 심볼 메타데이터 로드

---

## 3. Precision & Type Safety (CRITICAL)

### The "String" Protocol
> [!CAUTION]
> **CCXT API에 가격(Price)이나 수량(Amount)을 전달할 때는 반드시 `str` 타입이어야 합니다.**
>
> `float` 사용은 부동소수점 오차로 인해 **엄격히 금지**됩니다.

### Precision Guards (주문 전 필수)
```python
# Decimal → String 변환 (거래소 규격에 맞춤)
safe_amount = exchange.amount_to_precision(symbol, amount)  # Returns str
safe_price = exchange.price_to_precision(symbol, price)      # Returns str

# 주문 시 String 전달
await exchange.create_order(
    symbol=symbol,
    type="limit",
    side="buy",
    amount=safe_amount,  # str
    price=safe_price     # str
)
```

### Python Type Flow
```
Decimal (비즈니스 로직)
    ↓
amount_to_precision() / price_to_precision()
    ↓
str (CCXT API 전달)
    ↓
Exchange API
```

---

## 4. Error Handling Hierarchy

### 에러 분류 및 대응 전략

| 예외 타입 | 원인 | 대응 전략 |
|----------|------|----------|
| `NetworkError` / `RequestTimeout` | 일시적 네트워크 장애 | **재시도 (Retry)** 로직 수행 |
| `DDoSProtection` | Rate Limit 도달 | `backoff` 시간 대기 후 재시도 |
| `InsufficientFunds` | 잔고 부족 | **즉시 중단 (Abort)**, 관리자 알림 |
| `ExchangeError` | 로직 오류 (심볼 오류, 주문 타입 미지원 등) | **즉시 중단**, 로그 기록 |
| `InvalidOrder` | 주문 파라미터 오류 (최소 수량 미달 등) | **즉시 중단**, 파라미터 검증 강화 |

### 재시도 로직 예시
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout))
)
async def create_order_with_retry(exchange, symbol, side, amount, price):
    return await exchange.create_order(symbol, "limit", side, amount, price)
```

---

## 5. Unified API Usage

### Standard Methods (권장)
- **거래소 고유의 Implicit API 사용 지양**
- **CCXT Unified API 사용 권장:**
  - `fetch_ohlcv`, `fetch_ticker`, `fetch_order_book`
  - `create_order`, `cancel_order`, `fetch_order`
  - `fetch_balance`, `fetch_positions`

### Implicit API Exception (제한적 허용)
- Unified API가 지원하지 않는 특정 기능에 한해서만 허용
- 예: 리슨키 연장, 선물 레버리지 설정
- **주석 필수 작성** (왜 Implicit API를 사용하는지)

```python
# ✅ Unified API (권장)
balance = await exchange.fetch_balance()

# ❌ Implicit API (지양)
# balance = await exchange.binance_private_get_account()

# ⚠️ Implicit API (주석과 함께 제한적 허용)
# Unified API에서 지원하지 않는 리슨키 연장 기능
await exchange.binance_private_put_userDataStream({"listenKey": key})
```

---

## 6. Idempotency (멱등성)

### Client Order ID
- 주문은 `client order ID`로 멱등하게 처리
- 패턴: `client_order_id = f"{strategy}_{symbol}_{timestamp}_{nonce}"`
- 네트워크 오류 시 중복 주문 방지

```python
from datetime import datetime
import uuid

def generate_client_order_id(strategy: str, symbol: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    nonce = uuid.uuid4().hex[:8]
    return f"{strategy}_{symbol.replace('/', '_')}_{timestamp}_{nonce}"

# 사용 예시
client_id = generate_client_order_id("tsmom", "BTC/USDT")
# "tsmom_BTC_USDT_20260204123456_a1b2c3d4"

await exchange.create_order(
    symbol="BTC/USDT",
    type="limit",
    side="buy",
    amount="1.0",
    price="50000",
    params={"clientOrderId": client_id}
)
```

---

## 7. Example Pattern

### ✅ Good (WebSocket & Precision Safe)
```python
import ccxt.pro as ccxt
from decimal import Decimal
from loguru import logger

async def run_market_maker(symbol: str, target_price: Decimal, amount: Decimal) -> None:
    """마켓 메이킹 전략 (WebSocket + REST 하이브리드)"""
    # 1. Exchange Config (WebSocket Default)
    exchange_config = {
        "apiKey": "ENV_VAR",
        "secret": "ENV_VAR",
        "enableRateLimit": True,
        "options": {"defaultType": "future"}  # 선물 거래 명시
    }

    async with ccxt.binance(exchange_config) as exchange:
        # 2. Essential Metadata Loading
        await exchange.load_markets()

        # 3. Precision Handling (Decimal -> String)
        safe_amount = exchange.amount_to_precision(symbol, amount)
        safe_price = exchange.price_to_precision(symbol, target_price)

        try:
            # 4. Hybrid Execution (WS Data + REST Order)
            # 데이터 수신은 WebSocket
            book = await exchange.watch_order_book(symbol)
            best_bid = Decimal(str(book["bids"][0][0]))

            # 주문은 Unified API (REST)
            order = await exchange.create_order(
                symbol=symbol,
                type="limit",
                side="buy",
                amount=safe_amount,  # String
                price=safe_price     # String
            )
            logger.info(f"Order Placed: {order['id']}")

        except ccxt.InsufficientFunds as e:
            logger.critical(f"Balance Error: {e}")
            raise  # Strategy Stop

        except ccxt.NetworkError as e:
            logger.warning(f"Connection unstable: {e}")
            # Retry logic...
```

### ❌ Bad (Sync, Float, Unsafe)
```python
import ccxt  # Sync library (deprecated)

def risky_trade():
    # Sync Library 사용 (Lag 발생)
    exchange = ccxt.binance()

    # load_markets 누락 -> 정밀도 정보 없음

    # Float 사용 위험 (0.001 -> 0.00099999로 전송될 수 있음)
    exchange.create_order("BTC/USDT", "limit", "buy", 0.001, 50000.5)

    # Context manager 미사용 -> 연결 누수
```

---

## 8. WebSocket User Data Stream

### 실시간 주문/포지션 동기화
- **User Data Stream:** 주문 체결, 잔고 변경, 포지션 변경 실시간 수신
- **CCXT Pro:** `watch_orders`, `watch_balance`, `watch_positions`

```python
async def watch_order_updates(exchange, symbol: str):
    """주문 업데이트 실시간 모니터링"""
    while True:
        try:
            orders = await exchange.watch_orders(symbol)
            for order in orders:
                logger.info(f"Order Update: {order['id']} - {order['status']}")
                # 포트폴리오 상태 업데이트
        except Exception as e:
            logger.error(f"Watch orders error: {e}")
            await asyncio.sleep(1)
```

---

## 9. Configuration Best Practices

### Exchange Config Template
```python
from pydantic import BaseModel, SecretStr, Field

class BinanceConfig(BaseModel):
    """Binance 거래소 설정"""
    api_key: str
    api_secret: SecretStr
    testnet: bool = Field(default=False, description="테스트넷 사용 여부")
    default_type: str = Field(default="spot", description="spot | future | margin")
    enable_rate_limit: bool = Field(default=True, description="Rate Limit 자동 처리")

    def to_ccxt_config(self) -> dict:
        """CCXT 설정 딕셔너리 생성"""
        return {
            "apiKey": self.api_key,
            "secret": self.api_secret.get_secret_value(),
            "enableRateLimit": self.enable_rate_limit,
            "options": {
                "defaultType": self.default_type,
                "testnet": self.testnet
            }
        }
```
