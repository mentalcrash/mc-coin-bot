# 암호화폐 트레이딩 특화 안티패턴

암호화폐 시장 고유의 특성으로 인해 전통 주식 시장과 다른 추가 검증이 필요한 항목들.

---

## 1. 24/7 시장 & 타임스탬프

```python
# ❌ 주식 시장 기준의 "일봉" 사용
df_daily = df.resample("D").agg({"open": "first", "close": "last"})
# 문제: "D"는 UTC 00:00 기준 — 거래소마다 일봉 기준 시간이 다름
# Binance: UTC 00:00, Bybit: UTC 00:00, 한국 거래소: KST 00:00

# ✅ 거래소 기준 시간 명시
df_daily = df.resample("D", offset="0h").agg(...)  # UTC 기준 명시
```

## 2. 펀딩비 (Perpetual Futures)

```python
# ❌ 가장 흔한 누락: 펀딩비 미모델링
# Perpetual Future는 8시간마다 펀딩비 발생
# 장기 Long 포지션 + 양의 펀딩비 = 지속적 비용 누적
backtest_pnl = position * (close - entry)  # 펀딩비 없음

# ✅ 펀딩비 포함
def apply_funding(position, funding_rate, notional):
    """양의 펀딩비: Long이 Short에게 지불, 음의 펀딩비: 반대"""
    if position > 0:
        return -abs(funding_rate) * notional  # Long은 보통 비용
    elif position < 0:
        return abs(funding_rate) * notional   # Short은 보통 수취
    return 0

# 실전 펀딩비 범위: -0.1% ~ +0.3% (극단 시 ±1%)
# 연간 누적: 0.01% × 3 × 365 = 10.95% — 무시할 수 없는 비용
```

## 3. 청산(Liquidation) 메커니즘

```python
# ❌ 레버리지 사용하면서 청산 미모델링
leverage = 10
position = equity * leverage
pnl = position * price_change  # 10% 하락 → 100% 손실인데 청산 미처리

# ✅ 청산 체크 필수
def check_liquidation(position, entry_price, current_price, leverage, maint_margin=0.005):
    """
    Binance 교차 마진 기준 청산 가격 계산
    maint_margin: 유지 마진율 (BTC 기준 0.4-0.5%)
    """
    if position > 0:  # Long
        liq_price = entry_price * (1 - 1/leverage + maint_margin)
    else:  # Short
        liq_price = entry_price * (1 + 1/leverage - maint_margin)
    return current_price <= liq_price if position > 0 else current_price >= liq_price
```

## 4. 거래소 다운타임 & 핀바

```python
# ❌ 백테스트에서 연속적 데이터 가정
# 실전: 거래소 점검, API 장애, 네트워크 이슈로 데이터 갭 발생
# 2021-05-19 BTC 급락 시 다수 거래소 API 장애

# ✅ 데이터 갭 탐지 및 처리
def detect_gaps(df, expected_freq="1h"):
    time_diff = df.index.to_series().diff()
    expected = pd.Timedelta(expected_freq)
    gaps = df[time_diff > expected * 1.5]
    if len(gaps) > 0:
        logger.warning(f"Data gaps detected: {len(gaps)} gaps")
    return gaps

# ❌ 핀바(wick) 무시
# 1분봉에서 Close 기준으로만 손절 체크 → 핀바가 손절가를 관통했는데 미체크
if close < stop_loss:  # 핀바의 Low가 이미 stop_loss 아래를 찍었을 수 있음

# ✅ High/Low도 함께 체크
if low <= stop_loss:  # 봉 내 최저가가 손절가 이하
    fill_price = stop_loss  # 또는 보수적으로 low
```

## 5. 슬리피지 — 암호화폐 특화

```python
# 암호화폐 슬리피지 현실 (2026년 기준)
SLIPPAGE_ESTIMATES = {
    "BTC/USDT": 0.01,     # ~0.01% (높은 유동성)
    "ETH/USDT": 0.02,     # ~0.02%
    "Top 20 알트": 0.05,   # ~0.05%
    "Mid-cap 알트": 0.10,  # ~0.10%
    "Small-cap": 0.30,     # ~0.30% 이상
    "급변동 시장": 0.50,   # 급등/급락 시 0.50% 이상
}

# ❌ 모든 코인에 동일 슬리피지
slippage = 0.001  # BTC와 소형 알트에 같은 값

# ✅ 자산별/상황별 동적 슬리피지
def get_slippage(symbol, order_size_usd, volatility_percentile):
    base = SLIPPAGE_BASE[symbol]
    size_impact = (order_size_usd / ADV[symbol]) * 0.1
    vol_multiplier = 1 + (volatility_percentile / 100)
    return base + size_impact * vol_multiplier
```

## 6. 김치 프리미엄 / 거래소 가격 차이

```python
# ❌ 단일 거래소 데이터로 멀티 거래소 전략 백테스트
# Binance 가격 ≠ Upbit 가격 (김치 프리미엄 최대 30%+)

# ❌ 차익거래 전략에서 동시 체결 가정
profit = price_exchange_A - price_exchange_B  # 전송 시간 무시

# ✅ 전송 지연 + 네트워크 수수료 포함
transfer_time_minutes = {"BTC": 30, "ETH": 5, "USDT_TRC20": 3}
network_fee = {"BTC": 0.0001, "ETH": 0.001}
```

## 7. 시장 레짐 변화

```python
# ❌ 상승장에서만 백테스트
# 2021년 데이터만 사용 → 강세장 편향
# BTC: 2021 상승(+60%), 2022 하락(-65%), 2023 회복(+155%), 2024 상승(+120%)

# ✅ 최소 포함해야 할 시장 레짐
REQUIRED_REGIMES = {
    "상승장": "2023-01 ~ 2024-03",    # BTC 16K → 73K
    "하락장": "2022-01 ~ 2022-12",    # BTC 47K → 16K
    "횡보장": "2024-04 ~ 2024-10",    # BTC 60K-70K 레인지
    "급락":   "2022-05 (LUNA)",        # -99% 이벤트
    "급등":   "2024-11 (트럼프 당선)",  # 단기 급등
}

# 레짐별 성과를 반드시 분리 보고
```

## 8. 상폐/프로젝트 실패 리스크

```python
# ❌ Survivorship Bias — 현존하는 코인만 테스트
universe = ["BTC", "ETH", "SOL", "DOGE"]  # 2024년 기준 상위 코인
# 2021년 기준 상위였던 LUNA, FTT 등 포함되지 않음

# ✅ 시점별 유니버스 구성
def get_universe_at(date, top_n=50):
    """해당 시점 기준 시가총액 상위 N개"""
    market_cap = fetch_historical_market_cap(date)
    return market_cap.nlargest(top_n, "market_cap")["symbol"].tolist()
```

## 9. ccxt 라이브러리 특화 주의사항

```python
# ❌ ccxt 반환값 검증 없이 사용
ticker = exchange.fetch_ticker("BTC/USDT")
price = ticker["last"]  # ticker가 None이면? last가 None이면?

# ✅ 방어적 접근
ticker = exchange.fetch_ticker("BTC/USDT")
if ticker is None or ticker.get("last") is None:
    raise ValueError("Failed to fetch ticker")
price = float(ticker["last"])

# ❌ rate limit 미처리
for symbol in symbols:
    data = exchange.fetch_ohlcv(symbol, "1h")  # 수십 개 연속 호출 → ban

# ✅ rate limit 준수
for symbol in symbols:
    data = exchange.fetch_ohlcv(symbol, "1h")
    time.sleep(exchange.rateLimit / 1000)  # ms → sec

# ❌ 주문 상태 미확인
order = exchange.create_limit_buy_order("BTC/USDT", amount, price)
# 주문이 체결되었는지 확인하지 않음

# ✅ 주문 상태 폴링
order = exchange.create_limit_buy_order("BTC/USDT", amount, price)
while True:
    status = exchange.fetch_order(order["id"], "BTC/USDT")
    if status["status"] == "closed":
        break
    elif status["status"] == "canceled":
        logger.warning("Order canceled")
        break
    time.sleep(1)
```

## 10. Decimal vs Float 정밀도

```python
# ❌ float 연산의 누적 오차
total = 0.0
for _ in range(1000):
    total += 0.001  # 기대: 1.0, 실제: 0.9999999999999...

# 주문 수량에서 float 오차 → 거래소 rejection
amount = equity / price  # 0.12345678901234... → 거래소 소수점 제한 초과

# ✅ Decimal 사용 + 거래소 정밀도 준수
from decimal import Decimal, ROUND_DOWN

amount = Decimal(str(equity)) / Decimal(str(price))
# 거래소별 step_size 준수
step_size = Decimal(exchange.markets[symbol]["precision"]["amount"])
amount = amount.quantize(step_size, rounding=ROUND_DOWN)
```
