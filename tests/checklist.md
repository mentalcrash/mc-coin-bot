# 상세 감사 체크리스트

6단계 각 항목의 상세 검사 기준과 코드 예시.

---

## 1단계: 데이터 무결성 검사

### 1.1 Look-Ahead Bias (미래 참조)

```python
# ❌ 패턴 1: shift 음수
df["next_ret"] = df["close"].shift(-1) / df["close"] - 1
df["signal"] = np.where(df["next_ret"] > 0, 1, -1)  # 미래 수익률로 시그널

# ❌ 패턴 2: pct_change 음수
df["fwd_return"] = df["close"].pct_change(-5)

# ❌ 패턴 3: iloc/loc으로 미래 행 접근
for i in range(len(df)):
    if df["close"].iloc[i+1] > df["close"].iloc[i]:  # i+1 = 미래
        signals.append(1)

# ❌ 패턴 4: 전체 max/min으로 정규화
df["norm"] = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())

# ❌ 패턴 5: 당일 봉의 High/Low를 시그널에 사용
# High/Low는 봉이 완성되어야 확정 → 해당 봉의 시그널에 사용 불가
df["breakout"] = df["close"] > df["high"].rolling(20).max()  # shift 없이

# ✅ 올바른 패턴
df["breakout"] = df["close"] > df["high"].shift(1).rolling(20).max()
```

**grep 탐지:**

```bash
grep -rn "shift(-" --include="*.py"
grep -rn "pct_change(-" --include="*.py"
grep -rn "iloc\[.*+\|iloc\[.*i+1" --include="*.py"
grep -rn "\.min()\|\.max()\|\.mean()\|\.std()" --include="*.py" | grep -v "rolling\|expanding"
```

### 1.2 Data Leakage (데이터 누수)

```python
# ❌ 전체 데이터에 scaler.fit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["scaled"] = scaler.fit_transform(df[["close"]])  # 미래 포함

# ✅ 롤링 또는 expanding 기반
df["z_score"] = (
    (df["close"] - df["close"].expanding().mean()) /
    df["close"].expanding().std()
)

# ❌ train/test 분리 전에 피처 엔지니어링
df["feature"] = some_function(df)  # 전체 데이터
train = df[:split]
test = df[split:]

# ✅ 분리 후 각각 처리
train = df[:split]
test = df[split:]
train["feature"] = some_function(train)
test["feature"] = some_function(test)  # train 통계만 사용
```

### 1.3 Survivorship Bias

- 현재 상장된 코인/주식만으로 백테스트 → 상폐된 자산 무시
- 거래소 상폐 코인(LUNA, FTT 등) 포함 여부 확인
- "상위 N개 시가총액" 선정 시 현재 기준인지, 시점별 기준인지

### 1.4 타임스탬프 정합성

```python
# ❌ UTC와 로컬 시간 혼재
signal_time = pd.Timestamp("2024-01-15 09:00")  # 어떤 타임존?
data_time = pd.Timestamp("2024-01-15 09:00", tz="UTC")

# ✅ 모든 시간을 UTC로 통일
df.index = pd.to_datetime(df.index, utc=True)
```

---

## 2단계: 시그널 로직 검증

### 2.1 시그널 타이밍 (Signal-Execution Gap)

```python
# ❌ 시그널과 동일 봉에서 체결 (불가능)
df["signal"] = np.where(df["close"] > df["sma"], 1, 0)
df["position"] = df["signal"]  # 같은 봉에 진입
df["pnl"] = df["position"] * df["close"].pct_change()  # 시그널 봉 수익 포함

# ✅ 시그널 다음 봉에서 체결
df["signal"] = np.where(df["close"] > df["sma"], 1, 0)
df["position"] = df["signal"].shift(1)  # 다음 봉부터 포지션
df["pnl"] = df["position"] * df["close"].pct_change()
```

### 2.2 0 나눗셈 방어

```python
# ❌ 방어 없음
vol_scalar = target_vol / realized_vol  # realized_vol=0이면 Inf
atr_stop = price - (2 * atr)           # atr=0이면 stop=price (즉시 청산)

# ✅ 방어 코드
realized_vol = max(realized_vol, 1e-8)
vol_scalar = np.clip(target_vol / realized_vol, 0, max_leverage)

atr = max(atr, price * 0.001)  # 최소 0.1% ATR 보장
```

### 2.3 NaN 초기 윈도우

```python
# ❌ NaN 무시하고 시그널 생성
df["sma_200"] = df["close"].rolling(200).mean()
df["signal"] = np.where(df["close"] > df["sma_200"], 1, -1)
# 처음 199개: NaN과 비교 → False → 항상 -1 (Short)

# ✅ NaN 구간 명시적 처리
df["signal"] = np.where(
    df["sma_200"].isna(), 0,  # NaN 구간은 포지션 없음
    np.where(df["close"] > df["sma_200"], 1, -1)
)
```

### 2.4 부호 및 방향 일관성

```python
# ❌ 부호 반전 버그 (흔한 실수)
# 의도: RSI 낮으면 매수, 높으면 매도
df["signal"] = df["rsi"] - 50  # RSI=30 → signal=-20 → Short?? (의도와 반대)

# ✅ 명시적 방향 지정
df["signal"] = np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0))
```

---

## 3단계: 실행 현실성 검사

### 3.1 거래 비용 현실성 기준 (2026년 기준)

| 거래소 | Maker | Taker | 비고 |
|--------|-------|-------|------|
| Binance Futures | 0.02% | 0.05% | VIP 할인 적용 전 |
| Binance Spot | 0.10% | 0.10% | BNB 할인 가능 |
| Bybit | 0.02% | 0.055% | |
| OKX | 0.02% | 0.05% | |

**보수적 백테스트 기준**: 편도 0.05-0.10% (Taker 기준 + 슬리피지)

### 3.2 슬리피지 모델링

```python
# ❌ 슬리피지 없음
fill_price = df["open"].shift(-1)  # 이론적 체결가

# ❌ 고정 슬리피지 (비현실적으로 낮음)
slippage = 0.0001  # 0.01%

# ✅ 동적 슬리피지 (거래량 기반)
def estimate_slippage(order_size, daily_volume, avg_spread=0.0003):
    participation_rate = order_size / daily_volume
    market_impact = participation_rate * 0.1  # 시장 충격
    return avg_spread / 2 + market_impact
```

### 3.3 펀딩비 (Perpetual Futures)

```python
# ❌ 선물 포지션인데 펀딩비 미적용
pnl = position * price_change  # 펀딩비 누락

# ✅ 펀딩비 포함
# 8시간마다 발생, Long이면 양의 펀딩비 = 비용
funding_cost = position * funding_rate * notional_value
pnl = position * price_change - abs(funding_cost)
```

### 3.4 부분 체결 & 유동성

```python
# ❌ 주문 100% 체결 가정
order_qty = desired_position / current_price

# ✅ 유동성 제약 반영
max_order = daily_volume * max_participation_rate  # 예: 일거래량의 1%
order_qty = min(desired_qty, max_order)
if order_qty < desired_qty:
    logger.warning(f"Partial fill: {order_qty}/{desired_qty}")
```

---

## 4단계: 리스크 관리 검증

### 4.1 Stop-Loss 작동 검증

**핵심**: 설정(config)과 실행(execution)을 **모두** 확인. 설정만 있고 실행 없는 것이 가장 흔한 버그.

```python
# ❌ 패턴 1: 설정만 존재
class Strategy:
    def __init__(self):
        self.stop_loss_pct = 0.05  # 설정됨
    
    def on_bar(self, bar):
        if self.should_enter(bar):
            self.enter_position(bar)
        # stop_loss_pct를 체크하는 코드가 없음!

# ❌ 패턴 2: 조건문이 도달 불가능
if unrealized_pnl < -stop_loss:
    close_position()
# 하지만 unrealized_pnl이 업데이트되지 않음

# ✅ 올바른 패턴
def on_bar(self, bar):
    # 포지션 업데이트 → 손절 체크 → 신규 시그널 순서
    self.update_unrealized_pnl(bar)
    if self.check_stop_loss():
        self.close_position(bar, reason="stop_loss")
        return  # 손절 후 신규 진입 방지
    if self.should_enter(bar):
        self.enter_position(bar)
```

### 4.2 포지션 사이징

```python
# ❌ 고정 수량 (자본 대비 비율 무시)
order_size = 1.0  # 항상 1 BTC — 자본이 100만원이든 1억이든

# ❌ 켈리 기준 100% 적용 (과다 레버리지)
kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
position_size = equity * kelly  # Full Kelly = 매우 공격적

# ✅ 리스크 기반 사이징
risk_per_trade = equity * 0.02  # 거래당 최대 2% 리스크
stop_distance = entry_price * stop_loss_pct
position_size = risk_per_trade / stop_distance
position_size = min(position_size, equity * max_position_pct)
```

### 4.3 시스템 레벨 가드레일

다음이 **모두** 코드에 구현되어 있는지 확인:

```python
# 필수 가드레일 체크리스트
assert hasattr(system, 'max_daily_loss')       # 일일 최대 손실
assert hasattr(system, 'max_drawdown')         # 최대 드로다운 → 긴급 정지
assert hasattr(system, 'max_open_positions')   # 최대 동시 포지션 수
assert hasattr(system, 'max_leverage')         # 최대 레버리지
assert hasattr(system, 'cooldown_after_loss')  # 연속 손실 시 쿨다운
```

---

## 5단계: 백테스트 신뢰도 검사

### 5.1 과적합 지표

| 지표 | 정상 범위 | 과적합 의심 |
|------|----------|------------|
| Sharpe Ratio | 0.5 - 2.0 | > 3.0 |
| Profit Factor | 1.2 - 2.0 | > 3.0 |
| Win Rate | 40% - 65% | > 80% |
| 연간 수익률 | 10% - 100% | > 200% |
| 최대 드로다운 | 10% - 30% | < 5% (의심) |

Sharpe > 2.0이면 "왜 이렇게 높은가?" 반드시 질문. 거래 비용/슬리피지 누락이 가장 흔한 원인.

### 5.2 IS/OOS 분리

```python
# ❌ 전체 데이터로 학습 + 평가
model.fit(all_data)
score = model.evaluate(all_data)  # 학습 데이터로 평가

# ❌ 무작위 분할 (시계열에 부적합)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
# 시간순이 깨짐 → 미래 데이터가 train에 포함 가능

# ✅ 시간순 분할
split_idx = int(len(df) * 0.7)
train = df[:split_idx]
test = df[split_idx:]  # 반드시 학습 기간 이후

# ✅ Walk-Forward
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### 5.3 통계적 유의성

확인할 항목:

- 총 거래 횟수: 최소 100회 이상이어야 통계적 의미
- 시장 레짐별 성과 분해: 상승장에서만 수익이면 방향성 편향
- Monte Carlo 시뮬레이션: 거래 순서 랜덤 셔플 후 결과 분포
- Deflated Sharpe Ratio: 다중 전략 테스트 보정

---

## 6단계: 코드 품질 & 운영 안정성

### 6.1 에러 핸들링

```python
# ❌ API 호출에 에러 핸들링 없음
order = exchange.create_order(symbol, "market", "buy", amount)

# ✅ 재시도 + 타임아웃 + 로깅
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=10),
    retry=tenacity.retry_if_exception_type(ccxt.NetworkError)
)
def place_order(exchange, symbol, side, amount):
    try:
        order = exchange.create_order(symbol, "market", side, amount)
        logger.info(f"Order placed: {order['id']} {side} {amount} {symbol}")
        return order
    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds: {e}")
        raise
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error: {e}")
        raise
```

### 6.2 상태 관리 (라이브 트레이딩)

```python
# ❌ 메모리에만 포지션 저장
self.positions = {}  # 프로세스 재시작 시 유실

# ✅ 영속적 상태 + 거래소 동기화
def sync_positions(self):
    """거래소 실제 포지션과 로컬 상태 동기화"""
    exchange_positions = self.exchange.fetch_positions()
    local_positions = self.db.load_positions()
    discrepancies = self.compare(exchange_positions, local_positions)
    if discrepancies:
        logger.critical(f"Position mismatch: {discrepancies}")
        self.alert_admin(discrepancies)
```

### 6.3 시크릿 관리

```bash
# ❌ 탐지 패턴
grep -rn "api_key\s*=\s*['\"]" --include="*.py"
grep -rn "api_secret\s*=\s*['\"]" --include="*.py"
grep -rn "password\s*=\s*['\"]" --include="*.py"
```

올바른 방식: `.env` 파일 + `python-dotenv` 또는 시크릿 매니저
