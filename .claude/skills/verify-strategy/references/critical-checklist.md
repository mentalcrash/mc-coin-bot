# Critical Checklist (C1-C7) — 상세 검증 가이드

> 이 프로젝트의 전략 코드 패턴에 특화된 검증 기준.
> **1개라도 FAIL이면 Gate 0B 불통과.**

---

## C1: Look-Ahead Bias (미래 참조)

### 정의

시그널 생성 시 해당 시점에서 아직 관측할 수 없는 미래 데이터를 사용하는 결함.
**백테스트 결과를 완전히 무효화**하는 가장 치명적인 오류.

### 탐지 패턴

#### 직접적 미래 참조

```python
# FAIL: shift 음수 — 미래 값 직접 사용
df["signal"] = df["close"].shift(-1)
df["fwd_ret"] = df["close"].pct_change(-5)

# FAIL: iloc/loc으로 미래 행 접근
for i in range(len(df)):
    if df["close"].iloc[i+1] > df["close"].iloc[i]:
        signals.append(1)
```

**grep 탐지:**

```bash
grep -rn 'shift(-' src/strategy/{name}/
grep -rn 'pct_change(-' src/strategy/{name}/
grep -rn 'iloc\[.*+' src/strategy/{name}/
```

#### 전체 기간 통계

```python
# FAIL: 전체 데이터의 min/max/mean/std 사용
df["norm"] = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())
z = (x - df["close"].mean()) / df["close"].std()

# PASS: rolling 또는 expanding 사용
df["z"] = (df["close"] - df["close"].rolling(30).mean()) / df["close"].rolling(30).std()
df["z"] = (df["close"] - df["close"].expanding().mean()) / df["close"].expanding().std()
```

**grep 탐지:**

```bash
# rolling/expanding 없는 .mean()/.std()/.min()/.max() 탐지
grep -n '\.mean()\|\.std()\|\.min()\|\.max()' src/strategy/{name}/ | grep -v 'rolling\|expanding\|axis'
```

#### 당봉 High/Low 사용 (이 프로젝트에서 가장 흔한 위반)

```python
# FAIL: 당봉 High/Low는 봉 완성 전에는 미확정
# signal.py에서:
breakout = df["close"] > df["high"].rolling(20).max()   # shift 없음!
squeeze = df["low"] < df["low"].rolling(10).min()        # shift 없음!

# PASS: shift(1)로 전봉 기준
breakout = df["close"] > df["high"].shift(1).rolling(20).max()
squeeze = df["low"].shift(1) < df["low"].shift(1).rolling(10).min()
```

#### 시그널-체결 동시성

```python
# FAIL: Close에서 시그널 생성 + 같은 봉 Close에 체결
df["signal"] = np.where(df["close"] > sma, 1, 0)
df["position"] = df["signal"]  # shift 없음 = 같은 봉 체결
df["pnl"] = df["position"] * df["close"].pct_change()

# PASS: Signal[t] -> Execute[t+1]
df["signal"] = np.where(df["close"] > sma, 1, 0)
df["position"] = df["signal"].shift(1)  # 다음 봉에 적용
```

### 이 프로젝트 검증 방법

1. `signal.py`의 `generate_signals()` 함수 열기
2. 시그널 계산에 사용되는 **모든 변수**를 나열
3. 각 변수가 `shift(1)` 이상 적용되었는지 **한 줄씩** 확인
4. `preprocessor.py`에서 계산된 중간 지표도 추적 — `preprocess()` 단계에서 shift 없이 계산 후 `signal.py`에서 shift하는 것은 **허용** (이 프로젝트의 패턴)
5. 단, `signal.py`에서 `df["close"]`, `df["high"]`, `df["low"]`, `df["volume"]` 원본 컬럼을 shift 없이 직접 사용하면 **FAIL**

---

## C2: Data Leakage (데이터 누수)

### 정의

학습/피팅에 미래 데이터가 포함되어 성과가 과대평가되는 결함.

### 탐지 패턴

#### sklearn/statsmodels fit

```python
# FAIL: 전체 데이터에 fit
scaler.fit(df[["close"]])
model.fit(df[["feature1", "feature2"]], df["target"])

# PASS: expanding window 또는 rolling window
# HMM Regime 전략 예시 — expanding window fit
for i in range(min_train, len(df)):
    model.fit(df[:i])  # 과거 데이터만 사용
    prediction[i] = model.predict(df[i:i+1])
```

**grep 탐지:**

```bash
grep -n '\.fit(' src/strategy/{name}/
grep -n '\.fit_transform(' src/strategy/{name}/
grep -n 'train_test_split' src/strategy/{name}/
```

#### rolling window 길이 초과

```python
# FAIL: lookback이 전체 데이터보다 크면 expanding과 동일 -> 마지막 행은 전체 통계 포함
df["feature"] = df["close"].rolling(99999).mean()

# PASS: lookback이 합리적 범위
df["feature"] = df["close"].rolling(20).mean()
```

### 이 프로젝트 검증 방법

1. `preprocessor.py`에서 `rolling()`, `expanding()`, `ewm()` 호출 확인
2. 윈도우 크기가 `config.py`의 파라미터로 제어되는지 확인
3. ML 기반 전략(HMM 등)은 `fit()` 호출 시점이 expanding window인지 확인
4. `signal.py`에서 전체 DataFrame 통계(`df.mean()`, `df.quantile()` 등) 미사용 확인

---

## C3: Survivorship Bias

### 정의

현재 존재하는 자산만으로 과거를 백테스트하여 성과가 과대평가되는 결함.

### 이 프로젝트 검증 방법

1. 전략이 에셋 리스트를 하드코딩하는지 확인
2. 백테스트 기간(2020-01 ~ 2025-12) 동안 에셋이 상장되어 있었는지 확인
3. SOL/USDT(2020-08 상장) 등 중간 상장 에셋의 NaN 처리 확인

| 에셋 | 상장일 | 백테스트 시작 가능 |
|------|--------|-------------------|
| BTC/USDT | 2017+ | 2020-01 |
| ETH/USDT | 2017+ | 2020-01 |
| BNB/USDT | 2017+ | 2020-01 |
| SOL/USDT | 2020-08 | 2020-09 (1개월 여유) |
| DOGE/USDT | 2019+ | 2020-01 |

> 이 항목은 전략 코드 자체보다 **백테스트 설정**에 가까움.
> 전략 코드에 에셋 필터링 로직이 있으면 검증, 없으면 PASS.

---

## C4: Signal Vectorization

### 정의

벡터 연산의 잘못된 적용으로 시그널이 의도와 다르게 생성되는 결함.

### 탐지 패턴

#### DataFrame 루프

```python
# FAIL: for 루프로 시그널 생성 (느리고 오류 가능성 높음)
signals = []
for i in range(len(df)):
    if df["close"].iloc[i] > df["sma"].iloc[i]:
        signals.append(1)
    else:
        signals.append(0)

# PASS: 벡터화
df["signal"] = np.where(df["close"] > df["sma"], 1, 0)
```

**grep 탐지:**

```bash
grep -n 'for.*range.*len' src/strategy/{name}/
grep -n 'iterrows\|itertuples' src/strategy/{name}/
grep -n '\.append(' src/strategy/{name}/ | grep -v '__all__\|import'
```

#### pandas 인덱스 불일치

```python
# FAIL: 다른 길이의 Series 연산
series_a = df["close"].rolling(10).mean()     # NaN 9개
series_b = df["close"].rolling(20).mean()     # NaN 19개
df["signal"] = series_a - series_b            # 앞 19개는 NaN, 10~19는 NaN + 값 = NaN
# NaN 전파를 고려하지 않으면 의도와 다른 결과

# PASS: NaN 처리 명시
df["signal"] = (series_a - series_b).fillna(0)  # 또는 dropna 후 처리
```

#### fillna() 부적절 사용

```python
# FAIL: vol_scalar의 NaN을 0으로 채움 -> 포지션 0 = 시그널 무시
df["strength"] = df["direction"] * df["vol_scalar"].fillna(0)
# 초기 구간에서 vol_scalar가 NaN이면 strength도 0 -> 거래 미발생
# 이것이 의도된 동작인지, 아니면 1.0으로 채워야 하는지 확인 필요

# WARNING: fillna(1)도 위험할 수 있음 — 변동성 정보 없이 full position
# PASS: NaN 구간은 명시적으로 entries=False 처리
```

### 이 프로젝트 검증 방법

1. `preprocessor.py`와 `signal.py`에서 `for` 루프 사용 여부 확인
2. `fillna()` 호출마다 채우는 값의 **경제적 의미** 검증
3. `np.where()` 조건에서 NaN 동작 확인 (NaN 비교는 항상 False)
4. Series 간 연산에서 인덱스 정렬 문제 없는지 확인

---

## C5: Position Sizing

### 정의

vol-target 기반 포지션 사이징에서 0 나눗셈, 레버리지 초과, 부호 오류로 자금 위험 발생.

### 탐지 패턴

#### 0 나눗셈

```python
# FAIL: realized_vol이 0이면 Inf
vol_scalar = config.vol_target / realized_vol

# PASS: min_volatility 하한 적용
realized_vol = np.maximum(realized_vol, config.min_volatility)
vol_scalar = config.vol_target / realized_vol
```

**grep 탐지:**

```bash
grep -n 'vol_target.*/' src/strategy/{name}/
grep -n 'target_vol.*/' src/strategy/{name}/
grep -n 'min_volatility\|min_vol' src/strategy/{name}/
```

#### Annualization Factor

```python
# FAIL: 주식 기준 252 사용 (크립토는 24/7)
annualized_vol = daily_vol * np.sqrt(252)

# PASS: 크립토 기준 365 (또는 config에서 설정)
annualized_vol = daily_vol * np.sqrt(config.annualization_factor)  # 365
```

#### strength 부호 일관성

```python
# FAIL: direction=-1 (Short)인데 strength가 양수
direction = -1
vol_scalar = 0.8  # 양수
strength = vol_scalar  # 양수 — direction과 불일치!
# PM이 direction * |strength|로 처리한다면 괜찮지만, strength 자체에 부호를 기대하면 FAIL

# PASS: strength = direction * vol_scalar
strength = direction * vol_scalar  # -0.8 — 명시적 부호
```

### 이 프로젝트 검증 방법

1. `preprocessor.py`에서 `vol_scalar` 계산 과정 추적
2. `realized_vol` 계산에 `min_volatility` 하한 적용 확인
3. `vol_scalar` clip 여부 (max_leverage_cap) 확인 — PM에서 처리할 수도 있음
4. `signal.py`에서 `strength = direction * vol_scalar` 패턴 확인
5. `config.py`의 `annualization_factor` 기본값 확인 (365여야 함)

---

## C6: Cost Model

### 정의

거래 비용이 백테스트에 반영되지 않아 성과가 과대평가되는 결함.

### 이 프로젝트의 비용 구조

| 항목 | 기본값 | 적용 위치 |
|------|--------|----------|
| Maker Fee | 0.02% | BacktestEngine / EDA CostModel |
| Taker Fee | 0.04% | BacktestEngine / EDA CostModel |
| Slippage | 0.05% | BacktestEngine / EDA CostModel |
| Funding Rate | 0.01%/8h | BacktestEngine |
| Market Impact | 0.02% | BacktestEngine |
| **편도 합계** | **~0.11%** | |

### 탐지 패턴

```python
# FAIL: 전략 코드에서 비용을 0으로 오버라이드
class MyStrategy(BaseStrategy):
    @classmethod
    def recommended_config(cls):
        return {
            "cost_model": CostModel(fee=0, slippage=0),  # 비용 제거!
        }

# FAIL: 전략 내부에서 자체 PnL 계산 시 비용 미적용
pnl = position * price_change  # 비용 없음
```

### 이 프로젝트 검증 방법

1. `strategy.py`의 `recommended_config()`에서 비용 관련 설정이 0이 아닌지 확인
2. 전략 코드 내부에서 자체 PnL/equity 계산이 있는지 확인 (있으면 FAIL — 전략은 시그널만)
3. `config.py`에 비용 관련 파라미터가 있으면 기본값 확인

---

## C7: Entry/Exit Logic

### 정의

진입/청산 조건의 논리적 모순으로 불가능한 거래가 발생하는 결함.

### 탐지 패턴

#### 동시 Long + Short

```python
# FAIL: 같은 bar에서 entries=True이면서 direction이 변동
entries = pd.Series(True, index=df.index)  # 매 bar 진입?
direction = np.where(condition, 1, -1)      # 매 bar 방향 전환?
# -> 매 bar 포지션 반전 = 과도한 거래비용

# PASS: 방향 변경 시에만 entry
direction_change = direction != direction.shift(1)
entries = direction_change & (direction != 0)
```

#### ShortMode 처리

```python
# FAIL: ShortMode.DISABLED인데 숏 시그널 잔존
if config.short_mode == ShortMode.DISABLED:
    pass  # 아무 처리 없음 — direction=-1, strength<0 그대로 통과

# PASS: 숏 시그널 제거
if config.short_mode == ShortMode.DISABLED:
    strength = np.where(direction < 0, 0.0, strength)
    direction = np.where(direction < 0, 0, direction)
    entries = entries & (direction >= 0)

# FAIL: HEDGE_ONLY인데 hedge_threshold/hedge_strength_ratio 미적용
if config.short_mode == ShortMode.HEDGE_ONLY:
    # hedge_threshold 체크 없이 그냥 숏 허용
    pass

# PASS: HEDGE_ONLY 헷지 로직
if config.short_mode == ShortMode.HEDGE_ONLY:
    is_hedge = some_indicator < config.hedge_threshold
    strength = np.where(
        (direction < 0) & is_hedge,
        direction * vol_scalar * config.hedge_strength_ratio,
        np.where(direction < 0, 0.0, strength)
    )
```

#### entries + exits 동시 발생

```python
# WARNING: 같은 bar에서 entry와 exit가 모두 True
# VectorBT는 이 경우 exit 우선 처리하지만, 의도된 동작인지 확인 필요
conflict = entries & exits
if conflict.any():
    # 의도된 것인지 명시적 문서화 필요
    pass
```

#### direction 범위

```python
# FAIL: direction이 {-1, 0, 1} 범위 밖
direction = df["momentum"] / df["momentum"].abs().max()  # 연속 값
# VectorBT는 {-1, 0, 1} 정수만 기대

# PASS: 명시적 이산화
direction = np.sign(df["momentum"]).astype(int)
```

#### NaN strength + entries

```python
# FAIL: strength가 NaN인데 entries=True
# -> VectorBT에서 position size=NaN -> 예측 불가 동작
entries = direction != direction.shift(1)
strength = direction * vol_scalar  # vol_scalar 초기 NaN

# PASS: NaN 구간 entry 차단
valid = strength.notna() & (strength != 0)
entries = entries & valid
```

### 이 프로젝트 검증 방법

1. `signal.py`에서 `entries`, `exits`, `direction`, `strength` 생성 로직 추적
2. 동시 entries+exits 발생 가능성 확인
3. ShortMode 3종(DISABLED, HEDGE_ONLY, FULL) 각각의 분기 존재 확인
4. direction 값이 `{-1, 0, 1}` 정수인지 확인
5. strength NaN 구간에서 entries=False인지 확인
6. `StrategySignals` 반환 전 최종 정합성 확인
