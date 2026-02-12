# Strategy Candidates — Temp Staging

> Gate 0 PASS 아이디어의 임시 후보 목록. 구현 전 사용자 리뷰와 우선순위 결정용.

---

## 2026-02-10 — Strategy Discovery Session (6H/12H TF)

### 후보 #1: Acceleration-Conviction Momentum (`accel-conv`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Momentum (2nd derivative) + Candle Anatomy |
| **타임프레임** | 6H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 27/30 |
| **상태** | ✅ 구현 완료 (2026-02-11) |

**핵심 가설**: 가격 가속도(2차 미분)와 캔들 body conviction이 동시에 양(+)이면 추세 지속 확률이 극대화된다.

**경제적 논거**: Acceleration은 positive feedback loop가 강화되고 있음을 의미 (Ardila et al. 2021, Physica A). Body/range ratio는 세션 내 방향적 확신을 직접 측정하며, 두 독립 시그널의 결합이 단일 지표 전략의 Decay 문제를 완화한다. Gamma factor가 momentum factor를 2/3의 파라미터 조합에서 outperform.

**사용 지표**: `acceleration = returns.diff()`, `conviction = abs(close - open) / (high - low)`

**시그널 생성 로직**:

```
acc = rolling_mean(ret.diff(), N)          # smoothed acceleration
conv = rolling_mean(abs(C-O)/(H-L), N)    # smoothed body conviction
signal = sign(acc) * conv                  # direction × strength
→ signal > threshold: LONG
→ signal < -threshold: SHORT (HEDGE_ONLY)
→ otherwise: FLAT
```

**CTREND 상관 예측**: 낮음 (CTREND feature set에 acceleration/body ratio 미포함)

**예상 거래 빈도**: 150~300건/년

**차별화 포인트**: 2차 미분(acceleration)과 body conviction은 46개 전략 중 어느 것도 사용하지 않은 완전 새 카테고리. 가장 유사한 폐기 전략 없음.

**출처**: Ardila, Forro, Sornette (Physica A, 2021) "The acceleration effect and Gamma factor in asset pricing" + Bulkowski (2008) candle pattern statistics

**Gate 0 상세 점수**:

- 경제적 논거: 4/5 (학술 실증 + 이론적 기반)
- 참신성: 5/5 (완전 새 카테고리)
- 데이터 확보: 5/5 (OHLCV only)
- 구현 복잡도: 5/5 (ret.diff() + body ratio = 극간단)
- 용량 수용: 4/5 (6H → 1460 bars/year, 충분한 빈도)
- 레짐 독립성: 4/5 (추세장 강함, conviction 필터가 횡보장 보정)

---

### 후보 #2: Anchored Momentum (`anchor-mom`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Behavioral Finance (Psychological Anchoring) |
| **타임프레임** | 12H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 25/30 |
| **상태** | ✅ 구현 완료 (2026-02-11) |

**핵심 가설**: Rolling N-period high 대비 근접도(nearness)가 높을수록 상승 지속(under-reaction), 낮을수록 하락 압력(loss aversion 매도).

**경제적 논거**: 투자자가 최근 고점을 심리적 앵커로 사용. 고점 근처에서 과도한 매도 압력 → under-reaction → 이후 추가 상승. Jia et al. 2024-2026: cANCHOR factor ~130bp/week. 크립토 retail 지배 시장에서 behavioral bias가 극대화된다.

**사용 지표**: `nearness = close / rolling_max(close, N)`, `momentum = sign(close / close.shift(M) - 1)`

**시그널 생성 로직**:

```
nearness = close / rolling_max(close, lookback)
mom_sign = sign(close / close.shift(mom_lookback) - 1)

→ nearness > 0.95 AND mom_sign > 0: STRONG LONG
→ nearness > 0.85 AND mom_sign > 0: LONG
→ nearness < 0.80 AND mom_sign < 0: SHORT (HEDGE_ONLY)
→ otherwise: FLAT or reduced position
```

**CTREND 상관 예측**: 낮음~중간 (rolling high 정보 일부 공유 가능)

**예상 거래 빈도**: 80~200건/년

**차별화 포인트**: VWAP-Disposition(폐기, Sharpe 0.96)은 disposition effect + VWAP anchor. 이 전략은 anchoring bias + rolling high-water mark. 다른 behavioral mechanism. HEDGE_ONLY로 DOGE MDD -622% 방지.

**출처**: Jia, Simkins, Yan et al. (SSRN 5386180, 2024-2026) "Psychological Anchoring Effect and Cross Section of Cryptocurrency Returns"

**Gate 0 상세 점수**:

- 경제적 논거: 5/5 (최강 학술 근거, ~130bp/week 실증)
- 참신성: 4/5 (anchoring 미시도, VWAP-Disposition과 다른 메커니즘)
- 데이터 확보: 5/5 (OHLCV only)
- 구현 복잡도: 5/5 (rolling_max + nearness ratio = 극간단)
- 용량 수용: 3/5 (12H → 730 bars/year, 느린 신호)
- 레짐 독립성: 3/5 (장기 하락장에서 약화 가능)

---

### 후보 #3: Quarter-Day TSMOM (`qd-mom`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Intraday Time-Series Momentum |
| **타임프레임** | 6H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 25/30 |
| **상태** | ✅ 구현 완료 (2026-02-11) |

**핵심 가설**: 이전 6H session return이 다음 session return을 양(+)으로 예측. Late-informed trader의 정보 흡수 지연 메커니즘.

**경제적 논거**: Shen 2022: BTC에서 Sharpe 1.15, 연 수익 13.95%. 정보가 느린 투자자들이 세션 후반에 진입하며 모멘텀을 지속시킨다. 24시간을 4개 session으로 자연 분할하면 Asia/Europe/US/Late 각 세션 간 정보 흐름 포착 가능.

**사용 지표**: `prev_ret = close / close.shift(1) - 1`, `vol_filter = volume > rolling_median(volume, N)`

**시그널 생성 로직**:

```
prev_ret = close / close.shift(1) - 1
vol_ok = volume > rolling_median(volume, lookback)

→ prev_ret > 0 AND vol_ok: LONG
→ prev_ret < 0 AND vol_ok: SHORT (HEDGE_ONLY)
→ NOT vol_ok: FLAT (low conviction)
```

**CTREND 상관 예측**: 낮음 (daily vs sub-daily 메커니즘 완전 다름)

**예상 거래 빈도**: 200~400건/년

**차별화 포인트**: Session-Breakout(폐기, 1H range breakout)과 근본적으로 다름: range breakout ≠ return direction prediction. Shen 2022가 crypto에서 직접 검증한 intraday momentum.

**출처**: Shen (2022) "Bitcoin intraday time series momentum" (Financial Review)

**Gate 0 상세 점수**:

- 경제적 논거: 4/5 (Shen 2022 BTC Sharpe 1.15)
- 참신성: 4/5 (sub-daily TSMOM 미시도, session echo 우려)
- 데이터 확보: 5/5 (OHLCV only)
- 구현 복잡도: 5/5 (prev return + volume filter = 극간단)
- 용량 수용: 4/5 (6H → 4 signals/day, 충분)
- 레짐 독립성: 3/5 (횡보장에서 autocorrelation 감소)

---

### 후보 #4: Acceleration-Skewness Signal (`accel-skew`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Momentum (2nd derivative) + Higher Moments |
| **타임프레임** | 12H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 24/30 |
| **상태** | ✅ 구현 완료 (2026-02-11) |

**핵심 가설**: 가격 가속도가 양(+)이고 rolling skewness도 양(+)이면, 우상향 테일이 reward로 전환. Skewness가 음(-)이면 crash risk → 거래 중단.

**경제적 논거**: Acceleration은 positive feedback 강화 (Ardila et al.). 양의 skewness = 상승 잠재력 > 하락 리스크 (QuantPedia 2024: skewness lottery Sharpe 1.25). Return distribution의 형태 자체가 regime 정보를 담고 있어, skewness는 momentum의 quality filter로 작용.

**사용 지표**: `acceleration = returns.diff()`, `rolling_skew = returns.rolling(N).skew()`

**시그널 생성 로직**:

```
acc = rolling_mean(ret.diff(), N)
skew = returns.rolling(skew_window).skew()

→ acc > 0 AND skew > skew_threshold: LONG
→ acc < 0 AND skew < -skew_threshold: SHORT (HEDGE_ONLY)
→ skew 중립: position 유지 (no action)
→ skew 반대 부호: FLAT (crash risk 회피)
```

**CTREND 상관 예측**: 낮음

**예상 거래 빈도**: 60~150건/년

**차별화 포인트**: Acceleration + skewness 조합은 완전 미시도. Entropy-Switch(폐기)는 entropy=filter만, alpha 부재. 여기서는 acceleration이 primary alpha, skewness는 quality filter.

**출처**: Ardila et al. (2021) + QuantPedia "Skewness/Lottery Trading Strategy in Cryptocurrencies" (2024)

**Gate 0 상세 점수**:

- 경제적 논거: 4/5 (두 시그널 모두 학술 근거)
- 참신성: 5/5 (acceleration + skewness 조합 미사용)
- 데이터 확보: 5/5 (OHLCV only)
- 구현 복잡도: 4/5 (skewness rolling 계산 약간 복잡)
- 용량 수용: 3/5 (12H + skewness filter → 거래 감소)
- 레짐 독립성: 3/5 (강한 횡보에서 약화)

---

## 2026-02-12 — Strategy Discovery Session (1H/30m TF, Event-Driven Intraday)

> **테마**: 레이턴시 비민감 + 선택적 진입 (개인 투자자 최적화)
> **핵심 교훈**: 1m~1h에서 비용 drag이 핵심 제약 → 연 30~80건 이벤트 기반 전략만 생존 가능

### 후보 #5: Abnormal Day Momentum (`abnorm-mom`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Event-Driven Momentum |
| **타임프레임** | 1H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 26/30 |
| **상태** | 🔵 후보 |

**핵심 가설**: 비정상 수익률일(abnormal day)을 조기 감지하면 당일~익일 momentum continuation을 포착할 수 있다.

**경제적 논거**: 대규모 price move는 information arrival을 반영하며, 크립토 24/7 시장에서 정보 소화에 시간이 걸려 일중/익일 continuation이 발생한다. Caporale & Plastun (2020)이 BTC/ETH/LTC에서 직접 검증: abnormal day의 hourly return이 일반일 대비 유의하게 크고, dynamic trigger로 당일 중 조기 감지 가능.

**사용 지표**: `rolling_std(daily_returns, 20d)`, `cum_intraday_ret = close / day_open - 1`

**시그널 생성 로직**:

```
daily_ret_std = std(daily_returns, 20)     # 20일 rolling
threshold = 1.5 * daily_ret_std            # dynamic

cum_ret = (close / day_open) - 1           # 매 1H bar 계산

if hours_elapsed >= 8:
    if cum_ret > threshold:   → LONG
    if cum_ret < -threshold:  → SHORT (HEDGE_ONLY)

Exit: 익일 종료 또는 trailing ATR stop
```

**CTREND 상관 예측**: 낮음 (event-driven intraday ≠ ML ensemble daily)

**예상 거래 빈도**: 30~60건/년

**차별화 포인트**: "Abnormal day detection → intraday momentum" 접근은 54개 전략 중 최초. QD-Mom(이전 bar return 방향)과 근본적으로 다름 — ADM은 누적 intraday return이 동적 임계값을 초과하는지 감지. 매일 거래하지 않고 비정상일에만 진입하므로 noise 과적합 위험 극저.

**출처**: Caporale & Plastun (2020) "Momentum effects in the cryptocurrency market after one-day abnormal returns" (Financial Markets and Portfolio Management)

**Gate 0 상세 점수**:

- 경제적 논거: 4/5 (Caporale & Plastun, crypto 직접 검증, BTC/ETH/LTC)
- 참신성: 5/5 (abnormal day detection 완전 미시도)
- 데이터 확보: 5/5 (OHLCV only, rolling std)
- 구현 복잡도: 5/5 (rolling std + threshold + cum return)
- 용량 수용: 3/5 (30-60건/년, 비용 효율적이나 희소)
- 레짐 독립성: 4/5 (abnormal events는 모든 레짐에서 발생)

---

### 후보 #6: Volume Shock Dual-Mode (`vol-shock`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Event-Driven (Volume Microstructure) |
| **타임프레임** | 1H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 25/30 |
| **상태** | 🔵 후보 |

**핵심 가설**: 비정상 거래량 급증 시 bar return의 부호에 따라 continuation(informed buying) vs reversal(panic liquidation)을 구분하여 매매한다.

**경제적 논거**: Volume spike + positive return = informed buying → continuation (Kyle 1985, Continuous Auctions). Volume spike + negative return = panic liquidation → overreaction → bounce (crypto liquidation cascades). 이 두 메커니즘은 경제적으로 서로 다르며, 방향에 따른 차별적 대응이 단일 모드(reversal only) 전략보다 우월하다.

**사용 지표**: `vol_ratio = volume / rolling_median(volume, 48)`, `bar_ret = (close - open) / open`

**시그널 생성 로직**:

```
vol_ratio = volume / rolling_median(volume, 48)
bar_ret = (close - open) / open
ret_threshold = rolling_std(returns, 48) * 1.0

if vol_ratio > 3.0:
    if bar_ret > ret_threshold:      → LONG (informed continuation)
    if bar_ret < -ret_threshold:     → LONG (panic reversal, 다음 bar)
    # HEDGE_ONLY SHORT: symmetric logic for negative shocks

Exit: 4-8h trailing ATR stop
```

**CTREND 상관 예측**: 낮음 (volume-event ≠ ML feature ensemble)

**예상 거래 빈도**: 40~80건/년

**차별화 포인트**: Vol-Climax(4H, 재검증 대기)는 reversal only. VSDM은 bar return 부호에 따라 continuation vs reversal을 dual-mode로 구분하는 최초 전략. 또한 1H TF에서의 적용은 미시도.

**출처**: Kyle (1985) "Continuous Auctions and Insider Trading" + crypto liquidation cascade 연구 (2024-2025 다수)

**Gate 0 상세 점수**:

- 경제적 논거: 4/5 (informed trading + liquidation cascades, Kyle 1985)
- 참신성: 4/5 (dual-mode, Vol-Climax는 reversal only)
- 데이터 확보: 5/5 (OHLCV volume)
- 구현 복잡도: 5/5 (volume ratio + return sign + threshold)
- 용량 수용: 3/5 (40-80건/년)
- 레짐 독립성: 4/5 (volume shocks는 모든 레짐에서 발생)

---

### 후보 #7: Intraday Overextension Reversal (`intraday-or`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Intraday Mean Reversion (Range-Normalized) |
| **타임프레임** | 30m |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 25/30 |
| **상태** | 🔵 후보 |

**핵심 가설**: 일중 누적 수익률이 "정상 일일 범위"(rolling ATR)의 80%를 초과하면, 과잉반응으로 평균회귀가 발생한다.

**경제적 논거**: Intraday overextension은 noise trader overreaction + leveraged liquidation cascade의 결과. Wen et al. (2022)이 크립토에서 intraday reversal 패턴을 확인. 일일 ATR 대비 비율 정규화로 모든 변동성 레짐에서 adaptive하게 작동. BB-RSI(가격 레벨 기반 밴드)와 근본적으로 다른 접근: 수익률 vs 범위 비율 기반.

**사용 지표**: `cum_intraday_ret = close / day_open - 1`, `daily_range = rolling_mean(high - low, 20)`

**시그널 생성 로직**:

```
daily_range = rolling_mean(daily_high - daily_low, 20)  # 20d ATR
cum_ret = (close / day_open) - 1
overext = abs(cum_ret * day_open) / daily_range

if overext > 0.80:
    if cum_ret > 0:  → SHORT (HEDGE_ONLY, overextended up)
    if cum_ret < 0:  → LONG  (overextended down, reversal)

Exit: day_open 복귀 (VWAP proxy) 또는 max 6h hold
```

**CTREND 상관 예측**: 매우 낮음 (counter-trend MR ≠ trend-following ML)

**예상 거래 빈도**: 40~80건/년

**차별화 포인트**: "cum_intraday_ret / rolling_daily_range" 비율은 54개 전략 중 미사용. BB-RSI(가격 레벨 대비 밴드)와 근본적으로 다름 — 이 전략은 수익률 크기를 일일 범위로 정규화하여 "오늘 얼마나 많이 움직였는가?"를 측정. ATR 정규화로 고/저변동성 레짐 자동 적응.

**출처**: Wen, Bouri, Xu, Zhao (2022) "Intraday return predictability in the cryptocurrency markets" (North American Journal of Economics and Finance)

**Gate 0 상세 점수**:

- 경제적 논거: 4/5 (intraday overreaction, Wen et al. 크립토 실증)
- 참신성: 5/5 (cum_ret / daily_range 비율 완전 미시도)
- 데이터 확보: 5/5 (OHLCV only)
- 구현 복잡도: 4/5 (daily range tracking + intraday cum ret)
- 용량 수용: 3/5 (40-80건/년)
- 레짐 독립성: 4/5 (ATR 정규화로 자동 적응)
