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
