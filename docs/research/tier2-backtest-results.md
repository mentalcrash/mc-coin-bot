# Tier 2 Strategy Backtest Results & Failure Analysis

**Period**: 2022-01-01 ~ 2025-12-31 (Bear → Recovery → Bull full cycle)
**Coins**: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, DOGE/USDT
**Capital**: $100,000 per strategy
**Fee**: 0.1% round-trip (VectorBT default)

---

## 1. Executive Summary

6개 Tier 2 전략 중 **유의미한 성과(Sharpe > 0.5)를 보인 전략은 2개**:

| Rank | Strategy | EW Sharpe | EW CAGR | EW MDD | Verdict |
|------|----------|-----------|---------|--------|---------|
| 1 | **Donchian Ensemble** | **1.00** | **+6.9%** | **-5.9%** | **PASS** |
| 2 | **TTM Squeeze** | 0.30 | +3.0% | -16.5% | Borderline |
| 3 | Stoch-Mom | 0.20 | +1.7% | -17.8% | Fail |
| 4 | MTF-MACD | 0.17 | +0.7% | -12.5% | Fail |
| 5 | Larry-VB | -0.36 | -9.4% | -45.2% | **Fail (destructive)** |
| 6 | Mom-MR Blend | -0.34 | -4.0% | -19.2% | Fail |

**Buy & Hold 비교**: BTC +85.3%, ETH -21.1%, BNB +63.5%, SOL -30.3%, DOGE -28.9%
→ Donchian Ensemble은 Buy&Hold 대비 MDD를 1/11로 줄이면서 안정적 수익 달성.

---

## 2. Strategy-by-Strategy Analysis

### 2.1 Donchian Ensemble (Winner)

**Concept**: 9개 lookback(5~360일) Donchian 채널 신호의 평균 앙상블

#### Full Period Performance (5-coin avg)

| Metric | Value |
|--------|-------|
| Sharpe | 0.69 |
| CAGR | +6.5% |
| MDD | -10.5% |
| Win Rate | 45.9% |
| Avg Trades | 104 |

#### EW Portfolio

| Metric | Value |
|--------|-------|
| Sharpe | **1.00** |
| CAGR | **+6.9%** |
| MDD | **-5.9%** |
| Trades | 204 |
| Profit Factor | 2.14 |

#### Regime Analysis (5-coin avg Sharpe)

| Bear (2022) | Recovery (2023) | Bull (2024-25) |
|-------------|-----------------|----------------|
| -0.43 | 0.03 | **0.90** |

#### Strengths

- **앙상블 효과**: 9개 lookback 평균으로 개별 채널 노이즈 상쇄, 안정적 신호
- **Multi-asset 분산**: EW 포트폴리오에서 Sharpe 1.00 달성 (개별 0.69 → 포트폴리오 1.00)
- **MDD 방어**: Buy&Hold 대비 MDD 67~94% 감소
- **Coin-neutral**: SOL Sharpe 0.87, BTC 0.75, ETH 0.73 — 코인별 편차 낮음 (σ=0.14)

#### Weaknesses

- **Bear 시장 약세**: 2022년 평균 Sharpe -0.43 (SOL -2.21 극단적 손실)
- **Signal lag**: 360일 lookback은 극단적 지연, but 앙상블이 단기(5일) 신호로 보완
- **Recovery 미포착**: 2023년 Sharpe 0.03 — 바닥 탈출 시그널 지연

#### Verdict: **PASS — Tier 3 (EDA 통합) 후보**

---

### 2.2 Larry Williams VB (Worst Performer)

**Concept**: `close > open + k * prev_range` 돌파 시 1일 보유

#### Full Period Performance (5-coin avg)

| Metric | Value |
|--------|-------|
| Sharpe | **-0.39** |
| CAGR | **-13.2%** |
| MDD | **-53.9%** |
| Win Rate | 45.7% |
| Avg Trades | **498** |

#### EW Portfolio

| Sharpe | CAGR | MDD | Trades |
|--------|------|-----|--------|
| -0.36 | -9.4% | -45.2% | **2,357** |

#### Failure Diagnosis

1. **Transaction Cost Drag (핵심 원인)**: 연간 ~125건 거래 × 0.1% fee = 연 12.5% 비용.
   k=0.5 breakout의 미세 수익을 비용이 잠식.
2. **Whipsaw**: 1-bar hold 패턴은 매일 진입/청산 판단 → 횡보장에서 연속 손실 누적
3. **Coin Specificity**: BNB에서 CAGR -28% (최악), 거래비용 대비 변동폭이 작은 코인에서 치명적
4. **Regime**: Bear(2022) Sharpe +0.20, Bull(2024-25) Sharpe **-0.87** — 상승장에서 오히려 손실 (daily reversal이 추세를 역행)

#### Root Cause

일봉 VB는 암호화폐의 높은 수수료 구조에서 비용 초과. 원래 주식시장의 일일 변동성(~1%)에 최적화된 전략이
암호화폐 변동성(~3-5%)에서는 k factor 조정이 필요하나, 근본적으로 **고빈도 1-bar hold가 비용 부담**.

#### Verdict: **FAIL — 폐기 (비용 구조적 문제)**

---

### 2.3 MTF MACD

**Concept**: MACD(12,26,9) crossover + trend filter, 음봉 청산

#### Full Period Performance (5-coin avg)

| Metric | Value |
|--------|-------|
| Sharpe | 0.12 |
| CAGR | +0.7% |
| MDD | -14.4% |
| Win Rate | 48.0% |
| Avg Trades | **20** |

#### EW Portfolio

| Sharpe | CAGR | MDD | Trades |
|--------|------|-----|--------|
| 0.17 | +0.7% | -12.5% | 92 |

#### Regime Analysis (5-coin avg Sharpe)

| Bear (2022) | Recovery (2023) | Bull (2024-25) |
|-------------|-----------------|----------------|
| **0.50** | **0.62** | -0.06 |

#### Failure Diagnosis

1. **Signal Density 부족**: 연간 평균 5건 거래 — 시그널이 너무 드물어 기회비용 높음
2. **Bull 시장 실패**: 2024-25 Sharpe -0.06. MACD는 지연 지표로 상승 추세의 초반 진입 실패
3. **Exit Rule (음봉 청산)**: 단일 음봉으로 청산하면 건전한 조정에서도 조기 이탈
4. **Coin Specificity**: BNB Sharpe -0.61 (MACD가 BNB의 낮은 변동성에서 신호 미생성)

#### Paradox: Bear에서 성과 ↑, Bull에서 ↓

MACD trend filter가 Bear(2022)에서 대부분 flat → 손실 회피. 하지만 이는 "잘한 것"이 아니라 "아무것도 안 한 것". Bear에서 거래 0-2건.

#### Verdict: **FAIL — Exit rule 개선 필요 (음봉 → ATR trailing stop)**

---

### 2.4 Stochastic Momentum Hybrid

**Concept**: Stochastic %K/%D crossover + SMA(30) trend filter + ATR sizing

#### Full Period Performance (5-coin avg)

| Metric | Value |
|--------|-------|
| Sharpe | **0.50** |
| CAGR | +3.2% |
| MDD | -8.2% |
| Win Rate | 46.1% |
| Avg Trades | 140 |

#### EW Portfolio

| Sharpe | CAGR | MDD | Trades |
|--------|------|-----|--------|
| 0.20 | +1.7% | -17.8% | 57 |

#### Regime Analysis

| Bear (2022) | Recovery (2023) | Bull (2024-25) |
|-------------|-----------------|----------------|
| -0.92 | 0.59 | **0.65** |

#### Analysis

- **개별 코인 성과는 양호** (avg Sharpe 0.50, 모든 코인 양수 수익)
- **MDD 방어 우수**: 평균 MDD -8.2% (Donchian Ensemble 다음으로 낮음)
- **EW 포트폴리오 약화**: 개별 0.50 → 포트폴리오 0.20으로 하락 (상관관계 높음)
- **Bear 시장 취약**: -0.92 — Stochastic은 횡보/하락장에서 과잉 신호 생성

#### Why EW Portfolio Underperforms Individual

Stochastic crossover는 코인 간 동기화 경향 → 모든 코인이 동시에 long/flat → 분산 효과 미미.
Donchian Ensemble은 lookback 다양성으로 코인 간 시점 분산이 자연스럽게 발생.

#### Verdict: **FAIL (borderline) — 개별 코인에서는 유효하나 포트폴리오 분산 효과 부재**

---

### 2.5 Momentum + Mean Reversion Blend

**Concept**: Mom Z-Score(28d) + MR Z-Score(14d) 50/50 블렌딩

#### Full Period Performance (5-coin avg)

| Metric | Value |
|--------|-------|
| Sharpe | **-0.20** |
| CAGR | -3.5% |
| MDD | -28.0% |
| Win Rate | 47.4% |
| Avg Trades | 56 |

#### Regime Analysis

| Bear (2022) | Recovery (2023) | Bull (2024-25) |
|-------------|-----------------|----------------|
| **-0.82** | 0.06 | -0.30 |

#### Failure Diagnosis

1. **Alpha Cancellation**: Momentum(+)과 MR(-)이 서로 상쇄.
   이론적으로 "직교 alpha"이지만, 실제로는 **동일 시간축에서 상반된 신호가 동시에 발생** → 약한 combined signal
2. **BNB 치명적**: Sharpe -0.56, CAGR -9.2% — 낮은 변동성 코인에서 z-score 신호가 노이즈에 매몰
3. **Bear 시장 최악**: -0.82 — Mom은 하락 추세 따라가고, MR은 반등 기대 → 둘 다 실패
4. **Holding Period 이슈**: z-score 신호 전환이 느려 평균 보유 기간 ~20일, 중간 변동에 노출

#### Root Cause

Mom과 MR의 "직교성"은 **다른 시간축**(예: Mom 월봉 + MR 시간봉)에서만 유효.
동일 일봉에서 50/50 블렌딩은 신호 강도만 약화시킴.

#### Verdict: **FAIL — 컨셉 자체의 구조적 문제 (동일 TF 블렌딩 무효)**

---

### 2.6 TTM Squeeze

**Concept**: BB가 KC 안으로 수축 후 해제 시 momentum 방향 진입

#### Full Period Performance (5-coin avg)

| Metric | Value |
|--------|-------|
| Sharpe | 0.07 |
| CAGR | -0.0% |
| MDD | -30.2% |
| Win Rate | 49.6% |
| Avg Trades | 24 |

#### EW Portfolio

| Sharpe | CAGR | MDD | Trades |
|--------|------|-----|--------|
| **0.30** | +3.0% | -16.5% | 87 |

#### Regime Analysis

| Bear (2022) | Recovery (2023) | Bull (2024-25) |
|-------------|-----------------|----------------|
| -0.60 | -0.08 | **0.27** |

#### Analysis

- **EW 포트폴리오 > 개별 평균**: 개별 0.07 → 포트폴리오 0.30 (분산 효과 존재)
- **BTC에서만 양호**: BTC Sharpe 0.57, 나머지 코인은 -0.3 ~ +0.07
- **신호 희소성**: 연간 6건 — squeeze fire 이벤트 자체가 드묾
- **MDD 과다**: 개별 코인 평균 -30.2% → stop-loss 없이 큰 손실 방치

#### Why Portfolio Effect

Squeeze fire는 코인별 비동기적 (BTC squeeze ≠ ETH squeeze) → 자연 분산.
하지만 개별 성과가 약해 포트폴리오로도 Sharpe 0.30 수준에 머묾.

#### Verdict: **Borderline — BB/KC 파라미터 최적화 + SL 추가 시 가능성 있음**

---

## 3. Cross-Strategy Comparison

### 3.1 Risk-Adjusted Metrics (EW Portfolio)

| Strategy | Sharpe | Sortino | Calmar | Profit Factor |
|----------|--------|---------|--------|---------------|
| **Donchian Ensemble** | **1.00** | **1.61** | **1.17** | **2.14** |
| TTM Squeeze | 0.30 | 0.23 | 0.18 | 1.39 |
| Stoch-Mom | 0.20 | 0.25 | 0.09 | 2.83 |
| MTF-MACD | 0.17 | 0.09 | 0.05 | 1.16 |
| Mom-MR Blend | -0.34 | -0.29 | 0.21 | 0.85 |
| Larry-VB | -0.36 | -0.49 | 0.21 | 0.95 |

### 3.2 Regime Sensitivity (avg Sharpe)

| Strategy | Bear | Recovery | Bull | Spread |
|----------|------|----------|------|--------|
| Donchian Ensemble | -0.43 | 0.03 | **0.90** | 1.33 |
| MTF-MACD | **0.50** | **0.62** | -0.06 | 0.68 |
| Stoch-Mom | -0.92 | 0.59 | 0.65 | 1.57 |
| TTM Squeeze | -0.60 | -0.08 | 0.27 | 0.87 |
| Larry-VB | 0.20 | -0.10 | **-0.87** | 1.07 |
| Mom-MR Blend | -0.82 | 0.06 | -0.30 | 0.88 |

**핵심 발견**:

- 모든 전략이 Bear(2022)에서 고전 — Bear hedge 전략 부재
- MTF-MACD만 Bear에서 양수 (거래 0-2건으로 시장 회피)
- Bull에서 Donchian Ensemble만 강한 양수 (0.90)

### 3.3 Coin Specificity (Sharpe 표준편차)

| Strategy | σ(Sharpe) | Most Dependent |
|----------|-----------|----------------|
| Donchian Ensemble | **0.14** | Lowest — coin-neutral |
| MTF-MACD | 0.43 | BNB에서 약함 |
| Stoch-Mom | 0.26 | BNB에서 약함 |
| TTM Squeeze | 0.35 | BTC 편향 |
| Larry-VB | 0.36 | BNB에서 치명적 |
| Mom-MR Blend | 0.30 | BNB에서 치명적 |

---

## 4. Failure Pattern Taxonomy

### Pattern 1: Transaction Cost Death (Larry-VB)

- **증상**: 높은 거래 빈도 + 낮은 win rate + 비용 초과
- **원인**: 1-bar hold 패턴의 구조적 비용 부담
- **교훈**: 암호화폐에서 일봉 단타는 비용 우위 없이 불가

### Pattern 2: Alpha Cancellation (Mom-MR Blend)

- **증상**: 약한 combined signal + 모든 레짐에서 부진
- **원인**: 동일 타임프레임에서 상반된 alpha source 블렌딩
- **교훈**: 멀티 alpha는 다른 타임프레임 또는 다른 자산에서 결합해야 유효

### Pattern 3: Signal Starvation (MTF-MACD, TTM Squeeze)

- **증상**: 연간 거래 5-6건, 기회비용 과다
- **원인**: 엄격한 entry 조건 (MACD cross + trend, squeeze fire)
- **교훈**: 신호가 너무 드물면 통계적 유의성 확보 불가

### Pattern 4: Whipsaw in Sideways (Stoch-Mom)

- **증상**: Bear에서 -0.92, 과잉 신호
- **원인**: Stochastic 오실레이터가 횡보장에서 빈번한 crossover 생성
- **교훈**: 오실레이터 기반 전략은 추세 필터가 핵심 (SMA만으로 부족)

---

## 5. Conclusions & Recommendations

### 5.1 Tier 3 (EDA 통합) 후보

1. **Donchian Ensemble** — EW Sharpe 1.00, MDD -5.9%. 앙상블 효과로 안정적. 즉시 EDA 통합 권장.

### 5.2 파라미터 최적화 후 재평가

2. **TTM Squeeze** — Exit SMA → ATR trailing stop, BB/KC 파라미터 sweep 후 재시도
2. **Stoch-Mom** — 개별 코인 성과(0.50)는 양호. 추세 필터 강화(ADX 추가) + 코인 선별

### 5.3 폐기

4. **Larry-VB** — 비용 구조적 문제. 1-bar hold는 암호화폐에서 비가역적 실패.
2. **Mom-MR Blend** — 동일 TF 블렌딩은 이론적으로 무효. 컨셉 자체 폐기.
3. **MTF-MACD** — 신호 빈도 부족. 실용성 없음.

### 5.4 Tier 1 vs Tier 2 교훈

- **Tier 1**: 5개 전략 중 0개 유의미 → 단순 지표 조합의 한계
- **Tier 2**: 6개 중 1개 유의미 (Donchian Ensemble) → **앙상블 + 분산이 핵심**
- **결론**: 단일 지표 < 앙상블, 단일 코인 < 멀티에셋. 복잡도는 불필요하고 **다양성**이 알파.

---

## Appendix: Raw Data Files

- `results/tier2_single_coin_metrics.csv` — 30개 개별 코인 결과 + 5개 B&H
- `results/tier2_multi_asset_metrics.csv` — 6개 EW 포트폴리오 결과
- `results/tier2_regime_analysis.csv` — 90개 레짐별 결과
