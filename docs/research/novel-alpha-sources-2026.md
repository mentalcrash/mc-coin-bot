# Novel Alpha Sources for 4H Crypto Futures Trading (2024-2026 Research)

**Date:** 2026-02-14
**Status:** Research Phase
**Target Timeframe:** 4H
**Target Assets:** BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, DOGE/USDT

---

## Executive Summary

이 문서는 2024-2026년 최신 학술 연구를 기반으로 4H 타임프레임 크립토 선물 거래에 적용 가능한 **참신한 알파 소스**를 정리합니다. 기존 시스템에서 실패한 접근법(순수 평균회귀, 단일지표 모멘텀, 세션/계절성 패턴)을 제외하고, **실증적 근거가 있는 새로운 신호 메커니즘**에 집중합니다.

---

## 1. Funding Rate Dynamics as Directional Signal

### 1.1 Core Research

**Paper:** "Predictability of Funding Rates" (Emre Inan, SSRN October 2025)
**Paper:** "Arbitrage in Perpetual Contracts" (Dai, Li, Yang, SSRN September 2025)

### 1.2 Signal Mechanism

- **기존 접근:** Funding rate arbitrage (롱 spot + 숏 futures, carry 수취)
- **신규 접근:** Funding rate의 **변화율**과 **극값**을 방향성 신호로 활용
  - **High positive funding (>0.03% / 8h):** 과도한 롱 포지션 → 청산 리스크 → **숏 신호**
  - **High negative funding (<-0.01% / 8h):** 과도한 숏 포지션 → 숏 커버 랠리 → **롱 신호**
  - **Funding rate 변화율:** 급격한 전환(음→양, 양→음)은 sentiment reversal 지표

### 1.3 Economic Rationale

- Perpetual futures의 funding rate는 **시장 포지셔닝의 실시간 스냅샷**
- 극단적 funding은 **과도한 레버리지 축적** → 청산 cascade 가능성 상승
- SSRN 2025 연구: Funding rate는 **예측 가능**(ML 모델로 4시간 후 funding rate 예측 시 Sharpe 2.3)

### 1.4 Implementation for 4H TF

```python
# Pseudocode
funding_rate = fetch_funding_rate(symbol)  # Binance 8h funding
funding_ma_24h = rolling_mean(funding_rate, window=3)  # 24h MA
funding_std = rolling_std(funding_rate, window=7)  # 7-period std

# Z-score normalization
funding_zscore = (funding_rate - funding_ma_24h) / funding_std

# Signal generation
if funding_zscore > 2.0:  # Extremely bullish positioning
    signal = -1  # Short (expect mean reversion)
elif funding_zscore < -1.5:  # Extremely bearish positioning
    signal = 1  # Long (expect short squeeze)
else:
    signal = 0
```

**Data Source:** Binance Futures API (`/fapi/v1/fundingRate`)
**Latency:** 8시간마다 업데이트 → 4H bar에서 2-bar 지연 가능

### 1.5 Empirical Evidence (2024-2025)

- Average funding rates stabilized at 0.015% per 8-hour period in 2025 (50% increase from 2024)
- Funding rate arbitrage yielded average annual returns of 19.26% in 2025 (up from 14.39% in 2024)
- Machine learning approach predicting funding 4 hours ahead generated 31% annual returns with Sharpe 2.3

---

## 2. Multi-Lookback Momentum Ensemble

### 2.1 Core Research

**Paper:** "Cryptocurrency Volume-Weighted Time Series Momentum" (Huang, Sangiorgi, Urquhart, SSRN December 2024)
**Paper:** "Dynamic time series momentum of cryptocurrencies" (ScienceDirect, 2021)

### 2.2 Signal Mechanism

- **문제:** 단일 lookback momentum(예: 20일)은 regime 변화에 취약
- **해결:** 여러 lookback 기간의 momentum signal을 **ensemble**로 결합
  - Short-term: 5일 (1주)
  - Medium-term: 20일 (1개월)
  - Long-term: 60일 (3개월)

### 2.3 Economic Rationale

- 크립토는 **multi-horizon momentum** 특성 (Liu & Tsyvinski 2021: 1-4주 horizon)
- Volume-weighted return이 price-only return보다 **유의미하게 우수** (Sharpe +0.3)
- Ensemble 접근은 **단일 lookback의 과적합 위험 감소**

### 2.4 Implementation for 4H TF

```python
# Multi-lookback momentum ensemble
lookbacks = [30, 120, 360]  # 5d, 20d, 60d in 4H bars (6 bars/day)

weights = []
for lb in lookbacks:
    ret = log_return(close, shift=lb)
    vol = rolling_std(ret, window=lb)
    sharpe_proxy = ret / vol
    weights.append(sharpe_proxy)

# Equal-weight ensemble
ensemble_signal = sum(weights) / len(weights)

# Z-score normalization
signal_norm = (ensemble_signal - rolling_mean(ensemble_signal, 180)) / rolling_std(ensemble_signal, 180)

if signal_norm > 0.5:
    signal = 1
elif signal_norm < -0.5:
    signal = -1
else:
    signal = 0
```

**Note:** Volume-weighted return 사용 시 `VWAP` 기반 return 계산 필요

### 2.5 Recent Evidence (2024-2025)

- Study finds strong evidence of cryptocurrency TSMOM using volume-weighted market return
- Bitcoin's time-series momentum has weakened in recent years (2024), with early-year predictability declining
- Momentum factor explains average returns in both time series and cross-sectional analyses

---

## 3. Realized Volatility Decomposition (Jump vs Continuous)

### 3.1 Core Research

**Paper:** "Price dynamics and volatility jumps in bitcoin options" (Springer Financial Innovation, 2024)
**Paper:** "Does Bitcoin futures trading reduce the normal and jump volatility in the spot market?" (ScienceDirect, 2022)

### 3.2 Signal Mechanism

- **Realized Variance (RV)** 분해:
  - **Continuous component (CV):** 정상적 확산 변동성
  - **Jump component (JV):** 불연속적 급등락 (뉴스, 청산 cascade)
- **JV 비율이 높으면** → 시장 불안정 → **포지션 축소** 또는 **역추세 진입**

### 3.3 Economic Rationale

- Crypto는 **jump volatility 비중이 높음** (전통자산 대비 2-3배)
- Jump는 **정보 충격**(예: Fed 발언, 거래소 해킹) 반영 → 트렌드 지속성 낮음
- GARCH-jump 모델이 HAR 모델 대비 **out-of-sample 예측력 우수**

### 3.4 Implementation for 4H TF

```python
# Bipower Variation (BPV) for continuous component
# RV = sum(r_t^2), BPV = μ_1^2 * sum(|r_t| * |r_{t-1}|)
μ_1 = sqrt(2 / pi)

returns_4h = log_return(close)
RV = rolling_sum(returns_4h**2, window=24)  # 4일 window
BPV = μ_1**2 * rolling_sum(abs(returns_4h) * abs(returns_4h.shift(1)), window=24)

# Jump component
JV = max(RV - BPV, 0)
JV_ratio = JV / RV

# Signal: High jump ratio → reduce position
if JV_ratio > 0.3:  # 30% jump contribution
    position_scaler = 0.5  # Half position
else:
    position_scaler = 1.0
```

**Note:** 고빈도 데이터(1m) 필요 시 계산 정밀도 향상 가능

### 3.5 Research Findings

- Augmented HAR models considering time-varying jumps show improved in-sample and out-of-sample forecasting
- Jump-induced volatility offers incremental information relative to Bitcoin implied volatility index
- Signed jumps component (positive vs negative shocks) separately studies impact on future volatility

---

## 4. Cross-Asset Lead-Lag Relationships

### 4.1 Core Research

**Paper:** "Lead-Lag relationship between Bitcoin and Ethereum: Evidence from hourly and daily data" (ScienceDirect, 2019)
**Finding:** BTC→ETH 방향으로 약한 정보 전이 우위 (BTC dominance 효과)

### 4.2 Signal Mechanism

- **BTC Dominance 변화율**을 altcoin 방향 신호로 활용
  - BTC.D 상승 → Risk-off → **ETH/SOL/DOGE 숏**
  - BTC.D 하락 → Risk-on → **ETH/SOL/DOGE 롱**
- **ETH/BTC ratio**의 돌파/붕괴를 altseason 신호로 활용
  - ETH/BTC > 0.04 breakout → **ETH 롱 가중**
  - ETH/BTC < 0.035 breakdown → **ETH 회피**

### 4.3 Economic Rationale

- BTC는 **price discovery leader** (CME 연구: BTC→ETH 정보 전이)
- BTC dominance는 **risk appetite proxy** (VIX와 유사)
- 자본 회전 패턴: BTC → ETH → L1 altcoins → memecoins

### 4.4 Implementation for 4H TF

```python
# BTC Dominance momentum
btc_dominance = fetch_btc_dominance()  # CoinMarketCap API
btc_d_ma_7d = rolling_mean(btc_dominance, window=42)  # 7d in 4H
btc_d_momentum = btc_dominance - btc_d_ma_7d

# ETH/BTC ratio signal
eth_btc_ratio = eth_close / btc_close
eth_btc_ma_30d = rolling_mean(eth_btc_ratio, window=180)  # 30d

# Altcoin position adjustment
if btc_d_momentum > 1.0:  # BTC dominance rising
    altcoin_exposure = 0.5  # Reduce altcoin weight
elif eth_btc_ratio > 0.04 and eth_btc_ratio > eth_btc_ma_30d:
    altcoin_exposure = 1.5  # Increase ETH/altcoin weight
else:
    altcoin_exposure = 1.0
```

**Data Sources:**
- BTC Dominance: CoinMarketCap API
- ETH/BTC: Binance spot pair `ETHBTC`

### 4.5 Market Evidence (2025-2026)

- Bitcoin dominance expected to remain elevated through 2026 during macro uncertainty
- 66% peak could mark dominance peak for 2025/2026 cycle
- ETH/BTC break above 0.04 could confirm new bull phase
- Bitcoin and Ethereum typically lead market trends, altcoins lag in price movements

---

## 5. Behavioral Finance Signals from OHLCV

### 5.1 Core Research

**Paper:** "Psychological Anchoring Effect, the Cross Section of Cryptocurrency Returns, and Cryptocurrency Market Anomalies" (SSRN, June 2022)
**Paper:** "Exploring investor behavior in Bitcoin: a study of the disposition effect" (Digital Finance, Springer, 2023)

### 5.2 Signal Mechanism

#### 5.2.1 Anchoring Proxy
- **52-week high nearness:** `price / max(close, 252 bars)`
- **경험적 발견:** Nearness가 높을수록 **향후 수익률 양의 상관** (t-stat > 3.0)
  - 해석: 가격이 52주 고점 근처 → 투자자 심리 긍정 → 추가 상승 가능성

#### 5.2.2 Disposition Effect Proxy
- **Volume imbalance on gains vs losses:**
  - 가격이 20일 MA 위 → "이익 구간" → 매도 압력 증가 (이익 실현)
  - 가격이 20일 MA 아래 → "손실 구간" → 매도 압력 감소 (손실 회피)
- **측정:** `vol_ratio = volume[price > ma20] / volume[price < ma20]`
- **신호:** vol_ratio가 높으면 → 단기 과매도 → **롱 진입**

### 5.3 Economic Rationale

- **Anchoring bias:** 투자자가 과거 고점/저점에 과도하게 집착
- **Disposition effect:** "winning trades 조기 청산, losing trades 보유" 편향
- Crypto는 **retail 비중 높음** → 행동재무학 편향 강하게 작동

### 5.4 Implementation for 4H TF

```python
# Anchoring: 52-week high nearness
high_52w = rolling_max(close, window=2184)  # 252d * 6 bars/day
nearness = close / high_52w

# Disposition effect proxy: Volume in gain vs loss zone
ma_20d = rolling_mean(close, window=120)  # 20d in 4H
gain_zone = close > ma_20d
loss_zone = close <= ma_20d

vol_gain = rolling_sum(volume * gain_zone, window=60) / rolling_sum(gain_zone, window=60)
vol_loss = rolling_sum(volume * loss_zone, window=60) / rolling_sum(loss_zone, window=60)
vol_ratio = vol_gain / vol_loss

# Signal combination
if nearness > 0.95 and vol_ratio > 1.3:
    signal = 1  # High anchoring + disposition selling → expect continuation
elif nearness < 0.8 and vol_ratio < 0.8:
    signal = -1  # Far from high + accumulation → expect reversal
else:
    signal = 0
```

**Note:** Volume spike 필터링 필요 (outlier 제거)

### 5.5 Research Evidence

- Strong positive cross-sectional association between anchoring (nearness to N-day high) and future cryptocurrency returns
- Crypto market exhibits reverse disposition effect in bullish periods, positive disposition effect in bearish periods
- Investors exhibiting disposition effect show lower portfolio performance

---

## 6. Information-Theoretic Approaches

### 6.1 Core Research

**Paper:** "Information Theory Quantifiers in Cryptocurrency Time Series Analysis" (MDPI Entropy, April 2025)
**Paper:** "Blockchain Technology, Cryptocurrency: Entropy-Based Perspective" (MDPI Entropy, 2022)

### 6.2 Signal Mechanism

#### 6.2.1 Shannon Entropy of Returns
- **Entropy = -Σ p_i * log(p_i)** (return distribution의 불확실성)
- **Low entropy:** 트렌드 강함 (예측 가능) → **모멘텀 전략 유리**
- **High entropy:** 변동성 높음 (예측 불가) → **포지션 축소**

#### 6.2.2 Complexity-Entropy Plane
- **Statistical Complexity (C):** 시계열 구조의 복잡도
- **C-H plane 위치:**
  - Chaotic regime (high C, mid H) → 단기 예측 가능
  - Stochastic regime (low C, high H) → 장기 예측 불가

### 6.3 Economic Rationale

- Crypto는 **chaotic (단기 <2년)** vs **stochastic (장기 >2년)** 이중 특성
- Entropy가 **BTC 가격 예측에 GARCH보다 우수** (실증)
- Information content가 높은 구간 = 신호 품질 향상

### 6.4 Implementation for 4H TF

```python
import numpy as np
from scipy.stats import entropy

# Shannon entropy of 30-day return distribution
returns_30d = log_return(close).rolling(180).apply(list)  # 30d window

def compute_entropy(ret_series):
    bins = np.histogram_bin_edges(ret_series, bins=20)
    hist, _ = np.histogram(ret_series, bins=bins)
    prob = hist / hist.sum()
    return entropy(prob + 1e-10)  # Avoid log(0)

entropy_signal = returns_30d.apply(compute_entropy)
entropy_ma = rolling_mean(entropy_signal, window=60)

# Low entropy → high conviction trend
if entropy_signal < entropy_ma * 0.8:
    conviction_scaler = 1.5  # Increase position in trending regime
else:
    conviction_scaler = 0.8  # Reduce position in random regime
```

**Computational Cost:** 높음 (rolling apply) → 사전 계산 권장

### 6.5 Research Findings (2025)

- Complexity-entropy causality plane reveals daily crypto series <2 years exhibit chaotic behavior
- Series >2 years display stochastic behavior
- Entropy of Bitcoin daily returns outperforms classical GARCH model
- Short-term prediction achievable, but completely unpredictable long-term

---

## 7. Novel Strategy Proposals

### 7.1 Funding-Momentum Hybrid

**Concept:** Funding rate 극값에서 multi-lookback momentum 신호 결합

```python
# Funding rate filter
funding_extreme = abs(funding_zscore) > 1.5

# Momentum ensemble (섹션 2 참고)
momentum_signal = ensemble_momentum(close, [30, 120, 360])

# Combined signal
if funding_zscore > 2.0 and momentum_signal < 0:
    signal = -1  # Funding overheated + bearish momentum
elif funding_zscore < -1.5 and momentum_signal > 0:
    signal = 1  # Funding oversold + bullish momentum
else:
    signal = 0
```

**Expected Edge:** Funding rate의 **regime filter** 역할 + momentum의 trend capture

---

### 7.2 Jump-Filtered Breakout

**Concept:** High jump volatility 구간에서 breakout 신호 억제

```python
# Breakout signal (Donchian-like)
high_20d = rolling_max(high, window=120)
low_20d = rolling_min(low, window=120)

if close > high_20d:
    raw_signal = 1
elif close < low_20d:
    raw_signal = -1
else:
    raw_signal = 0

# Jump volatility filter (섹션 3 참고)
if JV_ratio > 0.3:  # High jump
    signal = 0  # Suppress breakout in unstable regime
else:
    signal = raw_signal
```

**Expected Edge:** Jump volatility가 높은 구간의 false breakout 회피

---

### 7.3 Cross-Asset Rotation Strategy

**Concept:** BTC dominance cycle에 따라 BTC ↔ altcoin 자본 회전

```python
# BTC dominance momentum (섹션 4 참고)
if btc_d_momentum > 1.0:
    weights = {"BTC": 0.6, "ETH": 0.2, "SOL": 0.1, "DOGE": 0.1}
elif eth_btc_ratio > 0.04:
    weights = {"BTC": 0.2, "ETH": 0.5, "SOL": 0.2, "DOGE": 0.1}
else:
    weights = {"BTC": 0.4, "ETH": 0.3, "SOL": 0.2, "DOGE": 0.1}
```

**Expected Edge:** Capital flow를 선행 포착하여 alpha 극대화

---

### 7.4 Entropy-Gated Momentum

**Concept:** Low entropy 구간에서만 momentum 전략 활성화

```python
# Entropy regime (섹션 6 참고)
low_entropy_regime = entropy_signal < entropy_ma * 0.8

# Momentum signal
momentum = ensemble_momentum(close, [30, 120, 360])

if low_entropy_regime and momentum > 0.5:
    signal = 1
elif low_entropy_regime and momentum < -0.5:
    signal = -1
else:
    signal = 0  # Stay flat in high-entropy regime
```

**Expected Edge:** Chaotic regime에서 예측력 향상, stochastic regime 회피

---

### 7.5 Anchoring-Disposition Contrarian

**Concept:** Disposition effect 극값에서 역추세 진입

```python
# Anchoring nearness (섹션 5 참고)
nearness = close / rolling_max(close, 2184)

# Disposition vol ratio
vol_ratio = vol_gain / vol_loss

# Contrarian signal
if nearness > 0.98 and vol_ratio > 1.5:
    signal = -1  # Euphoria peak → short
elif nearness < 0.75 and vol_ratio < 0.7:
    signal = 1  # Capitulation → long
else:
    signal = 0
```

**Expected Edge:** Behavioral bias의 극값에서 mean reversion 포착

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Data Infrastructure (Week 1-2)

1. **Funding Rate Ingestion**
   - Binance `/fapi/v1/fundingRate` API 연동
   - Silver layer에 8시간 간격 데이터 저장
   - 4H candle과 alignment 처리

2. **BTC Dominance Feed**
   - CoinMarketCap API 연동 (또는 manual scraping)
   - Daily → 4H interpolation

3. **High-Frequency Data for Jump Calculation**
   - 1m candle 기반 RV/BPV 계산 파이프라인
   - 4H aggregation

### 8.2 Phase 2: Signal Development (Week 3-4)

1. **단일 신호 백테스트**
   - 각 섹션의 신호를 독립적으로 VBT 백테스트
   - Sharpe, CAGR, MDD 측정
   - G0 criteria 통과 여부 확인

2. **Signal Quality Metrics**
   - IC (Information Coefficient): signal vs forward return correlation
   - Hit rate: 방향성 정확도
   - Turnover: 거래 빈도

### 8.3 Phase 3: Strategy Integration (Week 5-6)

1. **Hybrid Strategy 구현**
   - 섹션 7의 5개 전략을 `BaseStrategy` 상속 구현
   - EDA 시스템 통합 (EventBus, PM, RM)

2. **Parameter Sweep**
   - Lookback, threshold, filter 파라미터 최적화
   - Overfitting 방지: IS/OOS, WFA 검증

### 8.4 Phase 4: Validation (Week 7-8)

1. **G0-G4 Gate 통과**
   - Multi-asset backtest (5 coins)
   - Tiered validation (IS/OOS, CPCV, PBO)
   - Correlation with existing strategies 확인

2. **Shadow Trading**
   - Paper trading 2주 실행
   - Live data feed로 signal 품질 검증

---

## 9. Expected Challenges

### 9.1 Data Quality Issues

- **Funding rate gaps:** 거래소 API 장애 시 결측치 처리
- **BTC dominance lag:** 실시간 feed 부재 시 1일 지연 가능
- **1m data volume:** Jump calculation을 위한 고빈도 데이터 저장 부담

**Mitigation:**
- Forward fill + missing data alert
- Daily batch update + interpolation
- Parquet compression + incremental storage

### 9.2 Signal Decay

- **Funding rate predictability:** 2025년 연구 기준, 2026년 이후 효율성 감소 가능
- **Behavioral bias:** 기관화 진행 시 retail 편향 약화

**Mitigation:**
- Quarterly signal re-evaluation
- Adaptive parameter tuning
- Ensemble approach (여러 신호 조합으로 decay 완충)

### 9.3 Execution Latency

- Funding rate는 8시간마다 업데이트 → 4H bar에서 최대 4시간 지연
- BTC dominance는 daily → 6-12시간 지연

**Mitigation:**
- Lag-aware backtesting (실제 지연 반영)
- Conservative entry (신호 확정 후 1-bar 대기)

---

## 10. Success Metrics

### 10.1 G0 Criteria (Minimum Viable)

- **Individual Asset Sharpe ≥ 0.5** (5 coins 중 3개 이상)
- **8-Asset EW Sharpe ≥ 0.8**
- **거래 빈도:** 연 20-100건 (4H TF 적정 범위)
- **MDD < 25%**

### 10.2 G1+ Criteria (Production Ready)

- **IS/OOS Sharpe drop < 30%**
- **CPCV mean Sharpe ≥ 0.6**
- **PBO < 60%**
- **DSR (Deflated Sharpe Ratio) ≥ 0.4**

---

## 11. References

### Academic Papers (SSRN, Springer, Elsevier)

1. **Inan, E. (2025).** "Predictability of Funding Rates." SSRN Working Paper.
   - [Link](https://papers.ssrn.com/sol3/Delivery.cfm/fe1e91db-33b4-40b5-9564-38425a2495fc-MECA.pdf?abstractid=5576424&mirid=1)

2. **Dai, M., Li, L., Yang, C. (2025).** "Arbitrage in Perpetual Contracts." SSRN Working Paper.
   - [Link](https://papers.ssrn.com/sol3/Delivery.cfm/5262988.pdf?abstractid=5262988&mirid=1)

3. **Huang, Z., Sangiorgi, I., Urquhart, A. (2024).** "Cryptocurrency Volume-Weighted Time Series Momentum." SSRN Working Paper 4825389.
   - [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389)

4. **Price dynamics and volatility jumps in bitcoin options (2024).** Financial Innovation, Springer.
   - [Link](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00653-z)

5. **Lead-Lag relationship between Bitcoin and Ethereum (2019).** Research in International Business and Finance, ScienceDirect.
   - [Link](https://www.sciencedirect.com/science/article/abs/pii/S0275531919300522)

6. **Jia, Y., Simkins, B. (2022).** "Psychological Anchoring Effect, the Cross Section of Cryptocurrency Returns." SSRN Working Paper 4170936.
   - [Link](https://papers.ssrn.com/sol3/Delivery.cfm/0d3dda1f-b7e3-4e2e-bf4e-b1413edb0b73-MECA.pdf?abstractid=4170936&mirid=1)

7. **Exploring investor behavior in Bitcoin: a study of the disposition effect (2023).** Digital Finance, Springer.
   - [Link](https://link.springer.com/article/10.1007/s42521-023-00086-w)

8. **Information Theory Quantifiers in Cryptocurrency Time Series Analysis (2025).** MDPI Entropy 27(4):450.
   - [Link](https://www.mdpi.com/1099-4300/27/4/450)

### Market Data & Analysis

- [Cryptocurrency market risk-managed momentum strategies (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377)
- [High-frequency dynamics of Bitcoin futures (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2214845025001188)
- [Herding and anchoring in cryptocurrency markets (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S2214635019301534)
- [Bitcoin Volatility Index (Bitbo)](https://bitbo.io/volatility/)
- [Crypto Funding Rates Dashboard (CoinMarketCap)](https://coinmarketcap.com/charts/funding-rates/)
- [Bitcoin Dominance Chart (CoinMarketCap)](https://coinmarketcap.com/charts/bitcoin-dominance/)

### Industry Reports (2025-2026)

- [Grayscale 2026 Digital Asset Outlook](https://research.grayscale.com/reports/2026-digital-asset-outlook-dawn-of-the-institutional-era)
- [Pantera: Navigating Crypto in 2026](https://panteracapital.com/blockchain-letter/navigating-crypto-in-2026/)
- [Coinbase Institutional: Crypto market predictions for 2026](https://www.coindesk.com/markets/2025/12/28/coinbase-says-three-areas-will-dominate-the-crypto-market-in-2026)

---

## 12. Next Actions

1. **Data Infrastructure Setup** (담당: User 승인 후 Claude 구현)
   - [ ] Binance funding rate API 연동
   - [ ] BTC dominance daily fetch + interpolation
   - [ ] 1m OHLCV → RV/BPV 계산 파이프라인

2. **Strategy Prototyping** (담당: Claude)
   - [ ] Funding-Momentum Hybrid 구현 (`funding-mom`)
   - [ ] Jump-Filtered Breakout 구현 (`jump-breakout`)
   - [ ] Entropy-Gated Momentum 구현 (`entropy-mom`)

3. **Backtest & Validation** (담당: User 리뷰 + Claude 실행)
   - [ ] 각 전략 VBT 백테스트
   - [ ] G0 criteria 통과 여부 확인
   - [ ] Correlation matrix (기존 30개 전략과 비교)

4. **Documentation** (담당: Claude)
   - [ ] Strategy YAML 메타데이터 작성
   - [ ] Lesson 기록 (research findings)
   - [ ] Gate criteria 업데이트 (새 metric 추가 시)

---

## Appendix A: Excluded Approaches (Known Failures)

### A.1 Pure Mean Reversion at 4H

- **Tier 5 교훈:** OU-MeanRev는 Sharpe 0.52로 G1 탈락
- **이유:** Crypto의 강한 트렌드 특성, 4H TF에서 MR window 불일치

### A.2 Single Indicator Momentum

- **Tier 2 교훈:** MTF-MACD는 연 5건 거래로 신호 부족
- **이유:** 단일 지표는 signal quality 낮음, 앙상블 필수

### A.3 OHLCV-Based Microstructure

- **Tier 5 교훈:** Flow-Imbalance, VPIN-Flow는 Sharpe -0.12~-1.63
- **이유:** BVC 근사는 실제 order flow 부재로 noise만 포착

### A.4 Session/Seasonality Patterns

- **Tier 5 교훈:** Session-Breakout, Hour-Season 전멸 (Sharpe -1.67~-6.48)
- **이유:** Crypto는 24/7 시장, FX session edge 전이 불가

### A.5 Regime Detection as Alpha

- **Tier 3 교훈:** HMM-Regime는 lagging indicator, 독립 alpha 아님
- **이유:** Regime은 **filter**로 유용, signal 생성은 다른 메커니즘 필요

---

**End of Document**
