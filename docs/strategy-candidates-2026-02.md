# 전략 후보 리스트 (2026-02-10)

> **배경**: 24개 전략 전원 폐기 후, 학술 논문/SSRN/arXiv/실무 리서치를 기반으로 새로운 전략 후보를 탐색.
> 기존 실패 교훈을 반영하여 **구조적 엣지, 멀티팩터 앙상블, OOS 로버스트성**에 초점.

---

## 기존 24개 전략 실패 교훈 (반면교사)

| # | 교훈 | 해당 실패 전략 |
|---|------|----------------|
| 1 | 단일 지표 전략은 IS에서만 작동, OOS에서 붕괴 | Vol-Adaptive (156%), Adaptive Breakout (201%), ADX (146%) |
| 2 | CAGR < 20%이면 안정적이어도 운용 효율 부족 | Donchian Ensemble (10.8%), BB-RSI (4.6%) |
| 3 | 앙상블 > 단일 지표, 멀티에셋 > 단일 코인 | Larry-VB, RSI Crossover, Z-Score MR |
| 4 | 동일 TF에서 Mom + MR 블렌딩은 alpha 상쇄 | Mom-MR Blend (109% decay) |
| 5 | Regime detection (HMM, Hurst)은 crypto에서 불안정 | HMM Regime (163%), Hurst/ER (noisy) |
| 6 | 파라미터 뾰족한 봉우리 = 오버피팅 | TTM Squeeze (고원 부재) |
| 7 | Time-series momentum은 단일 코인에서 OOS decay 극심 | TSMOM (87%), Enhanced TSMOM (85%) |

**새 전략의 필수 조건:**
- 구조적/행동적 엣지 (data-mining이 아닌 경제적 논거)
- IS/OOS decay < 50% 가능성
- 파라미터 2~3개 이하 또는 regularized ensemble
- CAGR > 20% 구조적 달성 가능 설계

---

## Gate 0 평가 기준

| 항목 | 1점 | 3점 | 5점 |
|------|-----|-----|-----|
| 경제적 논거 | 설명 불가 | 논문 있으나 crypto 미검증 | 행동편향/구조적 제약으로 설명 |
| 참신성 | SSRN 3년+ 공개 | 최근 1~2년 논문 | 미공개 또는 crypto 미적용 |
| 데이터 확보성 | 별도 수집 필요 | API 추가 연동 필요 | Binance OHLCV 즉시 사용 |
| 구현 난이도 | HFT 인프라 필요 | ML 파이프라인 필요 | VectorBT 단순 구현 |
| 수용 용량 | $100K 미만 | $100K~$1M | $1M+ 운용 |
| 레짐 의존성 | 특정 레짐만 | 2~3 레짐에서 유효 | 모든 시장 환경 |

> **PASS: >= 18/30점**

---

## 후보 전략 요약 테이블

| # | 전략 | G0 점수 | 예상 Sharpe | 유형 | 핵심 alpha source | 우선순위 |
|---|------|---------|-------------|------|-------------------|----------|
| **1** | Cross-Sectional Momentum | **24/30** | 1.5~1.7 | Cross-sectional | Behavioral (herding) | **1순위** |
| **2** | Funding Rate Carry | **25/30** | 1.4~2.3 | Carry/Structural | Market structure (funding premium) | **1순위** |
| **3** | CTREND ML Trend Factor | **22/30** | 1.3~2.0 | ML Ensemble | Technical aggregate (elastic net) | **2순위** |
| **4** | Multi-Factor Ensemble | **23/30** | 2.0~2.5 | Multi-factor | 직교 alpha 결합 | **2순위** |
| **5** | Copula Pairs Trading | **20/30** | 2.0~3.0 | StatArb | Cointegration mean-reversion | **3순위** |
| **6** | Volume-Weighted TSMOM | **21/30** | 1.5~2.2 | Momentum | Volume + momentum interaction | **3순위** |
| **7** | HAR Volatility Overlay | **19/30** | N/A (overlay) | Overlay | Vol forecast error | **보조** |

---

## 전략 1: Cross-Sectional Momentum (XSMOM)

### 개요

코인 **간** 상대 강도를 활용하는 long-short 전략. 최근 N일간 수익률로 코인을 랭킹하여 상위 quintile을 매수, 하위 quintile을 매도.

**기존 TSMOM과의 핵심 차이:**
- TSMOM: 단일 코인의 자기 자신 과거 수익률 → OOS decay 87%
- XSMOM: 코인 **간** 상대 순위 → market-neutral, beta 노출 없음

### 경제적 논거

- **Herding effect**: 투자자들이 최근 상승 코인에 몰림 → 상대적 outperformance 지속
- **Attention bias**: CoinMarketCap/거래소 UI가 24h 변동률 표시 → 자기강화 루프
- **Behavioral finance 핵심 anomaly**: equity 시장에서 수십년간 검증된 구조

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [Pure Momentum in Cryptocurrency Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4138685) | Fracassi & Kogan | 30일 lookback, 7일 holding, top/bottom quintile spread 유의 |
| [Cross-sectional Momentum in Crypto Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637) | Drogen, Hoffstein, Otte | 28일 lookback, 5일 holding, **Sharpe 1.51** (시장 0.84) |
| [TSMOM & XSMOM Comprehensive Analysis](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565) | Han, Kang, Ryu (2024) | 78개 코인, 거래비용 반영 후에도 유의. Small-cap에서 강함 |
| [Cryptocurrency Momentum and Reversal](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263) | Dobrynskaya | 2~4주 모멘텀, 4주+ 반전. Equity보다 전환점 빠름 |

### 구현 설계

```
Universe: Binance USDT-M perp 시가총액 Top 30~50 (volume filter)
Lookback: 21~28일 수익률 랭킹
Portfolio: Long top quintile (6~10 coins), Short bottom quintile
Rebalance: 주 1회 (5~7일 holding)
Sizing: Equal-weight 또는 inverse-vol-weight
Risk: Vol-target 35%, max single coin 15%
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **5** | Behavioral finance 핵심 (herding, attention). 수십년 equity 실증 |
| 참신성 | **3** | SSRN 2022~2024 다수 논문. Crypto에서는 비교적 최근 |
| 데이터 확보성 | **5** | Binance OHLCV 즉시 사용 |
| 구현 난이도 | **4** | 멀티에셋 랭킹 + long-short. 기존 EDA batch order 활용 가능 |
| 수용 용량 | **4** | Top 30 코인 대상이면 $1M+ 가능 |
| 레짐 의존성 | **3** | Cross-sectional이므로 시장 방향 무관하지만 횡보장에서 spread 축소 가능 |
| **합계** | **24/30** | **PASS** |

### 실패 모드 회피 분석

- **교훈 #1 (단일 지표 오버피팅)**: 파라미터 1개 (lookback). Rank-based이므로 연속 변수 최적화 없음
- **교훈 #7 (TSMOM OOS decay)**: Cross-sectional은 TSMOM과 직교. Market-neutral 설계
- **예상 IS/OOS decay**: < 30% (cross-sectional anomaly는 equity에서도 OOS 지속성 높음)

---

## 전략 2: Funding Rate Carry

### 개요

Perpetual futures의 **funding rate**를 alpha source로 활용. 양의 funding rate 코인을 short (carry 수취), 음의 funding rate 코인을 long (역carry 수취)하여 cross-sectional carry spread 확보.

**핵심**: 가격 예측이 아닌 **시장 구조에 내재된 risk premium** 수취.

### 경제적 논거

- **구조적 long bias**: 소매 투자자의 레버리지 매수 수요 → 양의 funding rate (long이 short에게 지불)
- **Arbitrage capital 부족**: 규제/마진 요건으로 차익거래 자본이 충분히 유입되지 않음
- **FX carry trade와 동일 원리**: 고금리 통화 매수/저금리 통화 매도의 crypto 버전

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [BIS Working Paper No. 1087: Crypto Carry](https://www.bis.org/publ/work1087.pdf) | BIS (2023) | 평균 carry 연 6~8%, 극단기 40%+. 구조적 premium 확인 |
| [Risk and Return of Cryptocurrency Carry Trade](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4666425) | Fan, Jiao, Lu, Tong (2024) | Long-short carry: **연 46.7%, Sharpe 0.77~1.60** |
| [Predictability of Funding Rates](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424) | Inan (2025) | DAR 모델로 다음 funding rate 예측 가능 |
| [Cryptocurrency as Investable Asset Class](https://arxiv.org/html/2510.14435v2) | arXiv (2025) | Full sample Sharpe 6.45, 2024 Sharpe 4.06 |
| [Presto Research: Funding Fee Arbitrage](https://www.prestolabs.io/research/optimizing-funding-fee-arbitrage) | Presto Labs | Static: 연 18%, Sharpe 1.4 / ML dynamic: 연 31%, Sharpe 2.3 |

### 구현 설계 (2가지 변형)

**변형 A: Cross-Asset Carry Sort (추천)**
```
Data: 매 8시간 Binance funding rate (전 USDT-M perp)
Universe: Top 30~50 코인 (funding rate 데이터 존재)
Signal: Funding rate 기준 cross-sectional 랭킹
Portfolio: Long bottom quintile (낮은/음의 FR), Short top quintile (높은 FR)
Rebalance: 일 1회 또는 8시간마다
Sizing: Funding rate 절대값에 비례 (carry 크기 반영)
Risk: Delta-neutral by construction (long-short balanced)
```

**변형 B: Funding Rate Prediction**
```
Features: 과거 FR, basis premium, OI 변화율, volume, 가격 momentum
Model: DAR (Double Autoregressive) 또는 simple linear
Signal: 예측 FR > threshold → short perp / 예측 FR < -threshold → long perp
Rebalance: 8시간 주기
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **5** | BIS 공식 인정. FX carry와 동일 구조적 premium |
| 참신성 | **4** | Crypto carry는 2023~2025 논문. Equity carry와 유사하지만 crypto 고유 |
| 데이터 확보성 | **4** | Binance funding rate API 제공. OHLCV 외 추가 데이터 필요 |
| 구현 난이도 | **4** | 변형 A는 단순 sorting. 변형 B는 DAR 모델 필요 |
| 수용 용량 | **4** | 유동성 높은 perp 대상, $1M+ 가능 |
| 레짐 의존성 | **4** | 강세장/약세장 모두 funding premium 존재 (방향만 변동) |
| **합계** | **25/30** | **PASS** |

### 리스크 요인

- **2025년 carry compression**: 시장 성숙화로 funding rate 축소 추세. Dynamic sizing 필수
- **Extreme volatility**: Flash crash 시 funding rate 급변 → stop-loss 필요
- **Exchange risk**: 단일 거래소(Binance) 의존

---

## 전략 3: CTREND (ML-Aggregated Trend Factor)

### 개요

28개 technical indicator를 **elastic net (L1+L2 regularization)**으로 결합하여 cross-sectional trend signal 생성. 단일 지표가 아닌 **regularized multi-indicator ensemble**.

### 경제적 논거

- **Information aggregation**: 개별 indicator는 partial information만 포착. ML 결합으로 full signal 추출
- **Cross-sectional ranking**: 시장 전체 대비 상대적 trend 강도 → market-neutral
- **Known factor 독립**: Market, size, momentum factor로 explain 불가 → 새로운 alpha source

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [A Trend Factor for the Cross-Section of Cryptocurrency Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4601972) | Fieberg, Liedtke, Poddig, Walker, Zaremba (JFQA, 2025) | 3,000+ 코인, **median Sharpe 1.34**, subperiod/market state robust |

### 핵심 방법론

**Input Features (28개 technical signals):**
- **Momentum**: MACD, MACD signal, rate of change (ROC) 다중 기간
- **Oscillators**: RSI, CCI, Williams %R, Stochastic
- **Moving Averages**: SMA/EMA cross (multi-horizon)
- **Volume**: OBV, Volume MACD, Chaikin Money Flow
- **Volatility**: Bollinger Band midpoint, ATR ratio

**Feature Importance (상위):**
1. Bollinger Band midpoint
2. Commodity Channel Index (CCI)
3. MACD
4. Volume MACD (volume EMA difference)
5. Chaikin Money Flow

**ML Pipeline:**
```
1. 28개 indicator 계산 (일별, 코인별)
2. Cross-sectional rank 변환 (outlier 제거)
3. Elastic net regression (alpha=0.5, lambda via CV)
   - Training: rolling 52주 window
   - Target: 다음 1주 수익률
4. CTREND score = predicted return rank
5. Long top quintile / Short bottom quintile
6. 주간 리밸런싱
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **4** | Information aggregation. 개별 indicator보다 ensemble이 우월한 건 well-established |
| 참신성 | **4** | JFQA 2025 게재 (top-tier journal). Crypto 특화 |
| 데이터 확보성 | **5** | Binance OHLCV만으로 전체 구현 가능 |
| 구현 난이도 | **3** | Elastic net 학습 파이프라인 필요. Sklearn으로 구현 가능하지만 rolling window 관리 복잡 |
| 수용 용량 | **3** | Large-cap 코인에서도 유의하지만, original 연구는 3,000+ 코인 사용 |
| 레짐 의존성 | **3** | Subperiod robust 확인. 그러나 전체 레짐에서 동일 성과 보장은 아님 |
| **합계** | **22/30** | **PASS** |

### 실패 모드 회피 분석

- **교훈 #1 (단일 지표)**: 28개 indicator의 regularized ensemble → 특정 지표 의존 없음
- **교훈 #6 (파라미터 민감성)**: Elastic net의 regularization이 자동으로 sparse selection → plateau 형성
- **위험**: 학습 파이프라인의 look-ahead bias 주의. Rolling window 엄격 관리 필수

---

## 전략 4: Multi-Factor Ensemble Portfolio

### 개요

**직교(orthogonal) alpha source** 3~5개를 결합한 cross-sectional long-short 포트폴리오. 개별 전략의 한계를 극복하고 **다각화를 통한 Sharpe 제곱근 법칙** 활용.

### 경제적 논거

- **Factor diversification**: 독립적 alpha source N개 결합 시 Sharpe ≈ sqrt(N) * 개별 Sharpe
- **Factor disagreement**: 서로 다른 signal이 상충할 때 자연스러운 turnover 감소
- **Robust by design**: 개별 factor가 일시적으로 실패해도 portfolio 수준에서 buffering

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [Cross-Sectional Alpha Factors in Crypto: 2+ Sharpe](https://blog.unravel.finance/p/cross-sectional-alpha-factors-in) | Unravel Finance | 3개 직교 포트폴리오, 각 **Sharpe ~2.5** |
| [Optimizing Cryptocurrency Returns: Factor-Based Investing](https://www.mdpi.com/2227-7390/12/9/1351) | MDPI (2024) | Market, size, momentum, liquidity factor 유의 |
| [Rules-Based Investing in the Cryptocurrency Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6072768) | Gokcen (SSRN) | 체계적 factor portfolio |
| [Diversification Benefits of Crypto Factor Portfolios](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4319598) | Han et al. (2024) | Crypto factor가 전통 자산 포트폴리오에 OOS 분산 효과 |

### 결합할 직교 팩터

| Factor | Alpha Source | XSMOM 상관 | 데이터 |
|--------|-------------|------------|--------|
| **XSMOM** (전략 1) | Behavioral (herding) | — | OHLCV |
| **Carry** (전략 2) | Structural (funding) | 낮음 | Funding rate API |
| **CTREND** (전략 3) | Technical aggregate | 부분 중복 | OHLCV |
| **Inverse Size** | Liquidity premium | 독립 | Market cap |
| **Volume Shock** | Short-term overreaction | 음의 상관 | OHLCV |

### 구현 설계

```
1. Factor computation (일별, 코인별):
   a. XSMOM: 21일 수익률 cross-sectional rank
   b. Carry: 8시간 funding rate cross-sectional rank
   c. CTREND: elastic net aggregate signal rank (주간 업데이트)
   d. Size: log(market_cap) inverse rank
   e. Volume shock: 5일 abnormal volume rank

2. Factor combination:
   a. 각 factor → z-score 정규화
   b. Equal-weight average (최적화 weight 사용 금지 → 오버피팅 방지)
   c. 선택: IC-weighted (직전 60일 Spearman IC 기반)

3. Portfolio construction:
   a. Combined score로 cross-sectional 랭킹
   b. Long top 20%, Short bottom 20%
   c. Inverse-vol position sizing per coin
   d. Gross exposure target: 200%

4. Risk overlay:
   a. Vol-target 35% annualized
   b. Max single coin weight: 15%
   c. Turnover constraint: max 50%/week
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **5** | Factor diversification은 portfolio theory 핵심. Markowitz 이후 검증됨 |
| 참신성 | **3** | Multi-factor 자체는 전통적. Crypto에 적용한 논문은 최근 |
| 데이터 확보성 | **4** | OHLCV + funding rate + market cap. 대부분 Binance API |
| 구현 난이도 | **3** | 5개 factor pipeline + 결합 + rebalancing. 기존 EDA 인프라 활용 가능 |
| 수용 용량 | **4** | Multi-factor는 turnover 분산 → capacity 높음 |
| 레짐 의존성 | **4** | 직교 factor 결합으로 레짐 의존성 최소화 |
| **합계** | **23/30** | **PASS** |

### 핵심 인사이트

- **6개 이상 factor 결합 시 marginal benefit → 0** (Unravel Finance)
- **3개 직교 포트폴리오가 최적**: 복잡도 대비 효율 극대화
- Equal-weight 결합이 optimized weight보다 OOS에서 robust

### 구현 순서

전략 1, 2, 3을 **먼저 개별 검증**한 후, 검증된 factor들만 결합하여 Multi-Factor Portfolio 구성.

---

## 전략 5: Copula-Based Pairs Trading

### 개요

코인 쌍의 **비선형 의존 구조**를 copula로 모델링하고, 조건부 확률 이탈 시 spread 진입. 기존 linear cointegration보다 robust한 market-neutral 전략.

### 경제적 논거

- **Crypto 쌍의 강한 공적분**: 같은 sector 코인들(L1 vs L1, DeFi vs DeFi)은 구조적 공적분
- **비선형 의존성**: Crypto tail event에서 상관관계 급변 → copula가 이를 포착
- **Market-neutral**: spread 기반이므로 시장 방향 무관

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [Copula-Based Trading of Cointegrated Cryptocurrency Pairs](https://link.springer.com/article/10.1186/s40854-024-00702-7) | Tadi & Witzany (Financial Innovation, 2025) | 5분 데이터, **Sharpe ~1.0~3.77**, copula BB7/BB8 최적 |
| [Trading Games: Beating Passive Strategies](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018) | Palazzi (J. Futures Markets, 2025) | Top-10 코인 90개 pair, optimized pairs > passive 전 지표 |
| [Profiting Off High Correlation of Crypto Pairs](https://link.springer.com/chapter/10.1007/978-3-031-68974-1_16) | Springer (2024) | **Win rate 79~100%** (live trading) |

### 구현 설계

```
Formation Period (3주):
1. Engle-Granger + KSS 공적분 검정으로 pair 후보 선별
2. Kendall's Tau 상관성 랭킹
3. Marginal distribution fitting (Student-t)
4. Copula family 선택 (BB7, BB8, Tawn) via AIC

Trading Period (1주):
1. Conditional copula probability h(1|2), h(2|1) 계산
2. h(1|2) < alpha & h(2|1) > 1-alpha → spread 진입
3. Probability → 0.5 수렴 시 청산
4. Stop-loss: spread 2.5 sigma 초과 시

Rolling: 3주 formation + 1주 trading = 104 cycles/year
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **4** | 공적분 관계의 일시적 이탈 → 복원. 구조적으로 합리적 |
| 참신성 | **3** | Copula pairs는 FX/equity에서 사용. Crypto 적용은 2024~2025 |
| 데이터 확보성 | **5** | Binance OHLCV 즉시 사용 (5분/1시간) |
| 구현 난이도 | **2** | Copula fitting, marginal distribution, 공적분 검정. scipy/statsmodels 가능하나 복잡 |
| 수용 용량 | **3** | Pair별 capacity 제한. 여러 pair 분산 필요 |
| 레짐 의존성 | **3** | 공적분 관계 자체가 레짐에 따라 변동 가능 (formation window가 adaptive) |
| **합계** | **20/30** | **PASS** |

### 리스크 요인

- **Cointegration breakdown**: Crypto 섹터 재편 시 쌍 관계 붕괴
- **구현 복잡도 높음**: Copula family selection, marginal fitting 등 기술적 난이도
- **Execution risk**: Spread가 좁으므로 슬리피지 영향 큼

---

## 전략 6: Volume-Weighted Time-Series Momentum (VW-TSMOM)

### 개요

기존 TSMOM에 **volume 가중치**를 추가하여 signal quality를 개선. Volume이 높은 시점의 momentum에 더 큰 가중치를 부여.

### 경제적 논거

- **Volume = Conviction**: 고거래량 동반 모멘텀은 informed trading 가능성 높음
- **Noise filtering**: 저거래량 구간의 가격 변동(noise)을 자동 필터링
- **기존 TSMOM의 약점 보완**: Volume dimension 추가로 오버피팅 감소

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [Cryptocurrency Volume-Weighted Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389) | Huang, Sangiorgi, Urquhart (2024) | 일일 0.94% excess return, **Sharpe 2.17**. Known factor로 explain 불가 |

### 구현 설계

```
1. Volume-weighted return 계산:
   VW_ret(t, lookback) = Σ(vol_i * ret_i) / Σ(vol_i)  (i = t-lookback to t-1)

2. Winner-minus-Loser portfolio:
   - 코인별 VW-TSMOM score 계산
   - Cross-sectional ranking
   - Long top quintile (winners), Short bottom quintile (losers)

3. Rebalance: 일 1회 또는 주 1회
4. Risk: Vol-target 35%, trailing stop 3x ATR
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **4** | Volume-conviction hypothesis. TSMOM + volume의 interaction |
| 참신성 | **3** | SSRN 2024 논문. Volume-weighted는 equity에서도 연구됨 |
| 데이터 확보성 | **5** | Binance OHLCV (volume 포함) 즉시 사용 |
| 구현 난이도 | **4** | 기존 TSMOM에 volume weighting 추가. 단순 |
| 수용 용량 | **3** | Cross-sectional이면 괜찮으나, 소형 코인 volume 제한 |
| 레짐 의존성 | **2** | Momentum 전략은 횡보장에서 약세. Volume이 완화하지만 구조적 한계 |
| **합계** | **21/30** | **PASS** |

### 주의사항

- 기존 TSMOM이 OOS decay 87%로 실패 → VW 버전도 같은 위험 있음
- **반드시 cross-sectional ranking으로 구현** (time-series 단독 사용 금지)
- Backtest 시 wash trading volume 필터링 필요

---

## 전략 7: HAR Volatility Forecast Overlay (보조 전략)

### 개요

**HAR-RV (Heterogeneous Autoregressive Realized Volatility)** 모델로 변동성을 예측하고, forecast error를 기반으로 다른 전략의 **포지션 사이징**을 조절하는 overlay.

### 경제적 논거

- **Volatility surprise = new information**: 예상보다 높은 vol → trend 가속 구간
- **Variance risk premium**: Implied vol > realized vol인 구간에서 premium 수취 가능 (OHLCV 근사)
- **Risk management**: Vol forecast로 position sizing → drawdown 방어

### 학술 근거

| 논문 | 저자 | 핵심 결과 |
|------|------|-----------|
| [Crypto Vol Forecasting: HAR, Sentiment, ML Horserace](https://link.springer.com/article/10.1007/s10690-024-09510-6) | Springer (2024) | HAR-RV가 crypto vol forecast에 consistently best |
| [Can ML Models Better Volatility Forecasting?](https://www.tandfonline.com/doi/full/10.1080/1351847X.2025.2553053) | Tandfonline (2025) | HAR 확장이 ML 대비 robust |

### 구현 설계

```
1. Parkinson volatility (OHLCV 기반):
   σ_park = sqrt(1/(4n*ln2) * Σ(ln(H/L))^2)

2. HAR-RV model (3 parameters only):
   RV_d+1 = β0 + β1*RV_daily + β2*RV_weekly + β3*RV_monthly
   (daily=1d, weekly=5d avg, monthly=22d avg realized vol)

3. Vol surprise = RV_realized - RV_predicted

4. Overlay logic:
   - Vol surprise > 0 → momentum signal confidence UP
   - Vol surprise < 0 → mean-reversion signal confidence UP
   - |Vol surprise| → position size scaling factor
   - High predicted vol → overall position size DOWN (risk mgmt)
```

### Gate 0 평가

| 항목 | 점수 | 근거 |
|------|------|------|
| 경제적 논거 | **3** | Variance risk premium 근사. 단독 alpha보다는 overlay용 |
| 참신성 | **3** | HAR-RV는 2009년 모델. Crypto 적용은 2024~2025 |
| 데이터 확보성 | **5** | Binance OHLCV (high/low 포함) 즉시 사용 |
| 구현 난이도 | **4** | OLS regression 3개 변수. 매우 단순 |
| 수용 용량 | **2** | Overlay이므로 단독 capacity 없음 |
| 레짐 의존성 | **2** | Vol forecast는 레짐 전환기에 정확도 하락 |
| **합계** | **19/30** | **PASS** (borderline) |

---

## 구현 로드맵

### Phase 1: 기초 Factor 검증 (즉시 착수)

| 순서 | 전략 | 예상 소요 | 근거 |
|------|------|----------|------|
| 1-A | **XSMOM** (전략 1) | 2~3일 | 파라미터 1개, 기존 인프라 활용, 가장 빠른 검증 |
| 1-B | **Funding Rate Carry** (전략 2) | 3~4일 | 데이터 파이프라인 추가 (FR API), 별도 alpha source |

> Phase 1의 두 전략은 **서로 직교**하므로 병렬 개발 가능.

### Phase 2: ML Factor 추가

| 순서 | 전략 | 예상 소요 | 근거 |
|------|------|----------|------|
| 2-A | **CTREND** (전략 3) | 4~5일 | 28개 indicator + elastic net pipeline |
| 2-B | **HAR Overlay** (전략 7) | 1~2일 | 단순 OLS, Phase 1 전략에 즉시 적용 |

### Phase 3: 통합 Multi-Factor

| 순서 | 전략 | 예상 소요 | 근거 |
|------|------|----------|------|
| 3 | **Multi-Factor Ensemble** (전략 4) | 3~4일 | Phase 1~2 검증된 factor 결합 |

### Phase 4: 심화 (선택)

| 순서 | 전략 | 예상 소요 | 근거 |
|------|------|----------|------|
| 4-A | **Copula Pairs** (전략 5) | 5~7일 | 구현 복잡, 별도 alpha source |
| 4-B | **VW-TSMOM** (전략 6) | 2~3일 | XSMOM과 비교 검증 |

---

## 핵심 설계 원칙 (기존 실패 방지)

1. **Cross-sectional first**: 모든 전략을 가능한 한 cross-sectional (market-neutral) 형태로 구현
2. **Equal-weight combination**: Factor weight 최적화 금지 → 오버피팅 방지
3. **Rolling OOS 검증**: 모든 backtest에서 IS/OOS split 필수 (Gate 2 기준 사전 체크)
4. **파라미터 최소화**: 전략당 핵심 파라미터 3개 이하
5. **구조적 엣지 우선**: "이 alpha가 왜 존재하는가?" 설명 불가하면 구현하지 않음

---

## 참고 문헌

### Cross-Sectional Momentum
- [Fracassi & Kogan - Pure Momentum in Cryptocurrency Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4138685)
- [Drogen, Hoffstein & Otte - Cross-sectional Momentum in Crypto Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637)
- [Han, Kang & Ryu - TSMOM & XSMOM Comprehensive Analysis](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565)
- [Dobrynskaya - Cryptocurrency Momentum and Reversal](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263)

### Funding Rate / Carry
- [BIS Working Paper No. 1087: Crypto Carry](https://www.bis.org/publ/work1087.pdf)
- [Fan, Jiao, Lu, Tong - Risk and Return of Crypto Carry Trade](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4666425)
- [Inan - Predictability of Funding Rates](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424)
- [arXiv - Cryptocurrency as Investable Asset Class](https://arxiv.org/html/2510.14435v2)
- [Presto Research - Funding Fee Arbitrage](https://www.prestolabs.io/research/optimizing-funding-fee-arbitrage)

### CTREND / ML Factor
- [Fieberg et al. - A Trend Factor for the Cross-Section (JFQA 2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4601972)
- [Huang, Sangiorgi, Urquhart - VW-TSMOM](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389)

### Multi-Factor
- [Unravel Finance - Cross-Sectional Alpha Factors in Crypto](https://blog.unravel.finance/p/cross-sectional-alpha-factors-in)
- [MDPI - Optimizing Cryptocurrency Returns: Factor-Based Investing](https://www.mdpi.com/2227-7390/12/9/1351)
- [Gokcen - Rules-Based Investing in Crypto](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6072768)
- [Han et al. - Diversification Benefits of Crypto Factor Portfolios](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4319598)

### Pairs Trading
- [Tadi & Witzany - Copula-Based Pairs Trading (Financial Innovation 2025)](https://link.springer.com/article/10.1186/s40854-024-00702-7)
- [Palazzi - Trading Games (J. Futures Markets 2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018)

### Volatility
- [HAR-RV Crypto Vol Forecasting (Springer 2024)](https://link.springer.com/article/10.1007/s10690-024-09510-6)
- [ML Vol Forecasting (European Journal 2025)](https://www.tandfonline.com/doi/full/10.1080/1351847X.2025.2553053)

### Microstructure
- [Easley et al. - VPIN: Flow Toxicity and Liquidity](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf)
- [Bitcoin Order Flow Toxicity and Price Jumps](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
