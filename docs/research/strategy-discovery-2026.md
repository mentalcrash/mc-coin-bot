# Cryptocurrency Trading Strategy Discovery (2026-02)

> Single-Asset, Long/Short 전략 리서치. 2025-2026 최신 학술 연구 및 실무 트렌드 기반.

---

## 현재 구현 현황

| Strategy | Type | Short Mode | Status |
|----------|------|------------|--------|
| VW-TSMOM | Momentum | HEDGE_ONLY | Implemented |
| BB-RSI | Mean-Reversion | DISABLED | Implemented |
| Donchian | Breakout (Turtle) | DISABLED | Implemented |
| Adaptive Breakout | Breakout + ATR | FULL | Implemented |

---

## 1. Momentum Strategies

### 1-A. Enhanced VW-TSMOM (Volume Weight 강화)

**기존 TSMOM 개선안** - 논문 기반 volume weighting 정교화

- **Core Logic**: 거래량 가중 모멘텀 + inverse volatility scaling
- **Signal**:
  ```
  vol_weight = volume / volume.rolling(lookback).mean()
  weighted_return = log_return * vol_weight
  momentum = weighted_return.rolling(lookback).sum()
  signal = sign(momentum) * vol_target / realized_vol
  ```
- **Crypto 적합성**: 거래량 급증이 institutional flow와 일치하는 경우가 많아 "진짜" 모멘텀과 noise를 구분하는 데 효과적
- **Parameters**: lookback (20-252), vol_target (15-40%), volume_lookback (20)
- **L/S**: Full Long/Short
- **Timeframe**: 1D (권장), 4H
- **Performance**: 일일 수익률 0.94%, 연간 Sharpe 2.17 (SSRN 연구)
- **Risks**: Wash trading에 취약, 저유동성 알트코인에서 slippage
- **Ref**: [Huang, Sangiorgi, Urquhart (2024) - "Cryptocurrency VW-TSMOM", SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389)
- **구현 난이도**: **LOW** (기존 TSMOM에 volume weight 로직 추가)

### 1-B. KAMA (Kaufman Adaptive Moving Average) Trend Following

**Adaptive smoothing으로 choppy 시장 자동 필터링**

- **Core Logic**: Efficiency Ratio(ER)로 smoothing constant를 동적 조절. ER 높으면(trending) 빠른 반응, ER 낮으면(choppy) 느린 반응. ATR 필터로 whipsaw 방지
- **Signal**:
  ```
  direction = abs(close - close.shift(lookback))
  volatility = abs(close.diff()).rolling(lookback).sum()
  ER = direction / volatility

  fast_sc = 2 / (fast_period + 1)
  slow_sc = 2 / (slow_period + 1)
  sc = (ER * (fast_sc - slow_sc) + slow_sc) ** 2
  KAMA = kama_recursive(close, sc)

  atr = ATR(high, low, close, period=14)
  long_entry  = (close > KAMA + atr_mult * atr) & (KAMA > KAMA.shift(1))
  short_entry = (close < KAMA - atr_mult * atr) & (KAMA < KAMA.shift(1))
  ```
- **Crypto 적합성**: BTC/ETH의 급격한 trending/choppy 교차에서 KAMA가 자동으로 adaptive. 고정 MA보다 월등한 noise 필터링
- **Parameters**: lookback (10), fast_period (2), slow_period (30), atr_period (14), atr_multiplier (1.0-2.0)
- **L/S**: Full Long/Short
- **Timeframe**: 1H, 4H, 1D
- **Risks**: KAMA 재귀 계산으로 초기 warmup 긴 편, lookback 민감성
- **Ref**: [Kaufman (1998) - "Trading Systems and Methods"](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama)
- **구현 난이도**: **LOW** - 벡터화 가능, BaseStrategy 패턴 적용 용이

---

## 2. Mean-Reversion Strategies

### 2-A. Z-Score Mean Reversion (Dynamic Lookback)

**변동성 regime에 따라 lookback 자동 조정**

- **Core Logic**: 가격의 Z-score를 동적 lookback으로 계산. 고변동기에는 짧은 lookback(빠른 반응), 저변동기에는 긴 lookback(noise 무시)
- **Signal**:
  ```
  vol = returns.rolling(20).std()
  vol_pct = vol.rolling(252).rank(pct=True)
  lookback = np.where(vol_pct > 0.7, short_lb, long_lb)

  z = (close - rolling_mean(close, lookback)) / rolling_std(close, lookback)
  long_entry  = z < -entry_z   # e.g., -2.0
  short_entry = z > entry_z    # e.g., +2.0
  exit = abs(z) < exit_z       # e.g., 0.5
  ```
- **Crypto 적합성**: Crypto의 비정상적 volatility clustering에서 고정 lookback은 부적합. 동적 조정이 regime 변화에 빠르게 적응
- **Parameters**: short_lb (10-20), long_lb (40-60), entry_z (1.5-2.5), exit_z (0.0-0.5)
- **L/S**: Full Long/Short
- **Timeframe**: 1H, 4H, 1D
- **Risks**: Z-score 가정(정규분포)이 crypto fat tail에서 위반될 수 있음
- **구현 난이도**: **LOW** - 벡터화 가능, 단순한 통계 모델

### 2-B. Stochastic RSI + Bollinger Bands Multi-Factor

**두 독립 지표의 교차 확인으로 false signal 감소**

- **Core Logic**: StochRSI와 BB %B가 동시에 극단값을 보일 때만 진입
- **Signal**:
  ```
  stoch_rsi = stochastic(RSI(close, 14), k=14, d=3)
  bb_pct = (close - bb_lower) / (bb_upper - bb_lower)

  long_entry  = (stoch_rsi.k < 0.10) & (bb_pct < 0.05)
  short_entry = (stoch_rsi.k > 0.90) & (bb_pct > 0.95)
  ```
- **Crypto 적합성**: Intraday 과매도/과매수 탐지에 유효. 두 지표의 독립적 확인이 noise 감소
- **Parameters**: rsi_period (14), stoch_k (14), stoch_d (3), bb_period (20), bb_std (2.0)
- **L/S**: Full Long/Short
- **Timeframe**: 5M, 15M, 1H (단기 전략)
- **Ref**: [FMZ Quant (2025) - "Multi-Factor Mean Reversion"](https://www.fmz.com/lang/en/strategy/489893)
- **구현 난이도**: **LOW** - 기존 BB-RSI 확장

---

## 3. Volatility Strategies

### 3-A. Garman-Klass Volatility Breakout

**OHLC 전체를 활용한 고효율 변동성 추정 + 압축/확장 탐지**

- **Core Logic**: Garman-Klass estimator로 intrabar 가격 움직임까지 반영한 변동성 계산. 변동성 압축(squeeze) 후 브레이크아웃 시 진입
- **Signal**:
  ```
  gk_vol = 0.5 * log(high/low)**2 - (2*log(2)-1) * log(close/open)**2
  gk_ma = gk_vol.rolling(lookback).mean()
  vol_ratio = gk_vol / gk_ma

  is_compressed = vol_ratio < compression_threshold  # e.g., 0.5

  breakout_up   = is_compressed.shift(1) & (close > high.rolling(20).max().shift(1))
  breakout_down = is_compressed.shift(1) & (close < low.rolling(20).min().shift(1))
  ```
- **Crypto 적합성**: 24/7 거래로 OHLC 데이터 풍부. GK estimator가 close-to-close vol보다 5-7배 효율적. BTC realized vol 최저점이 대형 가격 이동 직전에 발생함이 실증됨
- **Parameters**: gk_lookback (20-60), compression_threshold (0.3-0.6), breakout_lookback (20)
- **L/S**: Full Long/Short
- **Timeframe**: 4H, 1D
- **Risks**: 변동성 compression이 길어지면 false breakout 다수 발생
- **Ref**: [Garman & Klass (1980)](https://doi.org/10.1086/296046), [Djanga et al. (2024) - "Crypto Volatility Forecasting", SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5048674)
- **구현 난이도**: **MEDIUM** - GK 계산은 단순하나 compression detection 로직 필요

### 3-B. Variance Risk Premium (VRP) Strategy

**내재변동성(IV) vs 실현변동성(RV) 스프레드 활용**

- **Core Logic**: VRP = IV - RV. VRP가 높으면 시장이 과도한 공포(매수 기회), 낮으면 안일(매도 기회)
- **Signal**:
  ```
  IV = deribit_dvol_index()          # Deribit BTC DVOL
  RV = parkinson_vol(high, low, 30)  # 또는 GK vol
  VRP = IV - RV

  long_signal  = VRP > vrp_upper    # e.g., 20%p (과도한 공포)
  short_signal = VRP < vrp_lower    # e.g., -10%p (과도한 안일)
  ```
- **Crypto 적합성**: Crypto 옵션 시장에서 VRP는 전통 자산보다 크고 지속적. Deribit BTC IV가 RV를 평균 10-15%p 초과하는 구조적 특성
- **Parameters**: iv_source (Deribit DVOL), rv_estimator (Parkinson/GK), rv_period (30), vrp_thresholds
- **L/S**: Long bias (구조적으로 short vol = long underlying)
- **Timeframe**: 1D
- **Risks**: Deribit API 의존, 블랙스완 이벤트 시 VRP 급변, 옵션 데이터 없는 알트코인 적용 불가
- **Ref**: [Wiley (2025) - "Pricing Crypto Options With Vol of Vol"](https://onlinelibrary.wiley.com/doi/10.1002/fut.70029?af=R)
- **구현 난이도**: **MEDIUM** - 외부 데이터(Deribit API) 연동 필요

---

## 4. Regime Detection / Adaptive Strategies

### 4-A. HMM (Hidden Markov Model) Regime-Switching

**시장을 Bull/Bear/Sideways 3-state로 분류, 전략 자동 전환**

- **Core Logic**: GaussianHMM으로 시장 regime 분류 후, regime별 최적 전략/파라미터 적용
- **Signal**:
  ```
  features = [returns, volatility, volume_change]
  hmm = GaussianHMM(n_components=3, covariance_type="full")
  regime = hmm.predict(features)[-1]

  strategy_map = {
      BULL:     TsmomStrategy(lookback=20, vol_target=0.25),   # 공격적 momentum
      BEAR:     BbRsiStrategy(bb_std=2.5),                      # tight mean-reversion
      SIDEWAYS: None,                                            # cash (관망)
  }
  ```
- **Crypto 적합성**: BTC의 regime 전환은 전통 자산보다 급격하고 명확. 2025 학술 연구에서 HMM이 bullish/bearish/neutral 전환을 유의미하게 예측. Bayesian MCMC + HMM으로 1, 5, 30-step ahead forecasting 가능
- **Parameters**: n_states (3), features (returns, vol, volume), retrain_window (90-365일), transition_smoothing
- **L/S**: Regime 의존 (Bull=Long bias, Bear=Short bias, Sideways=Flat)
- **Timeframe**: 1D (regime detection), 4H/1H (execution)
- **Risks**: Regime 전환 감지 lag (1-3일), 과적합된 transition matrix
- **Ref**: [AJPAS (2025) - "HMM in Detecting Regime Changes in Bitcoin"](https://doi.org/10.9734/ajpas/2025/v27i7781), [MDPI (2025) - "Bitcoin Regime Shifts: Bayesian MCMC + HMM"](https://www.mdpi.com/2227-7390/13/10/1577)
- **구현 난이도**: **MEDIUM** - `hmmlearn` 의존성, 학습/재학습 파이프라인 필요

### 4-B. Volatility-Regime Adaptive Parameters

**변동성 수준에 따라 전략 파라미터 동적 조정 (Meta-Strategy)**

- **Core Logic**: 현재 변동성의 과거 percentile에 따라 전략 config를 자동 전환
- **Signal**:
  ```
  vol_20d = returns.rolling(20).std() * sqrt(252)
  vol_pct = vol_20d.rolling(252).rank(pct=True)

  if vol_pct > 0.8:       # High vol → 보수적
      config = Config(lookback=10, vol_target=0.10, stop_atr=2.0)
  elif vol_pct < 0.2:     # Low vol → 공격적
      config = Config(lookback=60, vol_target=0.30, stop_atr=4.0)
  else:                    # Normal
      config = Config(lookback=20, vol_target=0.20, stop_atr=3.0)
  ```
- **Crypto 적합성**: 프로젝트 Risk Parameter Sweep에서 이미 확인된 패턴 (Trailing Stop 3.0x ATR 최적, Rebalance 10% 최적). 이를 자동화하는 wrapper 전략
- **Parameters**: vol_regime_thresholds (0.2, 0.8), config_map (regime별 파라미터 세트)
- **L/S**: 기반 전략에 따라 가변
- **Timeframe**: 1D (regime 판단), 기반 전략 TF (execution)
- **Risks**: 과도한 parameter switching은 curve-fitting 위험
- **구현 난이도**: **LOW** - 기존 전략 wrapper, config 전환 로직만 추가

### 4-C. On-Chain Metrics Regime Filter (MVRV / SOPR / NVT)

**온체인 데이터를 매크로 regime indicator로 활용**

- **Core Logic**: MVRV Z-Score, SOPR, NVT 등 온체인 지표로 macro scoring → 기반 전략의 long/short bias 조정
- **Signal**:
  ```
  mvrv = get_mvrv_zscore()
  sopr_28d = get_sopr_change(28)

  macro_score = 0
  if mvrv < 1.0:   macro_score += 2   # 극단적 저평가
  elif mvrv > 3.5: macro_score -= 1   # 고평가

  if sopr_28d < -0.05: macro_score += 1  # 손실 매도 가속 → 바닥 근접

  bias = {score >= 2: 1.5, score <= -2: 0.3}.get(macro_score, 1.0)
  final_signal = base_signal * bias
  ```
- **Crypto 적합성**: Crypto만의 고유 alpha source. 2026년 2월 MVRV Z-Score 1.32는 "과열 아닌" 구간. MVRV, NVT, SOPR의 가격 예측력은 다수 연구에서 확인
- **Parameters**: mvrv_thresholds, sopr_window (28d), data_source (Glassnode/CryptoQuant)
- **L/S**: Bias 조정 (방향성 강화/약화)
- **Timeframe**: 1D, 1W (macro indicator)
- **Risks**: 온체인 API 의존성, BTC 전용 (알트코인 데이터 부족), 이번 사이클 top indicator 실패 사례 존재
- **Ref**: [Nansen - "Onchain Metrics for Crypto Price Prediction"](https://www.nansen.ai/post/onchain-metrics-key-indicators-for-cryptocurrency-price-prediction)
- **구현 난이도**: **MEDIUM** - 외부 API 연동 (Glassnode/CryptoQuant)

---

## 5. Microstructure Strategies

### 5-A. VPIN (Volume-Synchronized Probability of Informed Trading)

**정보 거래자 활동 확률로 가격 점프 예측**

- **Core Logic**: Volume bucket 기반 trade classification으로 buy/sell imbalance 측정. VPIN spike가 가격 점프(flash crash/pump) 예측
- **Signal**:
  ```
  buy_vol, sell_vol = classify_trades_bulk(trades, bucket_size=V)
  vpin = abs(buy_vol - sell_vol) / (buy_vol + sell_vol)
  vpin_ma = vpin.rolling(50).mean()

  is_toxic = vpin > vpin_ma + 2 * vpin.rolling(50).std()
  direction = sign(buy_vol - sell_vol)
  signal = direction * is_toxic
  ```
- **Crypto 적합성**: Information asymmetry 극심 (고래, 내부자). 2025 연구에서 VPIN이 Bitcoin 가격 점프를 유의미하게 예측
- **Parameters**: bucket_size (volume units), rolling_window (50), toxicity_threshold (2 sigma)
- **L/S**: Full Long/Short (order flow imbalance 방향)
- **Timeframe**: 1M, 5M (tick-level data 필요)
- **Risks**: Tick data 접근 필요, 거래소 API 한계, 고빈도 실행 인프라
- **Ref**: [ScienceDirect (2025) - "Bitcoin order flow toxicity and price jumps"](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
- **구현 난이도**: **HIGH** - tick data 인프라 + 실시간 처리 필요

### 5-B. Order Book Imbalance (OBI)

**호가창 bid/ask 볼륨 불균형으로 단기 가격 방향 예측**

- **Core Logic**: Top N levels의 bid/ask volume 비율로 단기 방향 예측
- **Signal**:
  ```
  bid_vol = sum(bid_quantities[:N])
  ask_vol = sum(ask_quantities[:N])
  obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
  signal = sign(obi) if abs(obi) > threshold else 0
  ```
- **Crypto 적합성**: 2025 연구에서 LOB microstructure features가 AUC > 0.55 달성. 동일 feature 중요도가 BTC/LTC/ETC 등 자산 간 전이 가능
- **Parameters**: depth_levels (5-20), update_frequency (100ms-1s), signal_threshold
- **L/S**: Full Long/Short
- **Timeframe**: Sub-1M (HFT), 5M (저빈도 변형)
- **Risks**: Latency 민감, spoofing/layering 영향
- **Ref**: [ArXiv (2025) - "Explainable Patterns in Cryptocurrency Microstructure"](https://arxiv.org/html/2602.00776)
- **구현 난이도**: **HIGH** - 실시간 orderbook data stream 필요

---

## 6. DeFi / Derivatives Strategies

### 6-A. Funding Rate Arbitrage (Spot-Perp Basis)

**Perpetual futures funding rate 수익화 (Delta-Neutral)**

- **Core Logic**: Funding rate가 양수이면 spot long + perp short (funding 수취), 음수이면 반대
- **Signal**:
  ```
  fr = get_funding_rate(symbol)           # 8h 정산
  annualized = fr * 3 * 365

  if fr > entry_threshold:                # e.g., 0.03%
      action = "spot_long + perp_short"   # funding 수취
  elif fr < -entry_threshold:
      action = "spot_short + perp_long"
  if abs(fr) < exit_threshold:
      action = "close_all"
  ```
- **Crypto 적합성**: 평균 funding rate 0.015%/8h (연간 15-20% yield). BTC open interest $150-200B 이상으로 시장 깊이 충분
- **Parameters**: entry_threshold (0.02-0.05%), exit_threshold (0.005%), max_position
- **L/S**: Delta-neutral (market-neutral)
- **Timeframe**: 8H (funding 정산 주기)
- **Performance**: 연간 15-20% (보수적)
- **Risks**: 급격한 funding rate 역전, liquidation risk, exchange counterparty risk
- **Ref**: [ScienceDirect (2025) - "Funding Rate Arbitrage Risk/Return Profiles"](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- **구현 난이도**: **MEDIUM** - Binance API perp 연동, 두 포지션 동시 관리

### 6-B. Futures Basis (Cash-and-Carry)

**Spot vs Quarterly Futures 프리미엄 capture**

- **Core Logic**: Contango 시 spot long + quarterly futures short, basis compression/만기 시 청산
- **Signal**:
  ```
  basis = (futures_price - spot_price) / spot_price
  annualized_basis = basis * (365 / days_to_expiry)

  if annualized_basis > min_yield:    # e.g., 15%
      position = "spot_long + futures_short"
  if annualized_basis < exit_yield or days_to_expiry < 3:
      position = "close_all"
  ```
- **Crypto 적합성**: 2024년 basis 25% (2월), 2025년에도 10% 수준 유지. SOL/XRP futures 50% annualized basis 기록
- **Parameters**: min_yield (10-15%), exit_yield (2-5%), max_days_to_expiry
- **L/S**: Delta-neutral
- **Timeframe**: 1D (포지션 관리), 1H (entry timing)
- **Performance**: 연간 10-25%
- **Ref**: [BIS Working Paper 1087 - "Crypto Carry"](https://www.bis.org/publ/work1087.pdf), [CME Group (2025)](https://www.cmegroup.com/openmarkets/equity-index/2025/Spot-ETFs-Give-Rise-to-Crypto-Basis-Trading.html)
- **구현 난이도**: **MEDIUM** - quarterly futures API 연동, roll 전략 설계

---

## 7. ML / Statistical Strategies

### 7-A. Regime-Specialist Random Forest Ensemble

**HMM regime 분류 + regime별 전문 RF 모델**

- **Core Logic**: HMM으로 regime 분류 후, 각 regime에 특화된 Random Forest가 포지션 결정
- **Signal**:
  ```
  regime = hmm.predict([returns, vol, volume_change])[-1]  # Bull/Bear/Sideways

  features = [rsi, macd, bb_pct, volume_ratio, funding_rate, oi_change]
  signal = rf_models[regime].predict_proba(features)
  position = 1 if signal[1] > 0.6 else (-1 if signal[0] > 0.6 else 0)
  ```
- **Crypto 적합성**: 단일 모델은 regime 변화에 취약. 전문가 앙상블이 crypto의 급격한 regime switch에 robust
- **Parameters**: n_regimes (3), rf_n_estimators (100-500), retrain_period (30-90일), confidence (0.6)
- **L/S**: Full Long/Short
- **Timeframe**: 4H, 1D
- **Risks**: HMM regime label이 실시간에서 noisy, 모델 staleness, 과적합
- **Ref**: [AJPAS (2025)](https://doi.org/10.9734/ajpas/2025/v27i7781), [GitHub: CryptoMarket Regime Classifier](https://github.com/akash-kumar5/CryptoMarket_Regime_Classifier)
- **구현 난이도**: **HIGH** - ML pipeline + HMM + 다중 모델 학습/서빙

### 7-B. PPO Position Sizing (Reinforcement Learning)

**연속 action space에서 position size 직접 출력**

- **Core Logic**: PPO 에이전트가 시장 상태를 관찰하고 [-1, +1] position size 출력. Risk-aware reward로 drawdown 제한
- **Signal**:
  ```
  state = normalize([ohlcv, indicators, portfolio_state, risk_metrics])
  position = ppo_agent.predict(state)  # float in [-1, 1]
  reward = return - lambda * max(0, drawdown - max_dd)
  ```
- **Crypto 적합성**: 이산 action(buy/sell/hold)보다 연속 sizing이 고변동 환경에서 granular한 리스크 관리 가능. 2025년 11월 급락 시 RL 전략이 2시간 내 regime 적응하여 +4.7% (시장 -11%)
- **Parameters**: learning_rate (3e-4), clip_ratio (0.2), entropy_coef (0.01), drawdown_lambda
- **L/S**: Full Long/Short (continuous action)
- **Timeframe**: 1H, 4H
- **Ref**: [SSRN (2025) - "Risk-Aware Deep RL for Crypto Trading"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5662930)
- **구현 난이도**: **HIGH** - RL 환경 구축, stable-baselines3, 학습 인프라

### 7-C. CNN-LSTM Hybrid Price Prediction

**CNN(패턴 추출) + LSTM(시퀀스 의존성) 가격 방향 예측**

- **Core Logic**: CNN으로 candlestick patterns 추출, LSTM으로 temporal dependencies 모델링
- **Signal**:
  ```
  x = cnn_extractor(ohlcv_window)  # local patterns
  pred = lstm_model(x)              # temporal dependencies
  signal = sign(pred - close)
  position_size = abs(pred - close) / close * vol_target / realized_vol
  ```
- **Crypto 적합성**: Crypto의 비선형 패턴(급등/급락)을 CNN이 포착하고 LSTM이 serial correlation 활용
- **Parameters**: cnn_filters (32-128), lstm_units (64-256), seq_length (30-90), dropout (0.2-0.4)
- **L/S**: Full Long/Short
- **Timeframe**: 1H, 4H
- **Ref**: [Springer (2025) - "ML approaches to crypto trading optimization"](https://link.springer.com/article/10.1007/s44163-025-00519-y)
- **구현 난이도**: **HIGH** - DL 파이프라인, GPU 학습 환경

---

## 구현 우선순위 매트릭스

현재 아키텍처(`BaseStrategy` + `EventBus` + EDA) 호환성, 난이도, 예상 alpha, 독립성 기준:

### Tier 1 - 즉시 구현 가능 (기존 코드 확장)

| # | Strategy | 난이도 | 근거 |
|---|----------|--------|------|
| 1 | **1-A. Enhanced VW-TSMOM** | LOW | 기존 TSMOM에 volume weight 정교화. 1줄 수준 변경 |
| 2 | **4-B. Vol-Regime Adaptive Params** | LOW | 기존 전략 wrapper. Config 전환 로직만 추가 |

### Tier 2 - 신규 전략 모듈 (1-2일)

| # | Strategy | 난이도 | 근거 |
|---|----------|--------|------|
| 3 | **1-B. KAMA Trend Following** | LOW | 새 전략 모듈이지만 벡터화 가능, BaseStrategy 패턴 적합 |
| 4 | **2-A. Z-Score Mean Reversion** | LOW | 단순 통계 모델, 동적 lookback으로 기존 BB-RSI 대비 차별화 |
| 5 | **3-A. GK Volatility Breakout** | MED | GK 계산 단순, 기존 Breakout과 차별화된 entry logic |

### Tier 3 - 외부 연동 필요 (1주)

| # | Strategy | 난이도 | 근거 |
|---|----------|--------|------|
| 6 | **4-A. HMM Regime-Switching** | MED | `hmmlearn` 의존성, 학습 파이프라인 |
| 7 | **4-C. On-Chain Regime Filter** | MED | 외부 API(Glassnode) 연동 |
| 8 | **6-A. Funding Rate Arb** | MED | Binance perp API 연동 |
| 9 | **3-B. VRP Strategy** | MED | Deribit API 연동 |

### Tier 4 - 인프라 구축 필요 (2주+)

| # | Strategy | 난이도 | 근거 |
|---|----------|--------|------|
| 10 | **7-A. Regime-Specialist RF** | HIGH | ML pipeline + HMM + multi-model |
| 11 | **7-B. PPO Position Sizing** | HIGH | RL environment + stable-baselines3 |
| 12 | **5-A. VPIN** | HIGH | Tick data infra |
| 13 | **7-C. CNN-LSTM** | HIGH | DL pipeline, GPU 환경 |

---

## 전략 포트폴리오 조합 제안

상관관계가 낮은 전략을 조합하여 포트폴리오 Sharpe 극대화:

```
[Momentum]        TSMOM(Enhanced) + KAMA
    ×                                        → Low correlation
[Mean-Reversion]  Z-Score MR + BB-RSI(Enhanced)
    ×                                        → Regime-dependent allocation
[Regime]          HMM Switcher (meta-strategy)
    ×                                        → Macro overlay
[On-Chain]        MVRV/SOPR Filter (bias adjustment)
```

**권장 조합 (Phase 1)**:
1. Enhanced VW-TSMOM (모멘텀) + Vol-Regime Wrapper (적응형)
2. KAMA Trend Following (모멘텀 보완)
3. Z-Score Mean Reversion (역추세)
4. GK Volatility Breakout (변동성)

이 4개 전략은 **서로 다른 시장 조건**에서 수익을 창출하며, 모두 `BaseStrategy` 패턴으로 구현 가능합니다.

---

## References

### Momentum
- [Huang et al. (2024) - Cryptocurrency VW-TSMOM, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389)
- [Fieberg et al. - CTREND Factor, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4601972)
- [Moskowitz et al. (2012) - Time Series Momentum, JFE](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly)
- [Kaufman (1998) - KAMA](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama)

### Mean Reversion
- [Arda (2025) - BB under Varying Market Regimes: BTC/USDT, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962)
- [FMZ Quant (2025) - Multi-Factor Mean Reversion](https://www.fmz.com/lang/en/strategy/489893)

### Volatility
- [Garman & Klass (1980) - Estimators of Securities Price Volatility](https://doi.org/10.1086/296046)
- [Djanga et al. (2024) - Crypto Volatility Forecasting, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5048674)
- [Wiley (2025) - Pricing Crypto Options](https://onlinelibrary.wiley.com/doi/10.1002/fut.70029?af=R)

### Regime Detection
- [AJPAS (2025) - HMM in Bitcoin Regime Changes](https://doi.org/10.9734/ajpas/2025/v27i7781)
- [MDPI (2025) - Bitcoin Regime Shifts: Bayesian MCMC + HMM](https://www.mdpi.com/2227-7390/13/10/1577)
- [Springer (2024) - Regime Switching Forecasting for Crypto](https://link.springer.com/article/10.1007/s42521-024-00123-2)

### Microstructure
- [ScienceDirect (2025) - Bitcoin Order Flow Toxicity](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
- [ArXiv (2025) - Explainable Patterns in Crypto Microstructure](https://arxiv.org/html/2602.00776)

### DeFi / Derivatives
- [ScienceDirect (2025) - Funding Rate Arbitrage](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- [BIS Working Paper 1087 - Crypto Carry](https://www.bis.org/publ/work1087.pdf)
- [CME Group (2025) - Crypto Basis Trading](https://www.cmegroup.com/openmarkets/equity-index/2025/Spot-ETFs-Give-Rise-to-Crypto-Basis-Trading.html)

### ML / RL
- [Tandfonline (2025) - DQN for Bitcoin Trading](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2594873)
- [SSRN (2025) - Risk-Aware Deep RL for Crypto](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5662930)
- [Springer (2025) - ML for Crypto Trading Optimization](https://link.springer.com/article/10.1007/s44163-025-00519-y)

### On-Chain
- [Nansen - Onchain Metrics for Crypto Price Prediction](https://www.nansen.ai/post/onchain-metrics-key-indicators-for-cryptocurrency-price-prediction)
- [Mann - Quantitative Alpha in Crypto Markets, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5225612)
