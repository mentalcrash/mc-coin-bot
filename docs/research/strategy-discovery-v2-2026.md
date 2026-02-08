# Cryptocurrency Trading Strategy Discovery V2 (2026-02)

> 기존 전략(VW-TSMOM, BB-RSI, Donchian, Breakout 등)이 2024-2025 시장에서 Buy-and-Hold 대비 underperform한 원인을 분석하고,
> 멀티 타임프레임 + 혼합 전략 + 레짐 적응형 전략을 포괄하는 차세대 전략 후보를 정리한 문서.

---

## 0. 왜 기존 전략이 2024-2025에서 실패했는가

### 시장 구조 변화

| 요인 | 과거 Cycle (2017-2021) | 현재 Cycle (2024-2025) |
|------|----------------------|----------------------|
| 주도 세력 | Retail FOMO | **기관 (ETF AUM $164-179B)** |
| 상승 패턴 | 12개월 스프린트, pump & dump | 연 단위 점진적 축적 |
| 변동성 | 극히 높음 (BTC vol 80%+) | 감소 (BTC vol 25-50%) |
| Cycle 지표 | Pi Cycle Top, MVRV Z-Score 유효 | **Pi Cycle Top 실패, MVRV 신호 불명확** |
| 가격 범위 | BTC $3K → $69K | BTC $40K → $125K |

### 전략별 실패 원인

**Momentum (TSMOM) 실패:**
- 기관 매수 패턴이 점진적이라 momentum signal의 timing이 어긋남
- 변동성 감소 → signal-to-noise ratio 악화
- Bull market에서 short momentum signal의 반복적 손실

**Mean Reversion (BB-RSI) 실패:**
- 강한 상승 추세(BTC $40K→$125K)에서 근본적으로 불리
- 가격이 평균에서 이탈 후 더 멀어짐 → reversion 실패
- Range-bound 시장에서만 효과적이나 2024-2025는 대부분 trending

**Breakout/Donchian 실패:**
- 단일 lookback period 의존 → regime 변화에 취약
- False breakout 빈도 증가 (기관이 유동성 공급)

> **핵심 교훈**: Bitcoin은 "더 커지고 더 느려지고 있다." 고정 threshold 대신 dynamic signal, 단일 전략 대신 전략 조합, daily only 대신 multi-timeframe 접근이 필요.

### References
- [Why Bitcoin Price Top Indicators Failed](https://bitcoinmagazine.com/markets/why-bitcoin-price-top-indicators-failed)
- [Has Bitcoin's four-year cycle failed?](https://www.panewslab.com/en/articles/dfd2dcb9-60c9-4cec-b05b-f7a8c6ff0eb4)
- [Cryptocurrency momentum has (not) its moments](https://link.springer.com/article/10.1007/s11408-025-00474-9)

---

## 1. Trend-Following 강화 전략

### 1-A. Donchian Ensemble (Multi-Lookback Trend)

**다중 lookback period의 Donchian Channel을 앙상블로 통합**

- **Core Logic**: 서로 다른 lookback(5, 10, 20, 30, 60, 90, 150, 250, 360일)의 Donchian 시그널을 평균화하여 단일 unified signal 생성
- **Signal**:
  ```
  lookbacks = [5, 10, 20, 30, 60, 90, 150, 250, 360]
  signals = [donchian_breakout_signal(lb) for lb in lookbacks]
  unified_signal = mean(signals)
  position_size = target_vol / realized_vol
  ```
- **Timeframe**: Daily
- **L/S**: Long-Flat (rotational, top 20 coins)
- **Performance**: Sharpe > 1.5, BTC 대비 annualized alpha +10.8%, survivorship bias-free dataset
- **Crypto 적합성**: 다중 lookback 앙상블이 regime 변화에 robust. 단일 lookback의 whipsaw 문제 해결
- **구현 난이도**: **LOW** — 기존 DonchianStrategy 확장
- **Ref**: [Zarattini, Pagani, Barbon (2025) - "Catching Crypto Trends", SSRN 5209907](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)

---

### 1-B. Risk-Managed Momentum (Barroso-Santa-Clara Scaling)

**6개월 realized variance의 역수로 momentum signal 스케일링**

- **Core Logic**: 변동성이 높을 때 position 축소, 낮을 때 확대. Crypto에서는 equity와 달리 return augmentation 효과
- **Signal**:
  ```
  realized_var_6m = returns.rolling(126).var()
  scaling_weight = target_vol**2 / realized_var_6m
  scaled_signal = raw_momentum_signal * scaling_weight
  ```
- **Timeframe**: Weekly rebalancing
- **L/S**: Full Long/Short
- **Performance**: Weekly return 3.18%→3.47%, Sharpe 1.12→**1.42** (+27%), excess kurtosis 63.89→47.32
- **핵심 인사이트**: Crypto momentum은 equity와 달리 prolonged crash가 없음 → 평균 scaling weight **1.14** (equity 0.9). 즉, **risk-taking을 늘리는 것이 유리**
- **구현 난이도**: **LOW** — 기존 TSMOM vol-target 확장
- **Ref**: [Cryptocurrency market risk-managed momentum strategies (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377), [Cryptocurrency momentum has (not) its moments (Springer, 2025)](https://link.springer.com/article/10.1007/s11408-025-00474-9)

---

### 1-C. Multi-Timeframe D1H1 MACD Trend Following

**Daily MACD로 추세 판단, Hourly에서 추세 방향으로만 진입**

- **Core Logic**: Elder "Triple Screen" 기반. D1 MACD가 bullish일 때만 H1에서 long entry
- **Signal**:
  ```
  D1_trend = "bullish" if D1_MACD > D1_signal else "bearish"
  H1_entry = (H1_MACD crosses above H1_signal) AND (D1_trend == "bullish")
  exit = first_negative_hourly_bar (close < open)
  ```
- **Timeframe**: D1 (filter) + H1 (entry)
- **L/S**: Long-only (기본), Long/Short (확장)
- **Performance (BTC, 2018-2025)**:
  | Version | Sharpe | MDD | Annual Return |
  |---------|--------|-----|---------------|
  | Pure H1 MACD | 0.33 | -23.9% | 4.6% |
  | D1H1 Filter | 0.80 | -12.4% | 6.6% |
  | **D1H1 + Trail Stop** | **1.07** | **~-10%** | **~8%** |
- **구현 난이도**: **LOW-MEDIUM** — 기존 CandleAggregator + MTFFilter 활용
- **Ref**: [Mesicek, Vojtko (2025) - "Multi-Timeframe Trend Strategy on Bitcoin", QuantPedia/SSRN 5748642](https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/)

---

### 1-D. Volatility-Adaptive Trend Following

**Multi-horizon MA + RSI filter + ATR scaling의 결합**

- **Core Logic**: Multi-timeframe RSI로 momentum 확인, ATR scaling으로 position sizing, EMA slope reversal로 exit
- **Signal**:
  ```
  trend = EMA_crossover(fast=10, slow=50)
  momentum_confirm = RSI(14) > 50 (long) or RSI(14) < 50 (short)
  adx_filter = ADX(14) > threshold
  entry = trend AND momentum_confirm AND adx_filter
  position_size = target_vol / (ATR(14) / close)
  exit = EMA_slope_reversal OR ATR_trailing_stop
  ```
- **Timeframe**: 4H, 1D
- **L/S**: Full Long/Short
- **Performance**: BTC-ETH 간 rho=0.8-0.9으로 joint optimization 유효, 통계적/경제적으로 유의미한 excess returns
- **구현 난이도**: **MEDIUM** — 기존 TSMOM + ATR trailing stop에 RSI filter 추가
- **Ref**: [Karassavidis et al. (2025) - "Volatility-Adaptive Trend-Following", SSRN 5821842](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5821842)

---

## 2. 혼합 / 복합 전략

### 2-A. Momentum + Mean Reversion 50/50 Blend

**TSMOM과 BTC-Neutral Mean Reversion의 시그널 블렌딩**

- **Core Logic**: Momentum leg (28일 Z-Score 기반)과 Mean Reversion leg (BTC beta 제거 잔차)를 50/50 결합
- **Signal**:
  ```
  # Momentum leg
  mom_zscore = returns.rolling(28).sum() / returns.rolling(28).std()
  mom_signal = 1 if mom_zscore > upper_third else (-1 if mom_zscore < lower_third else 0)

  # Mean Reversion leg
  residual = alt_return - beta * btc_return
  mr_signal = -sign(residual) if abs(residual) > threshold else 0

  combined = 0.5 * mom_signal + 0.5 * mr_signal
  ```
- **Timeframe**: Daily
- **L/S**: Full Long/Short
- **Performance**: 개별 Sharpe: Mom ~1.0, MR ~2.3, **결합 Sharpe: 1.71, Annualized Return 56%, T-stat 4.07**
- **결합이 우수한 이유**: 직교(orthogonal) alpha sources — Momentum은 trending, MR은 choppy 시장에서 각각 강함
- **구현 난이도**: **LOW** — 기존 TSMOM + BB-RSI signal-level blending
- **Ref**: [Systematic Crypto Trading Strategies (Plotnik, 2024)](https://medium.com/@briplotnik/systematic-crypto-trading-strategies-momentum-mean-reversion-volatility-filtering-8d7da06d60ed)

---

### 2-B. Stochastic Momentum Hybrid

**Stochastic Oscillator + Trend Filter + ATR Position Sizing**

- **Core Logic**: Stochastic %K/%D crossover로 mean reversion entry, MA trend filter로 추세 방향만 거래, ATR로 position sizing
- **Signal**:
  ```
  stoch_k, stoch_d = stochastic(close, k=14, d=3, smooth=3)
  trend = close > SMA(close, 30)

  long_entry = (stoch_k crosses above stoch_d) AND trend
  alt_long = (stoch_k > 20) AND (stoch_k > stoch_d) AND trend  # oversold exit
  short_entry = (stoch_k crosses below stoch_d) AND NOT trend

  # ATR-based position sizing
  vol_ratio = ATR(14) / ATR(14).rolling(100).mean()
  position = 0.95 if vol_ratio < 0.5 else (0.30 if vol_ratio > 2.0 else interpolate)
  exit = ATR_trailing_stop(multiplier=1.0)
  ```
- **Timeframe**: 4H, 1D
- **L/S**: Full Long/Short
- **Performance**: Stochastic(mean reversion) + MA(trend) + ATR(risk)의 세 가지 독립 차원 결합
- **구현 난이도**: **LOW** — BaseStrategy 표준 패턴으로 구현 가능
- **Ref**: [Stochastic Momentum Strategy (PyQuantLab)](https://pyquantlab.com/article.php?file=Stochastic+Momentum+Strategy+A+Trend-Following+and+Mean-Reversion+Hybrid.html)

---

### 2-C. MAX/MIN Combined Strategy

**Trend-Following(MAX) + Counter-Trend(MIN) 동시 운용**

- **Core Logic**: MAX = N일 신고가 시 매수(trend-following), MIN = N일 신저가 시 매수(mean-reversion). 두 전략 50/50 동시 운용
- **Signal**:
  ```
  max_signal = 1 if close == close.rolling(lookback).max() else 0
  min_signal = 1 if close == close.rolling(lookback).min() else 0
  combined = 0.5 * max_signal + 0.5 * min_signal
  ```
- **Timeframe**: Daily (lookback=10일 최적)
- **L/S**: Long-only
- **Performance**: 결합 시 Buy & Hold 대비 높은 수익률 + 낮은 drawdown. Bear market에서 MAX가 더 robust
- **구현 난이도**: **VERY LOW** — Donchian Channel 변형
- **Ref**: [Revisiting Trend-following and Mean-reversion in Bitcoin (QuantPedia)](https://quantpedia.com/revisiting-trend-following-and-mean-reversion-strategies-in-bitcoin/)

---

### 2-D. RSI Divergence + Multi-TF Trend

**상위 TF 추세와 일치하는 RSI Divergence만 거래**

- **Core Logic**: Weekly/Daily 추세 방향을 확인 후, 4H/1H에서 추세 방향과 일치하는 RSI divergence만 진입
- **Signal**:
  ```
  # Higher TF trend (Weekly)
  weekly_trend = close_weekly > SMA(close_weekly, 50)

  # Lower TF divergence (4H)
  bullish_div = (price_makes_lower_low) AND (RSI_makes_higher_low)
  bearish_div = (price_makes_higher_high) AND (RSI_makes_lower_high)
  hidden_bull = (price_makes_higher_low) AND (RSI_makes_lower_low)  # 추세 지속

  long_entry = weekly_trend AND (bullish_div OR hidden_bull)
  short_entry = NOT weekly_trend AND bearish_div
  ```
- **Timeframe**: Weekly (trend) + 4H (entry)
- **L/S**: Full Long/Short
- **Performance**: General divergence는 추세 전환, Hidden divergence는 추세 지속 포착. MTF 필터로 false signal 대폭 감소
- **구현 난이도**: **MEDIUM-HIGH** — Divergence detection algorithm (pivot point 기반) 구현 필요
- **Ref**: [Trading RSI and RSI Divergence (altFINS)](https://altfins.com/knowledge-base/trading-rsi-and-rsi-divergence/)

---

## 3. 인트라데이 / 단기 타임프레임 전략

### 3-A. Bitcoin Overnight Seasonality (22:00 UTC)

**22:00-23:00 UTC의 통계적으로 유의미한 양의 수익률 활용**

- **Core Logic**: 매일 22:00 UTC에 Long 진입, 00:00 UTC에 청산. 하루 2시간만 포지션 보유
- **Signal**:
  ```
  if current_hour == 21:  # UTC
      position = LONG
  elif current_hour == 23:
      position = FLAT
  # Enhancement: Volatility filter
  if rolling_vol > vol_threshold:
      position_size *= 1.5  # 고변동성 일에 더 큰 수익
  ```
- **Timeframe**: 1H
- **L/S**: Long-only
- **Performance**: 연 **40.64%** 수익률, Calmar 1.79. Vol filter 적용 시 Calmar **1.97**, MDD -18.87%
- **핵심 장점**: 하루 22시간은 현금 보유 → 자본 효율성 극대화, 시장 노출 최소
- **구현 난이도**: **VERY LOW** — 시간 기반 단순 로직
- **Ref**: [Overnight Seasonality in Bitcoin (QuantPedia)](https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin)

---

### 3-B. Larry Williams Volatility Breakout

**전일 Range의 k-factor를 당일 시가에 더해 Entry level 설정**

- **Core Logic**: Intraday trend persistence를 활용. 당일 시가 + k × 전일 Range를 돌파하면 진입, 00:00 UTC에 청산
- **Signal**:
  ```
  range = prev_high - prev_low
  entry_long = today_open + k * range   # k = 0.5~0.6
  entry_short = today_open - k * range

  if close > entry_long: position = LONG
  elif close < entry_short: position = SHORT

  stop_loss = (prev_low + entry_long) / 2  # or 2% fixed
  exit = next_day_open (00:00 UTC)
  ```
- **Timeframe**: 5M-1H (monitoring), Daily (range 기준)
- **L/S**: Full Long/Short
- **Performance**: ETH-PERP 2020: 113 trades, **304.7% return**, **MDD -7.23%** (B&H는 -61.45%)
- **핵심 장점**: Overnight risk 제거, 매우 낮은 MDD (7.23%)
- **구현 난이도**: **LOW** — 기존 Breakout 전략 유사 구조
- **Ref**: [Simple Crypto Breakout Strategy (GitHub)](https://github.com/SC4RECOIN/simple-crypto-breakout-strategy)

---

### 3-C. RSI Crossover Mean Reversion (4H)

**4H RSI 30/70 crossover가 crypto에서 특히 효과적**

- **Core Logic**: RSI가 30을 상향 돌파 시 Long, 70을 하향 돌파 시 Short
- **Signal**:
  ```
  rsi = RSI(close_4h, period=14)

  long_entry = (rsi > 30) AND (rsi.shift(1) <= 30)   # crosses above 30
  short_entry = (rsi < 70) AND (rsi.shift(1) >= 70)   # crosses below 70

  exit_long = rsi > 60  # mean zone
  exit_short = rsi < 40
  ```
- **Timeframe**: **4H** (최적, 1H 대비 Sharpe 2배)
- **L/S**: Full Long/Short
- **Performance**: 4H 기준 25 trades, 60% win rate, **Sharpe 5.13** (1H는 Sharpe 2.61)
- **핵심 장점**: 일봉 RSI는 overbought가 오래 지속되지만 4H에서는 micro-trend 반전 신호가 정확
- **구현 난이도**: **VERY LOW** — 기존 BB-RSI의 RSI 부분 분리
- **Ref**: [RSI Crossover on Bitcoin 4H (Medium)](https://medium.com/@AtomicScript/episode-3-rsi-crossover-strategy-1273d8b3f290)

---

### 3-D. TTM Squeeze (Volatility Compression Breakout)

**Bollinger Bands가 Keltner Channels 안으로 수축 후 breakout 진입**

- **Core Logic**: BB가 KC 안으로 들어가면 "squeeze" 상태. Squeeze 해제 시 momentum 방향으로 진입
- **Signal**:
  ```
  bb_upper = SMA(20) + 2.0 * std(20)
  bb_lower = SMA(20) - 2.0 * std(20)
  kc_upper = EMA(20) + 1.5 * ATR(20)
  kc_lower = EMA(20) - 1.5 * ATR(20)

  squeeze_on = (bb_upper < kc_upper) AND (bb_lower > kc_lower)
  squeeze_off = NOT squeeze_on AND squeeze_on.shift(1)

  momentum = linear_regression_slope(close, 20)  # or MACD histogram

  long_entry = squeeze_off AND (momentum > 0)
  short_entry = squeeze_off AND (momentum < 0)
  exit = close < SMA(21) (long) or close > SMA(21) (short)
  trailing_stop = ATR(7) * 3.0
  ```
- **Timeframe**: 1H-4H (최적, MDD ~12%)
- **L/S**: Full Long/Short
- **Performance**: 1H-4H에서 MDD 약 12%, squeeze→breakout 패턴이 crypto에서 빈번
- **구현 난이도**: **MEDIUM** — BB + KC + Momentum 조합, 기존 BB 코드 부분 활용
- **Ref**: [TTM Squeeze Quantitative Implementation (FMZQuant)](https://medium.com/@FMZQuant/volatility-compression-momentum-breakout-tracking-strategy-quantitative-implementation-of-ttm-dfe0232ccf51)

---

### 3-E. Session Breakout (Asian Range Breakout)

**Asian session 고/저를 Range로 설정, EU/US session 시작 시 돌파 진입**

- **Core Logic**: Asian session(00:00-09:00 UTC)의 High/Low 기록 후, EU/US session에서 breakout
- **Signal**:
  ```
  asian_high = max(high[00:00-09:00 UTC])
  asian_low = min(low[00:00-09:00 UTC])

  long_entry = close > asian_high  # during EU/US session
  short_entry = close < asian_low

  stop_loss = (asian_high + asian_low) / 2  # range midpoint
  take_profit = entry + 1.5 * (entry - stop_loss)
  exit = 00:00 UTC  # daily close
  ```
- **Timeframe**: 1H (entry), Daily (cycle)
- **L/S**: Full Long/Short
- **Performance**: EU-US overlap(12:00-16:00 UTC)에서 30-40% 높은 거래량 → breakout 성공률 향상
- **구현 난이도**: **MEDIUM** — 시간대 필터링 로직 필요
- **Ref**: [Asian Session Breakout Strategy (FMZQuant)](https://medium.com/@FMZQuant/breakthrough-strategy-for-high-and-low-points-in-the-asian-market-9e2b928b683c)

---

### 3-F. Intraday TSMOM (Half-Hour Predictability)

**Bitcoin 첫 30분 수익률이 마지막 30분 수익률을 예측**

- **Core Logic**: 하루를 48개 half-hour 구간으로 분할, 첫 구간과 두 번째 마지막 구간의 수익률로 마지막 구간 방향 예측
- **Signal**:
  ```
  signal_1 = first_30min_return
  signal_2 = second_last_30min_return
  combined = alpha * signal_1 + beta * signal_2

  if combined > threshold: long_last_30min
  elif combined < -threshold: short_last_30min
  ```
- **Timeframe**: 30min intervals (실제 구현 시 1H 적용 가능)
- **L/S**: Full Long/Short
- **Performance**: 첫 30min signal: 연 7.82%. 두 signal 결합: 연 **16.69%**, 하락장에서 특히 효과적
- **구현 난이도**: **LOW** — 기존 TSMOM 로직 재활용
- **Ref**: [Shen, Urquhart, Wang (2022) - "Bitcoin Intraday TSMOM"](https://centaur.reading.ac.uk/100181/)

---

## 4. 레짐 탐지 / 적응형 전략

### 4-A. ADX Regime Filter (Rule-Based)

**ADX로 trending/ranging 분류 → 전략 자동 전환**

- **Core Logic**: ADX 값으로 시장 상태 분류, 각 상태에 맞는 전략 적용. Crypto 전용 threshold: **ADX 15** (전통 시장 25보다 낮음)
- **Signal**:
  ```
  adx = ADX(high, low, close, period=14)
  atr = ATR(high, low, close, period=14)

  if adx < 15:                          # Strong ranging
      strategy = None                    # 비활성화 or MR only
  elif adx >= 15 and adx < 25:          # Weak trend
      strategy = blend(0.5 * momentum, 0.5 * mean_reversion)
  elif adx >= 25:                        # Confirmed trend
      strategy = trend_following         # TSMOM, Donchian, Breakout

  # ADX + ATR 조합
  if adx_rising and atr_high:           # 강한 변동성 트렌드
      exposure = 1.0
  elif adx_falling and atr_high:        # 방향 불명, 변동성 높음
      exposure = 0.3                    # hedge or cash
  elif adx_rising and atr_low:          # 조용한 트렌드
      exposure = 0.7                    # tight stops
  elif adx_falling and atr_low:         # 완전한 ranging
      strategy = mean_reversion
  ```
- **Timeframe**: Daily (regime), 전략 TF (execution)
- **L/S**: Regime 의존
- **Performance**: Trend-following이 ADX>25에서만 적용되면 false signal 대폭 감소. Ranging 시장 손실 회피
- **구현 난이도**: **LOW** — ADX 계산 + threshold 조건만 추가
- **Ref**: [BingX - ADX in Crypto Trading](https://bingx.com/en/learn/article/how-to-use-adx-indicator-in-crypto-trading)

---

### 4-B. Realized Volatility Structure Regime (Banerjee 2025)

**Short-term vs Long-term volatility 구조 + Normalized Momentum으로 3 regime 분류**

- **Core Logic**: 단기/장기 vol의 구조적 관계와 방향성 momentum으로 regime 분류
- **Signal**:
  ```
  vol_short = returns.rolling(10).std()
  vol_long = returns.rolling(60).std()
  vol_ratio = vol_short / vol_long
  norm_momentum = returns.rolling(20).sum() / returns.rolling(20).std()

  if vol_ratio > 1.2 and abs(norm_momentum) > 1.5:
      regime = "EXPANSION"     # 높은 변동성 + 강한 방향성
  elif vol_ratio < 0.8 and abs(norm_momentum) < 0.5:
      regime = "CONTRACTION"   # 낮은 변동성 + 약한 방향성
  else:
      regime = "NEUTRAL"
  ```
- **Regime별 전략**:
  | Regime | 전략 | 파라미터 |
  |--------|------|---------|
  | Expansion | TSMOM, Breakout | 적극적 position, loose stops |
  | Neutral | Standard momentum | 중간 exposure |
  | Contraction | BB-RSI, Z-Score MR | 작은 position, tight stops |
- **Timeframe**: Daily (regime), 4H/1H (execution)
- **구현 난이도**: **MEDIUM** — 기존 VolRegimeStrategy 확장
- **Ref**: [Banerjee (2025) - "Detecting Volatility Regimes in Crypto", SSRN 5920642](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5920642)

---

### 4-C. Hurst Exponent / Efficiency Ratio Regime

**시장의 trending/mean-reverting 성향을 정량적으로 분류**

- **Core Logic**: Hurst Exponent(H)로 persistent/anti-persistent 분류, Efficiency Ratio(ER)로 가격 효율성 측정
- **Signal**:
  ```
  # Efficiency Ratio (Kaufman)
  direction = abs(close - close.shift(lookback))
  volatility = abs(close.diff()).rolling(lookback).sum()
  ER = direction / volatility

  # Simplified Hurst proxy
  # H > 0.5: trending, H < 0.5: mean-reverting
  hurst_proxy = rolling_hurst(returns, window=100)

  if ER > 0.6 or hurst_proxy > 0.55:
      strategy = TREND_FOLLOWING
  elif ER < 0.3 or hurst_proxy < 0.45:
      strategy = MEAN_REVERSION
  else:
      strategy = BLEND_50_50
  ```
- **Timeframe**: Daily
- **L/S**: Regime 의존
- **Performance**: BTC 전체 평균 Hurst ~0.32 (anti-persistent 경향) → 장기적으로 mean reversion이 우세, 단 regime별로 크게 변동
- **구현 난이도**: **MEDIUM** — Rolling Hurst 계산 필요 (R/S analysis 또는 DFA)
- **Ref**: [Fractal Geometry & Bitcoin (MDPI, 2023)](https://www.mdpi.com/2504-3110/7/12/870)

---

### 4-D. HMM (Hidden Markov Model) Regime Switching

**GaussianHMM으로 Bull/Bear/Sideways 3-state 자동 분류**

- **Core Logic**: Log returns + rolling volatility를 feature로 HMM 학습. 각 regime에 최적 전략 적용
- **Signal**:
  ```python
  from hmmlearn.hmm import GaussianHMM

  features = np.column_stack([log_returns, rolling_vol])
  model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
  model.fit(features)  # expanding window, no look-ahead
  regime = model.predict(features)[-1]

  strategy_map = {
      BULL:     TsmomStrategy(vol_target=0.30),
      SIDEWAYS: BbRsiStrategy(bb_std=2.0),
      BEAR:     None  # cash or hedge
  }
  ```
- **Timeframe**: Daily (regime detection), 4H/1H (execution)
- **L/S**: Regime 의존
- **Performance**: 학술 연구에서 bull/bear/neutral 전환 유의미 예측. Static allocation 대비 alpha 생성
- **주의사항**: Look-ahead bias 방지(expanding window 필수), transition buffer (전환 확인 기간) 필요, Regime 수 3개가 실용적 최대
- **구현 난이도**: **MEDIUM-HIGH** — `hmmlearn` 의존성, 학습 파이프라인
- **Ref**: [AJPAS (2025) - "HMM in Bitcoin Regime Changes"](https://doi.org/10.9734/ajpas/2025/v27i7781), [PyQuantLab - Market Regime Detection using HMMs](https://www.pyquantlab.com/articles/Market%20Regime%20Detection%20using%20Hidden%20Markov%20Models.html)

---

### 4-E. Bollinger Bands Regime-Dependent Strategy Selection

**시장 Phase에 따른 BB 전략 최적화 (학술 검증)**

- **Core Logic**: 시장 phase(Bull/Bear/Accumulation)별로 BB 전략의 최적 variant가 다름
- **Signal 적용**:
  | Market Phase | 최적 전략 | 성과 |
  |-------------|----------|------|
  | **Bear (2018)** | BB Breakout | 우수 (vol expansion 포착) |
  | **Accumulation (2018-2020)** | BB Breakout | 양호 (risk-adjusted 우수) |
  | **Bull (2020-2021)** | BB Mean Reversion + Breakout | 양쪽 모두 우수 |
- **핵심**: Bear market에서 Mean Reversion은 실패하고 Breakout이 생존
- **구현 난이도**: **LOW** — 기존 BB-RSI + regime filter 결합
- **Ref**: [Arda (2025) - "Bollinger Bands under Varying Market Regimes: BTC/USDT", SSRN 5775962](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962)

---

## 5. 새로운 Alpha Sources (구조적 수익원)

### 5-A. Bitcoin ETF Flow Strategy

**ETF 자금 유입/유출 데이터로 기관 행동 추종**

- **Core Logic**: Daily ETF net flow가 양수면 BTC long, 음수면 cash
- **Signal**:
  ```
  daily_etf_flow = fetch_etf_net_flow()  # Farside Investors, CoinGlass

  if daily_etf_flow > 0:
      position = LONG
  else:
      position = CASH
  ```
- **Timeframe**: Daily
- **L/S**: Long-Flat
- **Performance**: **118.5% return** vs 81.7% B&H (2024.01-2025.03). 약 40% outperformance
- **핵심**: 절대적 top/bottom을 잡는 것이 아닌, steep loss 회피의 compounding effect
- **데이터 소스**: [Farside Investors](https://farside.co.uk/btc/), [CoinGlass](https://www.coinglass.com/etf/bitcoin), [Bitbo](https://bitbo.io/treasuries/etf-flows/)
- **구현 난이도**: **LOW** — CSV/RSS 파싱으로 시작 가능
- **Ref**: [Bitcoin Magazine Pro - ETF Flow Strategy Beats B&H by 40%](https://bitcoinmagazine.com/markets/bitcoin-etf-flow-strategy-beats-buy-and-hold)

---

### 5-B. Funding Rate Carry (Delta-Neutral)

**Spot Long + Perpetual Short로 funding rate 수취**

- **Core Logic**: Positive funding rate 환경에서 spot long + perp short → delta-neutral carry
- **Signal**:
  ```
  funding_rate = get_predicted_funding_rate(symbol)  # 8h settlement

  if funding_rate > entry_threshold:      # e.g., 0.015%
      action = "spot_long + perp_short"   # funding 수취
  elif funding_rate < exit_threshold:     # e.g., 0.005%
      action = "close_all"
  ```
- **Timeframe**: 8H (funding settlement)
- **L/S**: Delta-neutral
- **Performance**: Full sample 연 38%, **Sharpe 4+** (2020-2024). **2025년: 수익성 감소 (연 5-10%)**
- **주의**: ETF 도입 후 carry spread 약 3pp 감소. 2025년 C-4 factor model에서 Sharpe 음수 전환 보고 → **순수 carry보다 momentum과 결합이 robust**
- **구현 난이도**: **MEDIUM** — Binance API, spot+futures 동시 관리
- **Ref**: [BIS Working Paper 1087 - Crypto Carry](https://www.bis.org/publ/work1087.pdf), [Funding Rate Arbitrage (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S2096720925000818)

---

### 5-C. Optimized Cointegration Pairs Trading

**Cointegrated crypto pair의 spread Z-score 기반 거래**

- **Core Logic**: Top 10 crypto에서 cointegrated pair 식별, spread의 Z-score로 진입/청산, dynamic trailing stop
- **Signal**:
  ```python
  from statsmodels.tsa.stattools import coint

  # Pair selection (90 pairs → ~37 cointegrated)
  for pair in all_crypto_pairs:
      _, pvalue, _ = coint(price_a, price_b)
      if pvalue < 0.05: selected.append(pair)

  # Spread Z-score
  spread = price_a - hedge_ratio * price_b
  z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

  if z > 2.0:   short_a, long_b    # spread 수축 기대
  elif z < -2.0: long_a, short_b
  elif abs(z) < 0.5: close_all     # mean reversion 완료
  ```
- **Timeframe**: Daily
- **L/S**: Market-neutral (long-short pairs)
- **Performance**: 연 **71%**, Sharpe ~2.0, **MDD 14%** (2019-2024). Bull + Bear 모두 positive
- **구현 난이도**: **MEDIUM-HIGH** — Cointegration test, dynamic optimization, 새 전략 패턴 필요
- **Ref**: [Palazzi (2025) - "Trading Games: Beating Passive Strategies", J. Futures Markets](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018)

---

### 5-D. Altcoin Sector Rotation

**BTC Dominance 변화에 따른 알트코인 sector rotation**

- **Core Logic**: BTC.D 하락 시 altcoin으로 자금 이동. Capital rotation stages를 따라 순차 진입
- **Signal**:
  ```
  btc_d = get_btc_dominance()
  altseason_idx = get_altseason_index()  # Top 50 중 BTC outperform %

  # Capital Rotation Stages
  if btc_d > 60 and btc_d_rising:
      stage = 1  # BTC 주도 → BTC only
  elif altseason_idx > 25 and btc_d_falling:
      stage = 2  # ETH + Large-cap 활성화
  elif altseason_idx > 50:
      stage = 3  # Mid-cap, sector leaders
  elif altseason_idx > 75:
      stage = 4  # Small-cap (위험 최대)

  # 2024-2025 Hot Sectors
  sectors = {
      "RWA": ["ONDO", "SKY"],        # 평균 15x gains
      "DeFi": ["AAVE", "UNI"],       # TVL $63B ATH
      "AI": ["RENDER", "FET"],       # 새 narrative
  }
  ```
- **Timeframe**: Weekly/Daily
- **L/S**: Long-only (rotation)
- **Performance**: RWA sector 평균 15x gains (2024-2025), ETH +23%, SOL +31% vs BTC
- **구현 난이도**: **MEDIUM** — BTC.D + sector data 필요
- **Ref**: [Altcoin Rotation Q4 2025 (AInvest)](https://www.ainvest.com/news/timing-altcoin-rotation-strategic-entry-points-q4-2025-2512/)

---

## 6. Cross-Sectional Factor 전략

### 6-A. CTREND Factor (ML-based Multi-Indicator Trend)

**28개 technical indicator를 ML로 통합한 cross-sectional trend factor**

- **Core Logic**: 3,000+ coins에서 28개 technical signal을 Elastic Net으로 aggregate
- **Signal**:
  ```
  indicators = [
      bollinger_mid, CCI, MACD, RSI,  # 가장 중요
      SMA(10), SMA(20), EMA(10), EMA(20),
      volume_ratio, ATR, OBV,
      ... (28개 총)
  ]
  # Multiple horizons: 1d, 1w, 2w, 1m
  features = flatten([calc(ind, horizon) for ind in indicators for horizon in horizons])
  CTREND_score = elastic_net.predict(features)

  # Long-Short portfolio
  long = coins with top decile CTREND_score
  short = coins with bottom decile CTREND_score
  ```
- **Timeframe**: Weekly rebalancing
- **L/S**: Long-Short
- **Performance**: Weekly long-short alpha **2.62%** (t-stat 4.22), known factors에 subsumed 안됨, transaction cost 후에도 유의미, big & liquid coins에서도 지속
- **구현 난이도**: **HIGH** — ML pipeline, 28개 indicator, multi-coin universe
- **Ref**: [Fieberg et al. (2025) - "CTREND Factor", JFQA Vol.60](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178)

---

### 6-B. Cross-Sectional Momentum (Weekly 2-Week Returns)

**주간 cross-sectional ranking + inverse volatility weighting**

- **Core Logic**: N개 coin의 2주 수익률로 ranking, top quintile long / bottom quintile short
- **Signal**:
  ```
  returns_2w = prices.pct_change(14)
  rankings = returns_2w.rank(ascending=False)

  long_coins = rankings[rankings <= top_20_pct].index
  short_coins = rankings[rankings >= bottom_20_pct].index

  # Inverse volatility weighting
  vol = returns.rolling(30).std()
  weights = (1 / vol) / (1 / vol).sum()
  ```
- **Timeframe**: Weekly rebalancing
- **L/S**: Long-Short
- **Performance**: Weekly alpha +2.6%, t-stat 3.89 (full sample), +2.1% (post-2020)
- **Weekend effect**: 주말 momentum return이 주중보다 높음 (altcoin에서 더 강함)
- **구현 난이도**: **MEDIUM** — Multi-coin ranking, 기존 run_multi() 확장
- **Ref**: [arXiv 2510.14435 - C-4 Factor Model](https://arxiv.org/abs/2510.14435), [Weekend Effect in Crypto Momentum (ACR Journal)](https://acr-journal.com/article/the-weekend-effect-in-crypto-momentum-does-momentum-change-when-markets-never-sleep--1514/)

---

### 6-C. CGA-Agent RSI Crossover (Genetic Algorithm 최적화)

**Dual RSI crossover + MA filter를 Genetic Algorithm으로 파라미터 최적화**

- **Core Logic**: RSI-fast / RSI-slow crossover에 MA filter + ATR slope filter를 GA로 최적화
- **Performance (5분봉, 2024.12-2025.09)**:
  | Asset | PnL | Sharpe | Sortino |
  |-------|-----|--------|---------|
  | BTC | +2.17% | 1.26 | 2.51 |
  | ETH | +4.16% | **2.09** | **4.11** |
  | BNB | +9.27% | **2.99** | **6.55** |
- **구현 난이도**: **MEDIUM** — GA 최적화 파이프라인 필요
- **Ref**: [CGA-Agent (arXiv 2510.07943)](https://arxiv.org/abs/2510.07943)

---

## 7. 구현 우선순위 매트릭스

### 기준
1. **Evidence 강도**: 학술 논문 + 실제 백테스트 결과
2. **구현 난이도**: 기존 프로젝트 인프라 활용도
3. **Alpha 독립성**: 기존 전략과의 상관관계 (낮을수록 좋음)
4. **타임프레임 다양성**: Daily에 편중되지 않는지

---

### Tier 1 — 즉시 구현 (기존 코드 확장, 1-2일)

| # | Strategy | TF | Sharpe | 근거 |
|---|----------|------|--------|------|
| 1 | **Overnight Seasonality** (3-A) | 1H | Calmar 1.97 | 하루 2시간 노출, 연 40% |
| 2 | **RSI Crossover 4H** (3-C) | 4H | **5.13** | BB-RSI 재활용, 최고 Sharpe |
| 3 | **Risk-Managed Momentum** (1-B) | D1/W | **1.42** | 기존 vol-target 확장, 학술 강력 |
| 4 | **MAX/MIN Combined** (2-C) | D1 | >B&H | Donchian 변형, 극히 단순 |
| 5 | **ADX Regime Filter** (4-A) | D1 | - | 기존 전략에 filter 추가 |

### Tier 2 — 신규 전략 모듈 (2-5일)

| # | Strategy | TF | Sharpe | 근거 |
|---|----------|------|--------|------|
| 6 | **Donchian Ensemble** (1-A) | D1 | **>1.5** | SSRN 논문, alpha +10.8% |
| 7 | **Larry Williams VB** (3-B) | Intraday | - | MDD 7.23%, 검증된 전략 |
| 8 | **MTF D1H1 MACD** (1-C) | D1+H1 | **1.07** | QuantPedia 검증, MDD -10% |
| 9 | **Stochastic Momentum** (2-B) | 4H/D1 | - | 3차원 결합, BaseStrategy 적합 |
| 10 | **Mom+MR 50/50 Blend** (2-A) | D1 | **1.71** | T-stat 4.07, 연 56% |
| 11 | **TTM Squeeze** (3-D) | 1H-4H | - | BB 코드 재활용, MDD 12% |
| 12 | **ETF Flow Strategy** (5-A) | D1 | - | B&H +40% outperform |

### Tier 3 — 고급 구현 (1-2주)

| # | Strategy | TF | Sharpe | 근거 |
|---|----------|------|--------|------|
| 13 | **Vol Structure Regime** (4-B) | D1 | - | SSRN 2025, VolRegime 확장 |
| 14 | **Hurst/ER Regime** (4-C) | D1 | - | Trending/MR 정량 분류 |
| 15 | **Pairs Trading** (5-C) | D1 | **~2.0** | 연 71%, MDD 14%, market-neutral |
| 16 | **Vol-Adaptive Trend** (1-D) | 4H/D1 | - | SSRN 2025, RSI filter |
| 17 | **XS Momentum** (6-B) | W | - | Factor model 기반 |
| 18 | **Funding Rate Carry** (5-B) | 8H | 4+ (과거) | Delta-neutral, 수익성 감소 중 |
| 19 | **HMM Regime** (4-D) | D1 | - | hmmlearn 의존성 |

### Tier 4 — 인프라 구축 필요 (2주+)

| # | Strategy | TF | Sharpe | 근거 |
|---|----------|------|--------|------|
| 20 | **CTREND Factor** (6-A) | W | - | ML pipeline, 3000+ coins |
| 21 | **CGA-Agent RSI** (6-C) | 5M | 2.09-2.99 | GA 최적화 파이프라인 |
| 22 | **Altcoin Rotation** (5-D) | W/D1 | - | Sector data 필요 |
| 23 | **RSI Divergence+MTF** (2-D) | W+4H | - | Divergence detector 구현 |

---

## 8. 권장 포트폴리오 조합

### 상관관계가 낮은 전략 결합으로 Sharpe 극대화

```
[Trend-Following]    Donchian Ensemble + Risk-Managed Momentum
    ×                                                         → Low correlation
[Mean-Reversion]     RSI Crossover 4H + Stochastic Momentum
    ×                                                         → Timeframe diversification
[Intraday]           Overnight Seasonality + Larry Williams VB
    ×                                                         → Regime overlay
[Regime]             ADX Filter → strategy selection
    ×                                                         → New alpha source
[Structural]         ETF Flow (directional) + Pairs Trading (neutral)
```

### 최적 3-Strategy Portfolio (Phase 1 권장)

1. **Risk-Managed TSMOM** (Daily, trend-following)
   - Sharpe 1.42, 학술 검증 최고
2. **RSI Crossover 4H** (4H, mean-reversion)
   - Sharpe 5.13, Daily와 다른 TF
3. **ADX Regime Filter** (meta-strategy)
   - ADX<15 시 MR 전환, ADX>25 시 Momentum 전환

### 확장 5-Strategy Portfolio (Phase 2 권장)

4. **Donchian Ensemble** (Daily, ensemble trend)
   - Multi-lookback으로 단일 TSMOM 보완
5. **Overnight Seasonality** (1H, time-based)
   - 완전히 독립적 alpha source

---

## 9. 핵심 인사이트 요약

1. **Carry 전략 사망 경고**: Full sample Sharpe 6.45 → 2025년 음수. Market maturation의 증거
2. **Momentum은 살아있다**: Vol-scaling으로 Sharpe 1.12→1.42. Crash risk 없이 return augment
3. **Multi-lookback ensemble이 단일 lookback 우월**: Donchian ensemble Sharpe >1.5
4. **4H가 Daily보다 RSI 전략에 유리**: Sharpe 5.13 vs 2.61 (2배 차이)
5. **Overnight seasonality는 free lunch에 가까움**: 하루 2시간, 연 40%, Calmar 1.97
6. **ADX threshold 15가 crypto 최적**: 전통 시장의 25보다 낮음
7. **ETF flow는 새로운 alpha source**: B&H +40% outperform (2024-2025)
8. **Regime filter가 모든 전략의 성능을 개선**: Bear에서 MR 실패, Breakout 생존 (학술 검증)
9. **Anomaly는 시간 경과에 따라 감소**: Simple, robust 전략이 복잡한 전략보다 지속 가능
10. **Cross-sectional > Time-series**: CTREND, factor model 등 coin selection이 추가 alpha

---

## References (전체)

### Trend-Following
- [Zarattini et al. (2025) - Catching Crypto Trends, SSRN 5209907](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907)
- [Karassavidis et al. (2025) - Volatility-Adaptive Trend-Following, SSRN 5821842](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5821842)
- [Mesicek, Vojtko (2025) - Multi-TF Bitcoin Strategy, QuantPedia/SSRN 5748642](https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/)
- [Fieberg et al. (2025) - CTREND Factor, JFQA Vol.60](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178)

### Momentum & Risk Management
- [Risk-Managed Momentum (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377)
- [Cryptocurrency momentum has (not) its moments (Springer, 2025)](https://link.springer.com/article/10.1007/s11408-025-00474-9)
- [Stop-loss rules and momentum payoffs (JBEF, 2023)](https://www.sciencedirect.com/science/article/abs/pii/S2214635023000473)
- [Weekend Effect in Crypto Momentum (ACR Journal)](https://acr-journal.com/article/the-weekend-effect-in-crypto-momentum-does-momentum-change-when-markets-never-sleep--1514/)

### Hybrid / Composite
- [Systematic Crypto Trading Strategies (Plotnik, 2024)](https://medium.com/@briplotnik/systematic-crypto-trading-strategies-momentum-mean-reversion-volatility-filtering-8d7da06d60ed)
- [Stochastic Momentum Strategy (PyQuantLab)](https://pyquantlab.com/article.php?file=Stochastic+Momentum+Strategy+A+Trend-Following+and+Mean-Reversion+Hybrid.html)
- [Revisiting Trend-following and Mean-reversion in Bitcoin (QuantPedia)](https://quantpedia.com/revisiting-trend-following-and-mean-reversion-strategies-in-bitcoin/)

### Intraday
- [Overnight Seasonality in Bitcoin (QuantPedia)](https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin)
- [Shen, Urquhart, Wang (2022) - Bitcoin Intraday TSMOM](https://centaur.reading.ac.uk/100181/)
- [TTM Squeeze Implementation (FMZQuant)](https://medium.com/@FMZQuant/volatility-compression-momentum-breakout-tracking-strategy-quantitative-implementation-of-ttm-dfe0232ccf51)
- [RSI Crossover on Bitcoin 4H](https://medium.com/@AtomicScript/episode-3-rsi-crossover-strategy-1273d8b3f290)
- [Simple Crypto Breakout Strategy (GitHub)](https://github.com/SC4RECOIN/simple-crypto-breakout-strategy)

### Regime Detection
- [Banerjee (2025) - Detecting Volatility Regimes, SSRN 5920642](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5920642)
- [Arda (2025) - BB under Varying Market Regimes, SSRN 5775962](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962)
- [Two Sigma - ML Approach to Regime Modeling](https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/)
- [Fidelity - Bitcoin Price Phases](https://www.fidelitydigitalassets.com/research-and-insights/bitcoin-price-phases-navigating-bitcoins-volatility-trends)
- [BingX - ADX in Crypto Trading](https://bingx.com/en/learn/article/how-to-use-adx-indicator-in-crypto-trading)
- [Fractal Geometry & Bitcoin (MDPI, 2023)](https://www.mdpi.com/2504-3110/7/12/870)

### New Alpha Sources
- [Bitcoin Magazine Pro - ETF Flow Strategy](https://bitcoinmagazine.com/markets/bitcoin-etf-flow-strategy-beats-buy-and-hold)
- [BIS Working Paper 1087 - Crypto Carry](https://www.bis.org/publ/work1087.pdf)
- [Palazzi (2025) - Trading Games: Pairs Trading, J. Futures Markets](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018)
- [Funding Rate Arbitrage (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- [Altcoin Rotation (AInvest)](https://www.ainvest.com/news/timing-altcoin-rotation-strategic-entry-points-q4-2025-2512/)

### Factor Models
- [C-4 Factor Model (arXiv 2510.14435)](https://arxiv.org/abs/2510.14435)
- [Sparkline Capital - Crypto Factor Investing](https://www.sparklinecapital.com/post/crypto-factor-investing)
- [Fama-MacBeth Crypto Factor Model (MDPI)](https://www.mdpi.com/2227-7390/12/9/1351)
- [CGA-Agent RSI Optimization (arXiv 2510.07943)](https://arxiv.org/abs/2510.07943)

### Market Structure
- [Why Bitcoin Price Top Indicators Failed](https://bitcoinmagazine.com/markets/why-bitcoin-price-top-indicators-failed)
- [Crypto Hedge Fund Statistics 2025](https://coinlaw.io/crypto-hedge-funds-statistics/)
- [Risk Premia in Bitcoin Market (arXiv 2410.15195)](https://arxiv.org/abs/2410.15195)
- [Cryptocurrency Anomalies and Economic Constraints (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S1057521924001509)
