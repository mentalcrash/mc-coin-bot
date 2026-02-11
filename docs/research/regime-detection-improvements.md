# Regime Detection 성능 개선 리서치

> **작성일:** 2026-02-10
> **대상:** `src/regime/` Ensemble Regime Detector
> **목적:** IS/OOS decay 50-160% 문제 해결 및 앙상블 분류 정확도 향상

---

## 현재 시스템 진단

### 아키텍처

```
Rule-Based (w=0.40)  ──┐
HMM 3-state (w=0.35) ──┼──▶ Weighted Avg ──▶ Hysteresis (5-bar) ──▶ Label
Vol-Structure (w=0.25)──┘
```

### 핵심 문제점

| 문제 | 원인 | 영향 |
|------|------|------|
| IS/OOS decay 50-160% | Regime parameter 과적합 | 독립 전략 4개 모두 실패 |
| Regime flickering | HMM memoryless transition | 잦은 전환 → 거래비용 증가 |
| 정보 부족 | 순수 가격 기반 features만 사용 | 선행 정보(funding rate, OI) 누락 |
| 고정 가중치 | 시장 상황 무관 동일 가중치 | 특정 시기 특정 detector 성능 반영 불가 |
| Feature 중복 | Rule-Based/Vol-Structure 모두 RV ratio 사용 | 앙상블 다양성 부족 |
| Geometric duration | HMM의 memoryless 체류 시간 | 비현실적 regime 지속 모델링 |

---

## 1. HMM 고급 변형

### 1-A. Sticky HMM (HDP-HMM) — P0

**현재 문제:** GaussianHMM은 self-transition과 cross-transition을 동등하게 취급 → regime flickering 유발. `min_hold_bars` hysteresis는 사후 필터링일 뿐 모델 레벨 해결이 아님.

**핵심 아이디어:** Self-transition probability에 bias(kappa)를 추가하여 현재 state에 머무르려는 관성(stickiness)을 모델에 내장.

```python
# bayesian-hmm 패키지 사용
from bayesian_hmm import HDPHMM

model = HDPHMM(
    emission_type='gaussian',
    sticky=True,
    kappa=50.0,        # self-transition bias (높을수록 지속적)
    alpha=1.0,         # concentration parameter
)
model.fit(observations)
```

**기대 효과:**

- Regime 전환 빈도 50-70% 감소
- Hysteresis filter 의존도 감소 (모델이 자체적으로 안정)
- False signal 대폭 감소

**난이도:** 중간 | **구현 시간:** 2-3일
**참고:** [Fox et al. (Berkeley)](https://people.eecs.berkeley.edu/~jordan/papers/stickyHDPHMM_LIDS_TR.pdf), [bayesian-hmm PyPI](https://pypi.org/project/bayesian-hmm/)

---

### 1-B. Markov-Switching AR (MSAR) — P1

**현재 문제:** HMM은 관측값만 모델링. Regime별 dynamics(trending에서는 momentum 지속, ranging에서는 mean-reversion)를 포착하지 못함.

**핵심 아이디어:** 각 regime에서 별도의 AR 프로세스를 학습. "Trending regime에서는 AR 계수가 양수(momentum), Ranging에서는 음수(MR)" 같은 패턴을 자동 학습.

```python
import statsmodels.api as sm

model = sm.tsa.MarkovAutoregression(
    endog=log_returns,
    k_regimes=3,
    order=4,              # AR(4)
    switching_ar=True,     # regime별 AR 계수
    switching_variance=True,  # regime별 분산
)
result = model.fit()
smoothed_probs = result.smoothed_marginal_probabilities  # (n_obs, 3)
```

**장점:**

- statsmodels 내장 → 추가 의존성 불필요
- Regime별 dynamics를 명시적으로 포착
- Trending/MR 구분 정확도 향상

**기대 효과:** Regime 분류 정확도 10-15% 향상 (특히 trending vs ranging 구분)
**난이도:** 낮음 | **구현 시간:** 1일
**참고:** [statsmodels MarkovAutoregression](https://www.statsmodels.org/dev/examples/notebooks/generated/markov_autoregression.html)

---

### 1-C. Hidden Semi-Markov Model (HSMM) — P2

**현재 문제:** HMM의 geometric duration 가정 — regime에 3일 있든 30일 있든 transition probability 동일. 비현실적.

**핵심 아이디어:** 각 state의 체류 시간 분포를 명시적으로 모델링. "Trending regime은 평균 15일, Volatile은 평균 5일" 같은 정보 활용 가능.

```python
# hsmmlearn 패키지
from hsmmlearn.hsmm import GaussianHSMM

model = GaussianHSMM(
    n_states=3,
    n_durations=50,  # 최대 체류 기간 (bars)
)
model.fit(observations)

# 또는 Bayesian HSMM (pyhsmm)
import pyhsmm
model = pyhsmm.models.WeakLimitHDPHSMM(
    alpha=6.0, gamma=6.0,
    obs_distns=[...],
    dur_distns=[
        pyhsmm.distributions.PoissonDuration(lmbda=15)  # trending 평균 15일
        for _ in range(3)
    ],
)
```

**기대 효과:**

- Regime 지속 시간 예측 가능 → "이 trending이 언제 끝날까" 추정
- Hysteresis를 모델 레벨에서 자연스럽게 구현

**난이도:** 중간 | **구현 시간:** 3일
**참고:** [hsmmlearn GitHub](https://github.com/jvkersch/hsmmlearn), [pyhsmm GitHub](https://github.com/mattjj/pyhsmm)

---

### 1-D. Online Learning (Sliding Window + Decay) — P1

**현재 문제:** `retrain_interval=21`의 expanding window — 오래된 패턴에 과적합. 2020년 crypto 패턴이 2025년에 유효할 보장 없음.

**권장 전략:**

```python
# Sliding window + exponential decay
WINDOW_SIZE = 504          # 2년 daily
DECAY_HALF_LIFE = 126      # 6개월

# 최근 데이터에 더 가중치
weights = np.exp(-np.log(2) * np.arange(WINDOW_SIZE)[::-1] / DECAY_HALF_LIFE)

# Weighted EM fitting
model.fit(recent_data[-WINDOW_SIZE:], sample_weight=weights)
```

**기대 효과:** Regime shift 적응 속도 2-4x 향상
**난이도:** 낮음 | **구현 시간:** 0.5일

---

## 2. Feature Engineering

### 2-A. Cross-Asset Regime Signals — P0

**가장 높은 ROI.** 현재 시스템은 순수 가격 기반 features만 사용 — 선행 정보를 놓치고 있음.

| Feature | 데이터 소스 | Regime Signal |
|---------|------------|---------------|
| **Funding Rate (8h)** | Binance API | >0.1%=과레버리지, <-0.05%=bearish |
| **Open Interest 변화율** | Binance API | 급감=liquidation cascade |
| **BTC Dominance** | CoinMarketCap/Binance | >60%=risk-off, <45%=alt season |
| **BTC-ETH Correlation** | 자체 계산 | 감소=regime transition 선행 |

```python
def cross_asset_regime_features(
    funding_rate_8h: float,
    oi_change_pct: float,
    btc_dominance: float,
    btc_eth_corr_30d: float,
) -> dict[str, float]:
    annual_fr = funding_rate_8h * 3 * 365

    return {
        "fr_zscore": funding_rate_zscore,
        "oi_momentum": oi_change_pct,
        "btc_dominance_level": sigmoid(btc_dominance, center=55, scale=5),
        "decorrelation_signal": 1.0 - btc_eth_corr_30d,
        "leverage_risk": sigmoid(annual_fr, center=25, scale=5),
    }
```

**핵심 근거:**

- 2025년 10월 crash: funding rate 10%→30% 상승 → $3.21B liquidation cascade
- Funding rate, OI는 **가격에 선행하는** 정보
- BTC-ETH decorrelation은 regime transition의 leading indicator

**기대 효과:** Regime 전환 조기 감지 1-3일
**난이도:** 낮음 | **구현 시간:** 1-2일
**참고:** [Amberdata Oct 2025 Crash Analysis](https://blog.amberdata.io/how-3.21b-vanished-in-60-seconds-october-2025-crypto-crash-explained-through-7-charts)

---

### 2-B. Multi-Timeframe Feature Fusion — P2

**현재 문제:** 단일 timeframe (1D)만 사용 — 미시/거시 regime 불일치 감지 불가.

```python
features = {
    # Short-term (4H)
    "rv_ratio_4h": compute_rv_ratio(close_4h, short=5, long=20),
    "er_4h": compute_er(close_4h, window=10),

    # Medium-term (1D) — 현재 사용
    "rv_ratio_1d": compute_rv_ratio(close_1d, short=5, long=20),
    "er_1d": compute_er(close_1d, window=10),

    # Long-term (1W)
    "rv_ratio_1w": compute_rv_ratio(close_1w, short=4, long=13),
    "er_1w": compute_er(close_1w, window=8),

    # Cross-TF alignment (모든 TF가 같은 regime이면 신뢰도 높음)
    "tf_alignment": int(all_same_regime),
}
```

**기대 효과:** TF alignment 시 regime 신뢰도 향상, false signal 30-50% 감소
**난이도:** 낮음 | **구현 시간:** 1-2일

---

### 2-C. Liquidation Cascade 조기 경보 — P1

```python
cascade_features = {
    "oi_to_mcap_ratio": open_interest / market_cap,      # >0.05 = 위험
    "funding_rate_zscore": z_score(funding_rate, window=30),
    "bid_ask_spread_expansion": spread / spread.rolling(20).mean(),
}
# 2025/10 사례: baseline $0.71B/h → peak $10.39B/h (15x)
```

---

### 2-D. Feature Diversity (Detector Orthogonality) — P2

**현재 문제:** Rule-Based와 Vol-Structure 모두 RV ratio 사용 → 앙상블 다양성 부족.

**권장 재구성:**

| Detector | Feature Space | 포착 대상 |
|----------|--------------|-----------|
| **A: Direction** | ER, momentum, trend strength | 방향성/추세 |
| **B: Volatility** | RV ratio, GARCH, ATR | 변동성 구조 |
| **C: External** | Funding rate, OI, BTC.D | 외부 시장 요인 |
| **D: Microstructure** (선택) | Volume profile, spread | 유동성 구조 |

**핵심:** 각 detector가 **서로 다른 정보 원천**을 사용해야 앙상블 효과 극대화.

---

## 3. 앙상블 방법 개선

### 3-A. Meta-Learner Stacking — P1

**현재:** 고정 가중치 weighted average → 모든 시장에서 동일.

**개선:** 각 detector의 확률 출력을 meta-learner의 input으로 사용.

```python
from sklearn.linear_model import LogisticRegression

# Level-0: 각 detector의 regime probabilities
X_meta = np.column_stack([
    rule_probs,           # (n, 3)
    hmm_probs,            # (n, 3)
    vol_structure_probs,  # (n, 3)
])  # → (n, 9)

# Level-1: Regularized logistic regression
meta = LogisticRegression(
    multi_class='multinomial',
    C=0.1,           # 과적합 방지
    max_iter=1000,
)
meta.fit(X_train, y_train)  # y: forward return 기반 사후 label
final_probs = meta.predict_proba(X_test)
```

**장점:**

- 고정 가중치 대비 OOS accuracy 10-20% 향상
- "HMM이 trending에서 잘 맞고, Rule-Based가 volatile에서 잘 맞다" 같은 패턴 자동 학습
- Walk-forward re-training으로 과적합 방지

**기대 효과:** Regime 분류 정확도 10-20% 향상
**난이도:** 낮음 | **구현 시간:** 1일

---

### 3-B. Dynamic Weight Adjustment — P1

**핵심 아이디어:** 최근 N bars에서의 각 detector 성과를 추적하여 가중치 실시간 조정.

```python
class DynamicEnsemble:
    def __init__(self, n_detectors: int = 3, decay: float = 0.95):
        self.weights = np.ones(n_detectors) / n_detectors
        self.decay = decay

    def update_weights(
        self,
        detector_predictions: list[str],
        realized_label: str,  # 사후 판정 (e.g., 20일 후 returns 기반)
    ) -> None:
        for i, pred in enumerate(detector_predictions):
            if pred == realized_label:
                self.weights[i] *= (1 + (1 - self.decay))
            else:
                self.weights[i] *= self.decay
        self.weights /= self.weights.sum()
```

**"realized_label" 정의 방법:**

- Forward 20일 return > 2sigma → VOLATILE
- Forward 20일 |return| > 1sigma + ER > 0.4 → TRENDING
- 그 외 → RANGING

**기대 효과:** 시장 변화에 자동 적응, regime shift 후 2-4주 내 가중치 재조정
**난이도:** 중간 | **구현 시간:** 2일

---

### 3-C. Bayesian Model Averaging (BMA) — P3

고정 가중치 대신 사후 확률(posterior probability)로 가중치 산출. 데이터가 많을수록 더 나은 detector에 자동으로 가중치 집중.

```python
# 간소화된 BMA
# 각 detector의 marginal likelihood 추정
log_marginal = [detector.log_likelihood(data) for detector in detectors]
weights = softmax(log_marginal)
```

---

## 4. Regime Transition 모델링

### 4-A. BOCPD (Bayesian Online Changepoint Detection) — P1

**현재 문제:** HMM은 "현재 regime"만 추정. "regime이 바뀌려 한다"는 조기 경보 기능 없음.

**핵심 아이디어:** 매 bar마다 "현재 run length (동일 regime 지속 기간)"의 확률 분포를 추적. Run length = 0이면 changepoint.

```python
# Adams & MacKay (2007) 알고리즘
# 또는 Facebook Kats 라이브러리
from kats.detectors.bocpd import BOCPDetector, BOCPDModelType

detector = BOCPDetector(data=time_series)
changepoints = detector.detector(
    model=BOCPDModelType.NORMAL_KNOWN_MODEL,
    changepoint_prior=0.01,  # 낮을수록 보수적
)
```

**실용적 통합 방안:**

```
일반 상황: hysteresis min_hold_bars = 5 (보수적)
BOCPD alert 시: min_hold_bars = 1 (즉시 전환 허용)
```

- BOCPD가 changepoint 감지 → 앙상블에 "transition mode" 진입 알림
- Transition mode에서는 hysteresis 완화 + 최근 detector에 가중치 부여

**기대 효과:** Regime transition 감지 속도 2-5일 향상
**난이도:** 중간 | **구현 시간:** 2일
**참고:** [Adams & MacKay 2007](https://arxiv.org/abs/0710.3742), [Python 구현](https://gregorygundersen.com/blog/2020/10/20/implementing-bocd/)

---

### 4-B. Statistical Jump Model — P3

**최근 주목받는 접근법.** Viterbi-like DP에 "jump penalty" 부과 → regime 전환에 비용을 부여.

```python
def jump_model_decode(
    observations: np.ndarray,
    n_regimes: int = 3,
    jump_penalty: float = 10.0,  # 높을수록 전환 억제
) -> np.ndarray:
    # emission: Gaussian log-likelihood per regime
    # transition: jump_penalty for regime change
    # DP 최적화로 최적 state sequence 탐색
    ...
```

**장점:**

- Jump penalty를 walk-forward CV로 최적화 가능
- Hysteresis를 모델 레벨에서 자연스럽게 구현
- HMM보다 해석 용이

**기대 효과:** Regime persistence 보장 + 최적 전환 시점 탐색
**난이도:** 중간-높음 | **구현 시간:** 5-7일
**참고:** [Regime-Aware Asset Allocation (arXiv:2402.05272)](https://arxiv.org/abs/2402.05272)

---

### 4-C. Time-Varying Transition Matrix — P2

```python
def rolling_transition_matrix(
    regime_labels: pd.Series,
    window: int = 126,  # 6개월
) -> np.ndarray:
    recent = regime_labels.iloc[-window:]
    transitions = np.zeros((3, 3))
    for i in range(len(recent) - 1):
        fr = label_to_idx[recent.iloc[i]]
        to = label_to_idx[recent.iloc[i + 1]]
        transitions[fr, to] += 1
    return transitions / transitions.sum(axis=1, keepdims=True)
```

**활용:** "Volatile → Trending 전환 확률이 현재 40%"  같은 정보로 포지션 사이징 조정.

---

## 5. Calibration & Validation

### 5-A. Probability Calibration — P2

**현재 문제:** Sigmoid 기반 score가 진정한 확률이 아님. "p_trending=0.6"이 실제 60%인지 보장 없음.

**Platt Scaling:**

```python
from sklearn.calibration import CalibratedClassifierCV

calibrator = CalibratedClassifierCV(
    base_estimator=regime_model,
    method='sigmoid',  # 샘플 <1000
    cv=5,
)
calibrator.fit(X_train, y_train)
calibrated_probs = calibrator.predict_proba(X_test)
```

**Isotonic Regression:** 샘플 >1000이면 더 유연.

**기대 효과:** Probability threshold 기반 trading 결정의 신뢰성 향상
**난이도:** 낮음 | **구현 시간:** 0.5일

---

### 5-B. Walk-Forward Regime Validation Framework — P3

**IS/OOS decay 근본 해결.**

```
Fold 1: [── Train 2Y ──][─ Val 6M ─][─ Test 6M ─]
Fold 2:      [── Train 2Y ──][─ Val 6M ─][─ Test 6M ─]
Fold 3:           [── Train 2Y ──][─ Val 6M ─][─ Test 6M ─]
```

**핵심 검증 지표:**

```python
@dataclass
class RegimeValidationMetrics:
    avg_duration: dict[str, float]         # 평균 regime 지속 (bars)
    self_transition_prob: dict[str, float]  # 자기 transition 확률 (>0.9 목표)
    regime_distribution: dict[str, float]   # 각 regime 비율
    is_oos_kl_divergence: float            # IS/OOS 분포 차이 (<0.1 목표)
    regime_sharpe: dict[str, float]         # 각 regime에서의 strategy Sharpe
    directional_accuracy: dict[str, float]  # TRENDING: |fwd|>3%, RANGING: |fwd|<3%
    transition_accuracy: float              # 실제 regime shift 감지 비율
```

**DSR에 regime trials 포함:**

```python
total_trials = (
    n_strategy_params
    * n_regime_thresholds
    * n_hold_bars_tested
    * n_weight_combinations
)
dsr = compute_deflated_sharpe(sharpe, total_trials, T, skew, kurtosis)
```

---

## 6. Crypto-Specific 고려사항

### 6-A. Funding Rate Regime

```python
def classify_funding_regime(funding_rate_8h: float) -> str:
    annual = funding_rate_8h * 3 * 365
    if annual > 25:   return "overleveraged_long"    # cascade 위험
    elif annual > 10: return "bullish_speculation"
    elif annual > -5: return "neutral"
    elif annual > -20: return "bearish_pressure"
    else:             return "overleveraged_short"    # short squeeze 가능
```

**2025/10 사례:** Annual FR 10%→30% → $3.21B liquidation in 60s

---

### 6-B. Correlation Regime

```python
def detect_correlation_regime(
    btc_returns: pd.Series,
    alt_returns: pd.Series,
    window: int = 30,
) -> str:
    corr = btc_returns.rolling(window).corr(alt_returns).iloc[-1]
    if corr > 0.85:   return "systemic_risk"      # 모든 자산 동반 하락 위험
    elif corr > 0.5:  return "normal"
    else:             return "decorrelation"        # alt season / idiosyncratic
```

**핵심:** BTC-ETH correlation 감소 = regime transition의 leading indicator (1-3일 선행)

---

### 6-C. Weekend/Off-Hours 효과

- 주말: 유동성 40-60% 감소 → momentum 효과 증폭, 그러나 regime signal 신뢰도 하락
- Asia session (00-08 UTC): 변동성 상대적 낮음
- **권장:** 주말 regime 전환 signal은 confidence multiplier 0.7 적용

---

## 7. State-of-the-Art 논문 (2024-2026)

### 7-A. Signature-MMD Regime Detection

Path signature + Maximum Mean Discrepancy. 비마코프, 비모수적 regime detection.

- **GitHub:** [issaz/signature-regime-detection](https://github.com/issaz/signature-regime-detection)
- **논문:** [arXiv:2306.15835](https://arxiv.org/abs/2306.15835)
- **난이도:** 높음

### 7-B. Wasserstein Distance Regime Clustering

Sliced Wasserstein K-means. 비가우시안 분포에서 전통 방법 대비 우수.

- **논문:** [arXiv:2110.11848](https://arxiv.org/abs/2110.11848)
- **난이도:** 중간-높음

### 7-C. Multi-Model Ensemble-HMM Voting (Oct 2025)

XGBoost/CatBoost + HMM voting. Walk-forward validation 검증.

- **논문:** [AIMS Press](https://www.aimspress.com/article/id/69045d2fba35de34708adb5d)
- **난이도:** 중간

### 7-D. Market States: Clustering + State Machines (Oct 2025)

K-means + transition matrix → probabilistic state machine. 6개 TF의 log-momentum + risk 사용.

- **논문:** [arXiv:2510.00953](https://arxiv.org/abs/2510.00953)
- **난이도:** 낮음-중간

### 7-E. State Street: t-Distribution Mixture (2025)

Gaussian 대신 t-distribution 사용 → heavy tail 대응. 23개 features → 4 regimes.

- **논문:** [SSGA](https://www.ssga.com/library-content/assets/pdf/global/pc/2025/decoding-market-regimes-with-machine-learning.pdf)
- **난이도:** 중간

### 7-F. HMM + RL Portfolio Management (2025)

Bayesian regime filtration + RL. Test accuracy >96%, lead time ~2.4 trading days.

- **논문:** [IDS 2025](https://www.cloud-conf.net/datasec/2025/proceedings/pdfs/IDS2025-3SVVEmiJ6JbFRviTl4Otnv/966100a067/966100a067.pdf)
- **난이도:** 높음

### 7-G. TCN-HMM Hybrid

TCN으로 multi-scale temporal features 추출 → HMM clustering. LSTM보다 3-5x 빠른 학습.

- **참고:** [TCN-HMM Framework](https://medium.com/call-for-atlas/temporal-convolutional-neural-network-with-conditioning-for-broad-market-signals-9f0b0426b2b9)
- **난이도:** 중간

---

## 8. 구현 우선순위 로드맵

### Tier 1 (P0) — 즉시 구현, 최대 ROI

| # | 개선 사항 | Impact | 난이도 | 시간 |
|---|----------|--------|--------|------|
| 1 | **Cross-asset features** (funding rate, OI, BTC.D) | 높음 | 낮음 | 1-2일 |
| 2 | **Sticky HMM** 교체 (flickering 근본 해결) | 높음 | 중간 | 2-3일 |

### Tier 2 (P1) — 높은 효과, 합리적 비용

| # | 개선 사항 | Impact | 난이도 | 시간 |
|---|----------|--------|--------|------|
| 3 | **MSAR** 모델 추가 (statsmodels 내장) | 중간 | 낮음 | 1일 |
| 4 | **Meta-learner stacking** (고정 가중치 대체) | 중간 | 낮음 | 1일 |
| 5 | **BOCPD** 조기 경보 통합 | 중간 | 중간 | 2일 |
| 6 | **Online learning** (sliding window + decay) | 중간 | 낮음 | 0.5일 |

### Tier 3 (P2) — 점진적 개선

| # | 개선 사항 | Impact | 난이도 | 시간 |
|---|----------|--------|--------|------|
| 7 | Multi-TF feature fusion (4h/1d/1w) | 중간 | 낮음 | 1-2일 |
| 8 | Probability calibration (Platt scaling) | 낮음-중간 | 낮음 | 0.5일 |
| 9 | Detector diversity 개선 (feature orthogonality) | 중간 | 중간 | 3-5일 |
| 10 | HSMM (duration modeling) | 중간 | 중간 | 3일 |
| 11 | Time-varying transition matrix | 낮음 | 낮음 | 1일 |

### Tier 4 (P3) — 연구/실험

| # | 개선 사항 | Impact | 난이도 | 시간 |
|---|----------|--------|--------|------|
| 12 | Statistical Jump Model | 높음 | 높음 | 5-7일 |
| 13 | Walk-forward validation framework | 높음 | 중간 | 3-5일 |
| 14 | Signature-MMD (비모수적) | 높음 | 높음 | 7-10일 |
| 15 | TCN feature extractor | 중간 | 높음 | 5-7일 |

---

## 9. 핵심 인사이트

### 근본 원칙

1. **"거래 가능한 안정성" > "분류 정확도"**
   - 정확하지만 불안정한 signal보다 약간 부정확하지만 일관된 signal이 낫다
   - Sticky HMM + BOCPD가 이 원칙을 가장 잘 구현

2. **IS/OOS decay의 근본 원인은 non-stationarity**
   - Crypto 시장 구조는 2-3년 주기로 근본적 변화
   - → Sliding window + online weight adjustment 필수
   - → 고정 parameter 최소화, 적응적 parameter 최대화

3. **Cross-asset features가 가격에 선행**
   - Funding rate, OI 변화는 가격 변동 1-3일 전에 신호 발생
   - 현재 시스템의 가장 큰 정보 손실 영역

4. **Regime은 strategy enhancement tool**
   - 독립 전략으로 사용 시 과적합 위험 극히 높음 (4개 모두 실패)
   - 기존 전략의 position sizing / vol-target 조정 도구로 사용 시 유효

5. **앙상블 다양성이 정확도보다 중요**
   - 같은 feature를 쓰는 3개 detector보다
   - 서로 다른 정보원의 2개 detector가 낫다

### DL에 대한 경고

- Crypto 일봉 학습 데이터 절대 부족 (~3,650 samples/coin)
- Non-stationarity → 과적합 위험 극히 높음
- **권장:** DL은 feature extractor로만 사용, 최종 분류는 전통 방법 유지
- LSTM/Transformer regime classifier 단독 사용은 비권장

---

## 참고 문헌

### HMM / 확률 모델

- Fox et al., "Sticky HDP-HMM" — [Berkeley](https://people.eecs.berkeley.edu/~jordan/papers/stickyHDPHMM_LIDS_TR.pdf)
- statsmodels MarkovAutoregression — [Docs](https://www.statsmodels.org/dev/examples/notebooks/generated/markov_autoregression.html)
- hsmmlearn — [GitHub](https://github.com/jvkersch/hsmmlearn)
- pyhsmm — [GitHub](https://github.com/mattjj/pyhsmm)

### Changepoint / Transition

- Adams & MacKay (2007), "BOCPD" — [arXiv:0710.3742](https://arxiv.org/abs/0710.3742)
- Gregory Gundersen BOCPD Python — [Blog](https://gregorygundersen.com/blog/2020/10/20/implementing-bocd/)
- Jump Model — [arXiv:2402.05272](https://arxiv.org/abs/2402.05272)

### Crypto Market

- Amberdata Oct 2025 Crash — [Blog](https://blog.amberdata.io/how-3.21b-vanished-in-60-seconds-october-2025-crypto-crash-explained-through-7-charts)
- FTI Oct 2025 Analysis — [Report](https://www.fticonsulting.com/insights/articles/crypto-crash-october-2025-leverage-met-liquidity)
- Liquidation Cascade Anatomy — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5611392)

### 최신 연구

- Signature-MMD — [arXiv:2306.15835](https://arxiv.org/abs/2306.15835)
- Wasserstein Clustering — [arXiv:2110.11848](https://arxiv.org/abs/2110.11848)
- Market States Clustering — [arXiv:2510.00953](https://arxiv.org/abs/2510.00953)
- State Street ML Regimes — [SSGA 2025](https://www.ssga.com/library-content/assets/pdf/global/pc/2025/decoding-market-regimes-with-machine-learning.pdf)
- HMM+RL — [IDS 2025](https://www.cloud-conf.net/datasec/2025/proceedings/pdfs/IDS2025-3SVVEmiJ6JbFRviTl4Otnv/966100a067/966100a067.pdf)
