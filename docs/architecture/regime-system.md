# Regime Detection System Architecture

시장 상태(TRENDING/RANGING/VOLATILE)를 실시간 감지하여 전략에 제공하는 공유 인프라.
5개 감지기의 앙상블 블렌딩 + EDA 통합 서비스로 구성.

> **핵심 목표:** 전략이 시장 레짐에 적응적으로 파라미터를 조절할 수 있도록 투명한 regime enrichment 제공

---

## 1. 전체 구조 개요

```
종가 시리즈          Derivatives DF (funding_rate, oi)
    │                       │
    ▼                       ▼
┌────────────────────────────────────────────────────────────┐
│              EnsembleRegimeDetector                          │
│  ┌──────────┐ ┌──────┐ ┌─────────┐ ┌──────┐ ┌───────────┐ │
│  │Rule-Based│ │ HMM  │ │Vol-Str. │ │ MSAR │ │Derivatives│ │
│  │ (항상)   │ │(선택)│ │ (선택)  │ │(선택)│ │  (선택)   │ │
│  └────┬─────┘ └──┬───┘ └───┬─────┘ └──┬───┘ └─────┬─────┘ │
│       └────┬─────┴────┬────┘           │           │       │
│            ▼          ▼                │           │       │
│   Weighted Average 또는 Meta-Learner ◄──┘           │       │
│            │                                       │       │
│            ▼                                       │       │
│   p_trending / p_ranging / p_volatile              │       │
│   + confidence (detector agreement)                │       │
│   + Hysteresis → regime_label                      │       │
│   + cascade_risk (from Derivatives) ◄──────────────┘       │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│               RegimeService                                 │
│  Backtest: precompute() → enrich_dataframe()                │
│  Live: _on_bar() → get_regime_columns()                     │
│  + trend_direction / trend_strength (EWM momentum)          │
│  + confidence / transition_prob / cascade_risk              │
│  + REGIME_CHANGE event (label 변경 시 EventBus 발행)          │
│  + RegimeContext API (전략 소비용)                              │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│             StrategyEngine                                   │
│  _apply_enrichments() → regime 컬럼 주입                      │
│  → strategy.generate_signals()에서 활용                       │
│  → RegimeContext로 vol_scalar 등 편의 메트릭 접근               │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 모듈 구조

### 2.1 파일 맵

| 파일 | 클래스 | 역할 |
|------|--------|------|
| `src/regime/config.py` | `RegimeDetectorConfig`, `HMMDetectorConfig`, `VolStructureDetectorConfig`, `MSARDetectorConfig`, `DerivativesDetectorConfig`, `MetaLearnerConfig`, `EnsembleRegimeDetectorConfig` | Pydantic 설정 (frozen=True) |
| `src/regime/detector.py` | `RegimeDetector`, `RegimeState` | Rule-Based 감지기 (RV Ratio + ER) |
| `src/regime/vol_detector.py` | `VolStructureDetector` | Vol-Structure 감지기 |
| `src/regime/hmm_detector.py` | `HMMDetector` | HMM 감지기 (hmmlearn, 선택) |
| `src/regime/msar_detector.py` | `MSARDetector` | MSAR 감지기 (statsmodels, 선택) |
| `src/regime/derivatives_detector.py` | `DerivativesDetector` | Derivatives 감지기 (funding rate + OI) |
| `src/regime/ensemble.py` | `EnsembleRegimeDetector` | 5개 감지기 앙상블 |
| `src/regime/service.py` | `RegimeService`, `RegimeServiceConfig`, `EnrichedRegimeState`, `RegimeContext` | EDA 통합 서비스 |

### 2.2 의존성 흐름

```
RegimeService (+ derivatives_provider)
  └─ EnsembleRegimeDetector
       ├─ RegimeDetector (항상)
       ├─ HMMDetector (선택 -- hmmlearn)
       ├─ VolStructureDetector (선택)
       ├─ MSARDetector (선택 -- statsmodels)
       └─ DerivativesDetector (선택 -- derivatives 데이터)

StrategyEngine → RegimeService (소비자)
EDARunner → RegimeService (생성자, derivatives_provider 주입)
```

---

## 3. 레짐 라벨 (3-State)

```python
class RegimeLabel(StrEnum):
    TRENDING  = "trending"   # 추세장: ER 높음, 방향성 강함
    RANGING   = "ranging"    # 횡보장: ER 낮음, 변동성 낮음
    VOLATILE  = "volatile"   # 고변동장: RV 높음, ER 낮음
```

모든 감지기가 이 3개 라벨에 대한 soft probability (합=1.0)를 출력합니다.

---

## 4. 감지기 상세

### 4.1 Rule-Based (항상 활성)

**파일:** `src/regime/detector.py`

**지표:**

- **RV Ratio** = `std(log_returns, short_window) / std(log_returns, long_window)`
  - 단기/장기 변동성 비율. >1이면 변동성 확장, <1이면 수축
- **Efficiency Ratio (ER)** = `|close[t] - close[t-N]| / sum(|close[i] - close[i-1]|)`
  - 방향성 효율. 1에 가까우면 강한 추세, 0에 가까우면 횡보

**분류 알고리즘:**

```
s_trending = sigmoid(ER, center=er_threshold, scale=6)
s_volatile = sigmoid(RV_ratio, center=rv_expansion, scale=8)
s_ranging  = (1 - sigmoid(RV_ratio, 1.0, scale=5)) * (1 - s_trending)

normalize → p_trending + p_ranging + p_volatile = 1.0
argmax → hard label
hysteresis(min_hold_bars) → 최종 라벨
```

**기본 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `rv_short_window` | 5 | 단기 변동성 윈도우 |
| `rv_long_window` | 20 | 장기 변동성 윈도우 |
| `er_window` | 10 | Efficiency Ratio 윈도우 |
| `er_trending_threshold` | 0.40 | 추세 판별 임계값 |
| `rv_expansion_threshold` | 1.3 | 변동성 확장 임계값 |
| `min_hold_bars` | 5 | 최소 레짐 유지 bar 수 |

**Warmup:** `rv_long_window + 5` bar

### 4.2 HMM (선택 - hmmlearn 필요)

**파일:** `src/regime/hmm_detector.py`

**알고리즘:**

- GaussianHMM expanding/sliding window training
- Features: `[log_returns, rolling_volatility]` (2D)
- State 매핑: mean return 기준 정렬
  - 최고 mean → Bull (TRENDING)
  - 최저 mean → Bear (TRENDING)
  - 중간 mean → Sideways (RANGING)
- VOLATILE을 직접 감지하지 않음 (p_volatile = 0.0)

**학습 주기:** `retrain_interval` (기본 21 bar)마다 재학습

**기본 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `n_states` | 3 | HMM 상태 수 |
| `min_train_window` | 252 | 최소 학습 데이터 (bar) |
| `retrain_interval` | 21 | 재학습 주기 |
| `vol_window` | 20 | 롤링 변동성 윈도우 |
| `sliding_window` | 0 | 학습 윈도우 (0=expanding) |
| `decay_half_life` | 0 | 가중치 반감기 (0=uniform) |

**Warmup:** `min_train_window + 1` bar

> **Known Issue — HMM State Ordering Instability:**
> HMM의 latent state → regime 매핑은 mean return 기준 정렬에 의존합니다.
> mean return이 유사한 경우 retrain 시 label flipping이 발생할 수 있습니다.
> 완화: `decay_half_life` 또는 `sliding_window`를 사용하여 최신 데이터에 가중치를 부여합니다.

### 4.3 Vol-Structure (선택)

**파일:** `src/regime/vol_detector.py`

**알고리즘:**

```
vol_ratio = vol_short / vol_long
norm_mom  = sum(log_returns[N]) / std(log_returns[N])

expansion_score = sigmoid(vol_ratio - 1.0)
momentum_score  = sigmoid(|norm_mom|, center=1.0)

p_trending = expansion_score * momentum_score
p_volatile = expansion_score * (1 - momentum_score)
p_ranging  = 1 - p_trending - p_volatile
```

**설계 의도:** 변동성 구조와 방향성을 동시에 고려. RV Ratio만 사용하는 Rule-Based보다 모멘텀 강도를 직접 반영.

**기본 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `vol_short_window` | 10 | 단기 변동성 윈도우 |
| `vol_long_window` | 60 | 장기 변동성 윈도우 |
| `mom_window` | 20 | 모멘텀 윈도우 |

**Warmup:** `max(vol_long_window, mom_window) + 1` bar

### 4.4 MSAR (선택 - statsmodels 필요)

**파일:** `src/regime/msar_detector.py`

**알고리즘:**

- Markov-Switching AutoRegression (MarkovAutoregression)
- 레짐별 variance 기준 매핑:
  - **3-regime:** 최저 분산 → RANGING / 중간 분산 → TRENDING / 최고 분산 → VOLATILE
  - **2-regime:** 최저 분산 → TRENDING / 최고 분산 → VOLATILE
- Smoothed marginal probabilities를 레짐 확률로 집계

**기본 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `k_regimes` | 3 | 레짐 수 |
| `order` | 2 | AR 차수 |
| `switching_variance` | True | 레짐별 분산 전환 |
| `sliding_window` | 504 | 학습 윈도우 (약 2년) |
| `min_train_window` | 252 | 최소 학습 데이터 |
| `retrain_interval` | 21 | 재학습 주기 |

**Warmup:** `min_train_window + 1` bar

### 4.5 Derivatives (선택 - derivatives 데이터 필요)

**파일:** `src/regime/derivatives_detector.py`

**입력:** DataFrame with `funding_rate`, `oi` columns

**지표:**

| Feature | 계산 | Signal |
|---------|------|--------|
| `funding_zscore` | `(FR - rolling_mean) / rolling_std` | 극단 레버리지 편향 |
| `funding_persistence` | 동방향 FR 연속 bar 수 | crash 선행지표 |
| `oi_change_rate` | `(OI - OI.shift(N)) / OI.shift(N)` | 레버리지 축적/해소 |
| `cascade_risk` | 위 3개의 가중 합성 0~1 | PM/RM 방어 트리거 |

**분류 알고리즘:**

```
s_volatile = sigmoid(|funding_zscore|, center=1.5, scale=4) *
             sigmoid(cascade_risk, center=0.4, scale=6)
s_trending = sigmoid(|funding_zscore|, center=0.5, scale=3) * (1 - s_volatile)
s_ranging  = max(0, 1 - s_trending - s_volatile)
normalize -> p_trending + p_ranging + p_volatile = 1.0
```

**Cascade Risk 계산:**

```
cascade_risk = 0.4 * sigmoid(|funding_zscore|, 1.5, 3.0)
             + 0.3 * sigmoid(|oi_change|, 0.1, 10.0)
             + 0.3 * (persistence / persistence_window).clip(0, 1)
```

**기본 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `funding_zscore_window` | 7 | Funding rate z-score 윈도우 |
| `oi_change_window` | 1 | OI 변화율 기간 |
| `funding_persistence_window` | 14 | 연속 동방향 FR 감시 |
| `cascade_risk_threshold` | 0.7 | 고위험 임계값 |

**Warmup:** `max(funding_zscore_window, funding_persistence_window) + 1` bar

---

## 5. 앙상블 방식

### 5.1 Weighted Average (기본)

```
blended_p = Σ(weight_i × p_i) / Σ(weight_i)   (NaN이 아닌 감지기만)
```

- 감지기별 warmup 완료 여부에 따라 **동적 가중치 재정규화**
- 가중치 합 = 1.0 검증 (활성 감지기 기준)

**기본 가중치:**

| 감지기 | 가중치 |
|--------|-------|
| Rule-Based | 1.0 |
| HMM | 0.0 |
| Vol-Structure | 0.0 |
| MSAR | 0.0 |
| Derivatives | 0.0 |

> 기본 설정에서는 Rule-Based만 활성. 다른 감지기 활성화 시 가중치 재배분 필요.

**감지기 활성화 예시 (Rule + Vol-Structure):**

```python
from src.regime.config import EnsembleRegimeDetectorConfig, VolStructureDetectorConfig
from src.regime.ensemble import EnsembleRegimeDetector

config = EnsembleRegimeDetectorConfig(
    vol_structure=VolStructureDetectorConfig(),  # Vol-Structure 활성화
    weight_rule_based=0.6,                       # 가중치 재배분
    weight_vol_structure=0.4,
    min_hold_bars=3,
)
detector = EnsembleRegimeDetector(config)
result = detector.classify_series(closes)
```

### 5.2 Confidence (Detector Agreement)

앙상블의 **confidence**는 활성 감지기가 최종 label에 동의하는 비율입니다:

```
confidence = (최종 label과 동일한 argmax를 가진 감지기 수) / (활성 감지기 수)
```

- 단일 감지기만 활성이면 `confidence = 1.0`
- 4개 감지기 모두 TRENDING이면 `confidence = 1.0`
- 3/4 TRENDING, 1/4 RANGING이면 `confidence = 0.75`

### 5.3 Meta-Learner (sklearn 필요)

```
features = [rule_pt, rule_pr, rule_pv, hmm_pt, hmm_pr, hmm_pv, ...]
labels   = forward_return 기반 사후 라벨
model    = LogisticRegression (walk-forward stacking)
```

- Walk-forward 방식: `train_window` 기간 학습 → `retrain_interval`마다 재학습
- Forward return 라벨링:
  - `|return| ≤ trending_threshold` → RANGING
  - `trending_threshold < |return| ≤ volatile_threshold` → TRENDING
  - `|return| > volatile_threshold` → VOLATILE
- **Live 미지원:** forward return이 필요하므로 incremental 모드에서는 weighted_average로 자동 폴백

---

## 6. Hysteresis (노이즈 필터링)

모든 감지기의 최종 라벨에 hysteresis를 적용하여 잦은 레짐 전환을 방지합니다.

```
if raw_label != current_label:
    if raw_label == pending_label:
        counter += 1
        if counter >= min_hold_bars:
            current_label = raw_label  # 전환 확정
    else:
        pending_label = raw_label
        counter = 1

# min_hold_bars 연속으로 같은 새 라벨이 나와야 전환
```

| 파라미터 | 범위 | 권장값 | 설명 |
|---------|------|-------|------|
| `min_hold_bars` | 1~20 | 3~5 | 작을수록 민감, 클수록 안정 |

### 6.1 Post-Blend Hysteresis 설계 의도

Hysteresis는 **앙상블 블렌딩 후** 최종 라벨에 적용됩니다 (per-detector가 아닌 post-blend).

**설계 trade-off:**

- **Post-blend (현재):** 앙상블 확률이 변동해도 최종 라벨은 안정적. 개별 detector 노이즈가 상쇄.
- **Per-detector:** 각 detector가 독립적으로 hysteresis를 적용. 더 안정적이나 앙상블 반응성 저하.

Post-blend 방식을 선택한 이유:

1. 단일 `min_hold_bars` 파라미터로 시스템 전체 안정성 제어
2. 앙상블 효과(노이즈 상쇄) 이후 적용하므로 불필요한 이중 필터링 방지
3. vectorized와 incremental에서 동일한 `apply_hysteresis()` 함수 공유 가능

---

## 7. RegimeService (EDA 통합)

### 7.1 역할

`EnsembleRegimeDetector`를 래핑하여 EDA 컴포넌트에 투명한 regime 정보를 제공합니다.
추가로 **추세 방향(trend_direction)과 강도(trend_strength)**를 EWM momentum으로 계산합니다.

### 7.2 출력 컬럼

```python
REGIME_COLUMNS = (
    "regime_label",            # str: "trending" | "ranging" | "volatile"
    "p_trending",              # float: 0.0~1.0
    "p_ranging",               # float: 0.0~1.0
    "p_volatile",              # float: 0.0~1.0
    "trend_direction",         # int: -1 (하락) | 0 (중립) | +1 (상승)
    "trend_strength",          # float: 0.0~1.0
    "regime_confidence",       # float: 0.0~1.0  -- detector agreement
    "regime_transition_prob",  # float: 0.0~1.0  -- 다음 bar 전환 확률
    "cascade_risk",            # float: 0.0~1.0  -- derivatives 기반 급락 위험
)
```

### 7.3 방향 계산 (EWM Momentum)

```
log_returns → EWM(span=direction_window).mean() → normalized / rolling_std
direction = +1 if normalized > threshold else -1
strength  = min(|normalized|, 1.0)
```

### 7.4 EnrichedRegimeState

```python
@dataclass
class EnrichedRegimeState:
    label: RegimeLabel           # TRENDING/RANGING/VOLATILE
    probabilities: dict[str, float]  # p_trending, p_ranging, p_volatile
    bars_held: int               # 현재 레짐 유지 bar 수
    raw_indicators: dict[str, float]  # rv_ratio, er 등
    trend_direction: int = 0     # -1 | 0 | +1
    trend_strength: float = 0.0  # 0.0~1.0
    confidence: float = 0.0      # detector agreement (0~1)
    transition_prob: float = 0.0 # 다음 bar 전환 확률 (0~1)
    cascade_risk: float = 0.0   # derivatives 급락 위험도 (0~1)
```

### 7.5 Transition Probability

3x3 Markov transition matrix를 라벨 시퀀스에서 추정하여 현재 레짐에서 다른 레짐으로
전환될 확률을 계산합니다. Laplace smoothing 적용.

```
transition_prob = 1.0 - P(current_label -> current_label)
```

- `precompute()`에서 vectorized label 시퀀스로 matrix 구축

> **Known Limitation — Look-Ahead in Backtest:**
> Backtest `precompute()`는 **전체 label 시퀀스**로 transition matrix를 추정합니다 (look-ahead).
> Live에서는 `_update_transition_counts()`로 expanding window 누적합니다.
> transition_prob는 정보용 메트릭이므로 전략 시그널에 직접 영향을 주지 않습니다.
> Phase 3에서 expanding-window 방식으로 개선 예정입니다.
- `_on_bar()`에서 incremental 카운터로 matrix 갱신

### 7.6 RegimeContext (전략 소비용 API)

```python
@dataclass(frozen=True)
class RegimeContext:
    label: RegimeLabel
    p_trending: float
    p_ranging: float
    p_volatile: float
    confidence: float
    transition_prob: float
    cascade_risk: float
    trend_direction: int
    trend_strength: float
    bars_in_regime: int
    suggested_vol_scalar: float  # 0.0~1.0
```

`suggested_vol_scalar` 계산:

```
base = p_trending * 1.0 + p_ranging * 0.4 + p_volatile * 0.2
if cascade_risk > threshold: return 0.1
return round(base * (0.5 + 0.5 * confidence), 2)
```

### 7.7 REGIME_CHANGE Event

레짐 라벨이 변경되면 EventBus에 `REGIME_CHANGE` 이벤트를 발행합니다:

```python
class RegimeChangeEvent(BaseModel):
    event_type: EventType = EventType.REGIME_CHANGE
    symbol: str
    prev_label: str
    new_label: str
    confidence: float
    cascade_risk: float = 0.0
    transition_prob: float = 0.0
```

- `DROPPABLE_EVENTS`에 포함 (stale 데이터 드롭 가능)
- bus가 None이면 발행 스킵 (backward compatible)

---

## 8. 데이터 흐름

### 8.1 백테스트 모드 (Vectorized)

```
EDARunner._build_strategy_engine_kwargs()
  │
  ├─ deriv_provider = _create_derivatives_provider()
  ├─ RegimeService(config, derivatives_provider=deriv_provider) 생성
  │
  ├─ regime_service.precompute(symbol, closes, deriv_df)
  │    ├─ EnsembleRegimeDetector.classify_series(closes, deriv_df) [벡터화]
  │    │    ├─ RegimeDetector.classify_series()
  │    │    ├─ HMMDetector.classify_series()  (활성 시)
  │    │    ├─ VolStructureDetector.classify_series()  (활성 시)
  │    │    ├─ MSARDetector.classify_series()  (활성 시)
  │    │    └─ DerivativesDetector.classify_series(deriv_df)  (활성 시)
  │    ├─ _compute_direction_vectorized(closes)
  │    ├─ _estimate_transition_matrix(labels) → 3x3 matrix
  │    ├─ confidence, transition_prob, cascade_risk 계산
  │    └─ 결과 캐시 → self._precomputed[symbol]
  │
  └─ StrategyEngine._on_bar()
       └─ _apply_enrichments()
            └─ _enrich_with_regime()
                 └─ regime_service.enrich_dataframe(df, symbol)
                      └─ precomputed.reindex(df.index)  [timestamp join]
```

### 8.2 라이브 모드 (Incremental)

```
RegimeService.register(bus) → EventType.BAR 구독 (1순위)
StrategyEngine.register(bus) → EventType.BAR 구독 (2순위)

WebSocket → BarEvent 발행
  │
  ├─ RegimeService._on_bar()  (1순위)
  │    ├─ TF 필터 (target_timeframe만)
  │    ├─ EnsembleRegimeDetector.update(symbol, close, derivatives)
  │    │    ├─ RegimeDetector.update() → 버퍼 기반 O(1)
  │    │    ├─ HMMDetector.update() → 주기적 재학습
  │    │    ├─ VolStructureDetector.update()
  │    │    ├─ MSARDetector.update()
  │    │    └─ DerivativesDetector.update() (derivatives 제공 시)
  │    ├─ _update_direction() → EWM momentum
  │    ├─ transition_prob, cascade_risk 업데이트
  │    ├─ label 변경 시 → REGIME_CHANGE 이벤트 발행
  │    └─ self._states[symbol] = EnrichedRegimeState
  │
  └─ StrategyEngine._on_bar()  (2순위)
       └─ _enrich_with_regime()
            ├─ enrich_dataframe() → 사전 계산 없음 (라이브)
            └─ get_regime_columns() → self._states 최신 값 broadcast
```

### 8.3 Live Warmup

```python
regime_service.warmup(symbol, historical_closes)
# 과거 close 리스트를 순차 update()하여 detector 초기화
```

---

## 9. StrategyEngine Enrichment 파이프라인

```python
def _apply_enrichments(df, symbol):
    df = self._enrich_with_regime(df, symbol)       # 1순위
    df = self._enrich_with_derivatives(df, symbol)
    df = self._enrich_with_features(df, symbol)
    df = self._enrich_with_onchain(df, symbol)
    df = self._enrich_with_macro(df, symbol)
    df = self._enrich_with_options(df, symbol)
    df = self._enrich_with_deriv_ext(df, symbol)
    return df
```

**Fallback 메커니즘:**

```python
def _enrich_with_regime(df, symbol):
    # 1. 사전 계산 우선 (백테스트)
    df = regime_service.enrich_dataframe(df, symbol)

    # 2. Fallback: 사전 계산 없으면 최신 state broadcast (라이브)
    if "regime_label" not in df.columns:
        cols = regime_service.get_regime_columns(symbol)
        if cols is not None:
            for col_name, col_val in cols.items():
                df[col_name] = col_val  # broadcast to all rows
```

---

## 10. 전략에서의 활용

### 10.1 Regime-Aware 파라미터 적응

```python
# 레짐별 Vol Target 조절 예시
adaptive_vol_target = (
    p_trending * trending_vol_target +   # 0.40 (공격적)
    p_ranging  * ranging_vol_target  +   # 0.15 (보수적)
    p_volatile * volatile_vol_target     # 0.10 (초보수)
)
vol_scalar = adaptive_vol_target / realized_vol
```

### 10.2 사전처리 패턴

```python
# strategy/regime_tsmom/preprocessor.py
def preprocess(df, config):
    result = add_regime_columns(df, config.regime)  # Rule-Based 단독
    # 또는 add_ensemble_regime_columns() 사용
    return tsmom_preprocess(result, config.to_tsmom_config())
```

### 10.3 YAML 설정 예시

```yaml
strategy:
  name: regime-tsmom
  params:
    lookback: 30
    regime:
      rv_short_window: 5
      rv_long_window: 20
      er_window: 10
      min_hold_bars: 3
    trending_vol_target: 0.40
    ranging_vol_target: 0.15
    volatile_vol_target: 0.10
```

---

## 11. 설정 계층

### 11.1 RegimeServiceConfig (EDA 통합)

```python
class RegimeServiceConfig(BaseModel):
    ensemble: EnsembleRegimeDetectorConfig  # 앙상블 설정
    direction_window: int = 10              # 추세 방향 EMA 윈도우
    direction_threshold: float = 0.0        # 추세 판별 임계값
    target_timeframe: str = "1D"            # BAR 이벤트 TF 필터
```

### 11.2 EnsembleRegimeDetectorConfig (감지기 조합)

```python
class EnsembleRegimeDetectorConfig(BaseModel):
    rule_based: RegimeDetectorConfig        # Rule-Based (항상 활성)
    hmm: HMMDetectorConfig | None           # HMM (None=비활성)
    vol_structure: VolStructureDetectorConfig | None  # Vol-Struct (None=비활성)
    msar: MSARDetectorConfig | None         # MSAR (None=비활성)
    derivatives: DerivativesDetectorConfig | None     # Derivatives (None=비활성)

    weight_rule_based: float = 1.0          # 가중치 합 = 1.0 필수
    weight_hmm: float = 0.0
    weight_vol_structure: float = 0.0
    weight_msar: float = 0.0
    weight_derivatives: float = 0.0

    min_hold_bars: int = 5                  # Hysteresis
    ensemble_method: "weighted_average" | "meta_learner"
    meta_learner: MetaLearnerConfig | None  # meta_learner 모드 시 필수
```

### 11.3 DerivativesDetectorConfig

```python
class DerivativesDetectorConfig(BaseModel):
    funding_zscore_window: int = 7       # 3~30
    oi_change_window: int = 1            # 1~7
    funding_persistence_window: int = 14 # 5~30
    cascade_risk_threshold: float = 0.7  # 0.3~1.0
```

---

## 12. Graceful Degradation

### 12.1 선택적 의존성

```python
# hmmlearn 미설치 시 자동 비활성
try:
    from hmmlearn.hmm import GaussianHMM
    _hmm_available = True
except ImportError:
    _hmm_available = False

# EnsembleRegimeDetector.__init__()
if config.hmm is not None and _hmm_available:
    self._hmm_detector = HMMDetector(config.hmm)
elif config.hmm is not None:
    logger.warning("hmmlearn not available -- HMM detector disabled")
```

### 12.2 Warmup 기간 동적 처리

- NaN인 감지기는 블렌딩에서 자동 제외
- 남은 감지기의 가중치를 재정규화하여 합 = 1.0 유지
- Rule-Based의 warmup이 가장 짧아 항상 먼저 활성화

### 12.3 Backward Compatibility

- `regime_service=None`이면 기존 StrategyEngine 동작 100% 유지
- regime 컬럼이 이미 DataFrame에 존재하면 덮어쓰지 않음

---

## 13. Dual API (Vectorized + Incremental)

모든 감지기가 두 가지 API를 제공합니다:

| API | 메서드 | 복잡도 | 용도 |
|-----|--------|-------|------|
| **Vectorized** | `classify_series(closes, deriv_df)` | O(n) | 백테스트 사전 계산 |
| **Incremental** | `update(symbol, close, derivatives)` | O(1)* | 라이브 BAR 단위 |

> *HMM/MSAR는 주기적 재학습 시 O(n). 일반 bar에서는 predict만 수행.

**일관성 보장:**

- 동일 데이터에 대해 vectorized와 incremental이 같은 레짐 라벨 생성
- Hysteresis 로직이 양쪽에서 동일하게 구현 (`apply_hysteresis` 공유)

---

## 14. 테스트 구조

### 14.1 단위 테스트

> 아래는 문서 시점 기준 목록입니다. 최신 테스트는 `tests/regime/` 디렉토리를 직접 확인하세요.

| 테스트 파일 | 대상 |
|------------|------|
| `tests/regime/test_detector.py` | RegimeDetector (vectorized + incremental) |
| `tests/regime/test_vol_detector.py` | VolStructureDetector |
| `tests/regime/test_hmm_detector.py` | HMMDetector |
| `tests/regime/test_msar_detector.py` | MSARDetector |
| `tests/regime/test_derivatives_detector.py` | DerivativesDetector (funding/OI/cascade risk) |
| `tests/regime/test_ensemble.py` | EnsembleRegimeDetector (앙상블 블렌딩 + confidence) |
| `tests/regime/test_regime_service.py` | RegimeService (precompute/incremental/warmup/transition/context/event) |

### 14.2 통합 테스트

| 테스트 파일 | 대상 |
|------------|------|
| `tests/eda/test_strategy_engine_regime.py` | StrategyEngine regime enrichment |

### 14.3 커버리지 요약

- Config: Pydantic 검증 (동결, 범위 체크, 가중치 합)
- Vectorized/Incremental 일관성
- Graceful degradation (라이브러리 미설치)
- EDA 통합 (precompute + fallback)
- Live warmup
- Confidence + Transition probability
- DerivativesDetector (cascade risk, funding/OI 분류)
- REGIME_CHANGE 이벤트 발행
- RegimeContext API (frozen, suggested_vol_scalar)
- Backward compatibility (derivatives_provider=None)

---

## 15. Backtest Advisor: Regime Analyzer

**파일:** `src/backtest/advisor/analyzers/regime.py`

백테스트 결과를 레짐별로 분리 분석하는 어드바이저 모듈:

- Trend regime 분류: Bull / Bear / Sideways
- Vol regime 분류: High Vol / Low Vol
- 레짐별 수익률, Sharpe, MDD 분리 계산
- 전략이 특정 레짐에서만 수익을 내는지 진단

---

## 16. 주의사항

1. **등록 순서:** RegimeService가 StrategyEngine보다 먼저 `register(bus)` 호출 필수
1. **Warmup:** 라이브 모드에서 `warmup()` 미호출 시 첫 N bar 동안 regime 없음
1. **Meta-learner Live 미지원:** forward return 필요 → weighted_average로 자동 폴백
1. **가중치 합:** `ensemble_method="weighted_average"` 시 활성 감지기 가중치 합 = 1.0 필수
1. **TF 필터:** RegimeService는 `target_timeframe`에 해당하는 BAR만 처리
1. **Derivatives 데이터 없이도 동작:** `derivatives_provider=None`이면 derivatives detector 비활성, cascade_risk=0.0
1. **REGIME_CHANGE 이벤트:** EventBus에 register() 호출 후에만 발행 (bus=None이면 스킵)
1. **Backtest cascade_risk=0.0:** derivatives 데이터 없는 backtest에서는 `cascade_risk` 컬럼이 전체 0.0. DerivativesDetector가 활성이어도 `deriv_df=None`이면 cascade_risk가 계산되지 않음

---

## 17. Future Enhancement: Regime 조건부 성과 추적

현재 시스템은 레짐 분류와 전환 확률을 실시간으로 계산하지만,
**레짐별 전략 성과**를 체계적으로 추적하는 메커니즘은 아직 없다.

### 17.1 목표

- 레짐별 Sharpe / Win Rate / PnL 실시간 집계
- "이 전략은 TRENDING에서만 수익" 패턴 자동 감지
- Regime-conditional position sizing 근거 제공

### 17.2 설계 스케치

```text
┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  FILL Event   │────▶│  RegimeTracker    │────▶│  Prometheus      │
│  (PnL, side)  │     │  (regime × asset) │     │  regime_pnl_total│
└──────────────┘     └───────────────────┘     │  regime_trades   │
                                                │  regime_sharpe   │
       ┌──────────────┐                         └──────────────────┘
       │ REGIME_CHANGE│────▶ label switch 기록          │
       └──────────────┘                                 ▼
                                                ┌──────────────────┐
                                                │  Grafana Panel   │
                                                │  - PnL by regime │
                                                │  - Drawdown curve│
                                                │  - Regime timeline│
                                                └──────────────────┘
```

### 17.3 Prometheus 메트릭 예시

| 메트릭 | Labels | 설명 |
|--------|--------|------|
| `regime_pnl_total` | `regime`, `asset`, `strategy` | 레짐별 누적 PnL |
| `regime_trade_count` | `regime`, `asset` | 레짐별 거래 횟수 |
| `regime_win_rate` | `regime`, `strategy` | 레짐별 승률 |
| `regime_duration_bars` | `regime` | 현재 레짐 지속 bar 수 |

### 17.4 구현 우선순위

1. **P1:** RegimeTracker 클래스 (FILL + REGIME_CHANGE 구독, in-memory 집계)
2. **P2:** Prometheus exporter 연동 (기존 `src/monitoring/metrics.py` 확장)
3. **P3:** Grafana 대시보드 템플릿 (JSON export)
4. **P4:** Regime-conditional sizing (RegimeContext.suggested_vol_scalar 확장)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-02-18 | 초기 문서 작성 -- 4개 감지기, 앙상블, RegimeService, EDA 통합 흐름 |
| 2026-02-18 | Phase 1-3: confidence, transition_prob, DerivativesDetector(5th), REGIME_CHANGE event, RegimeContext API |
| 2026-02-18 | 검증 및 보완: Parity 테스트, Edge case, Convergence tracking, Expanding-window transition, §17 Future Enhancement |
