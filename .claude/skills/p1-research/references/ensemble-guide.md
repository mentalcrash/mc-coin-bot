# Ensemble Strategy Design Guide

> 참조 코드: `src/strategy/ensemble/config.py`, `src/strategy/ensemble/aggregators.py`

---

## 1. Aggregation 선택 가이드

### 4개 방법 비교

| Method | 코드명 | 특성 | 적합 상황 | 추가 파라미터 |
|--------|--------|------|----------|-------------|
| Equal Weight | `equal_weight` | 가장 단순, 성과편차 작을 때 최적 | 서브전략 동급 성과 | 없음 |
| Inverse Volatility | `inverse_volatility` | 안정적 전략에 가중 부여 | 성과편차 크거나 안정성 중시 | `vol_lookback` (5~504, 기본 63) |
| Majority Vote | `majority_vote` | 방향 합의 중시, 강도 평균화 | 3+ 전략, 방향 신뢰도 중요 | `min_agreement` (0~1.0, 기본 0.5) |
| Strategy Momentum | `strategy_momentum` | 최근 성과 우수 전략 선택 | 4+ 전략, 레짐 적응 필요 | `momentum_lookback` (10~504, 기본 126), `top_n` (1~N) |

### 결정 플로우차트

```
서브전략 수?
├─ 2개
│  ├─ 성과 비슷 → Equal Weight
│  └─ 성과 편차 크다 → Inverse Volatility
├─ 3개
│  ├─ 성과편차 작음 → Equal Weight
│  ├─ 성과편차 큼 → Inverse Volatility
│  └─ 방향 합의 중요 → Majority Vote (min_agreement=0.67)
└─ 4개+
   ├─ 레짐 적응 원함 → Strategy Momentum
   └─ 안정적 가중 원함 → Inverse Volatility
```

### Aggregation 함수 시그니처

모든 aggregator는 동일 인터페이스:

```python
def aggregator(
    directions: pd.DataFrame,   # (n_bars, n_strategies) — -1/0/1
    strengths: pd.DataFrame,    # (n_bars, n_strategies) — 시그널 강도
    weights: pd.Series,         # (n_strategies,) — 정적 가중치
    **kwargs,                   # method별 추가 파라미터
) -> tuple[pd.Series, pd.Series]:  # (combined_direction, combined_strength)
```

---

## 2. EnsembleConfig 파라미터 가이드

### 필수 필드

| 필드 | 타입 | 제약 | 설명 |
|------|------|------|------|
| `strategies` | `tuple[SubStrategySpec, ...]` | `min_length=2` | 서브전략 목록 |
| `aggregation` | `AggregationMethod` | enum | 집계 방법 |

### SubStrategySpec

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | `str` | 필수 | Registry 등록명 (예: `"ctrend"`, `"tsmom"`) |
| `params` | `dict[str, Any]` | `{}` | `from_params(**params)` 전달 |
| `weight` | `float` | `1.0` | 정적 가중치 (`gt=0.0`) |

### 공통 파라미터

| 필드 | 범위 | 기본값 | 주의 |
|------|------|--------|------|
| `vol_target` | 0.05~1.0 | 0.35 | **이중 Vol Scaling 주의** (섹션 3) |
| `vol_window` | 5~252 | 30 | 변동성 계산 윈도우 |
| `annualization_factor` | >0 | 365.0 | 크립토 일봉 기준 |
| `short_mode` | 0 / 2 | 0 (DISABLED) | HEDGE_ONLY(1) 미지원 |

### model_validator 제약

- `aggregation == STRATEGY_MOMENTUM` → `top_n <= len(strategies)` 필수
- `frozen=True` → 생성 후 불변

---

## 3. 이중 Vol Scaling 함정

**문제**: 서브전략이 자체 `vol_target`을 적용하고, 앙상블이 다시 `vol_target`을 적용하면 시그널이 이중 축소된다.

```
서브전략 A: strength * (vol_target_A / realized_vol)  → 이미 vol-scaled
앙상블:     combined * (vol_target_ens / realized_vol) → 또 vol-scaled
결과:       실제 포지션 = 의도의 (vol_target_A / realized_vol)^2 배
```

**대응:**

1. 서브전략의 vol_target을 비활성화하고 앙상블에서만 적용 (권장)
1. 또는 앙상블 vol_target을 1.0으로 설정하여 raw signal 합산 후 PM에서 관리
1. 서브전략 `params`에서 vol_target 관련 파라미터를 명시적으로 조정

**점검 방법**: 앙상블 combined_strength의 범위가 [-1, 1] 이내인지 확인. 과도하게 작으면 이중 scaling 의심.

---

## 4. warmup_periods() 계산

```python
# EnsembleConfig.warmup_periods() 구현:
warmup = agg_lookback + vol_window + 1

# agg_lookback 결정:
#   equal_weight      → 0
#   inverse_volatility → vol_lookback (기본 63)
#   majority_vote     → 0
#   strategy_momentum → momentum_lookback (기본 126)
```

**전체 앙상블 warmup**:

```
total_warmup = max(sub_strategy_warmup for each sub) + ensemble_config.warmup_periods()
```

서브전략 warmup이 200이고 `strategy_momentum(lookback=126, vol_window=30)` 사용 시:

```
total = 200 + 126 + 30 + 1 = 357 bars
```

**주의**: 1D 기준 357일 = ~1년. 데이터 3년 미만이면 검증 구간 부족.

---

## 5. 앙상블 안티패턴 (AP1~AP8)

| # | 안티패턴 | 탐지 방법 | 대응 |
|---|---------|----------|------|
| AP1 | **동질성 함정** | 전원 동일 카테고리, 평균 상관 > 0.6 | 다른 카테고리 전략 교체/추가 |
| AP2 | **이중 Vol Scaling** | 서브전략+앙상블 모두 vol_target 적용 | 서브전략 vol_target 비활성 또는 앙상블 vol_target=1.0 |
| AP3 | **과적합 전략 세탁** | P4B(IS/OOS) FAIL 전략을 앙상블로 구제 | P4A PASS 이상만 서브전략 후보 |
| AP4 | **warmup 불일치** | total_warmup > 데이터 10% | 데이터 확보 또는 서브전략/aggregation 변경 |
| AP5 | **ShortMode 충돌** | 서브전략간 DISABLED/FULL 혼재 | 통일 (전원 DISABLED 또는 전원 FULL) |
| AP6 | **과다 서브전략** | N > 5 | 상관 높은 쌍 제거, 4개 이하 유지 |
| AP7 | **TF 불일치** | 서브전략간 TF 상이 (1D vs 4H) | 동일 TF 통일 (CandleAggregator는 전략 내부용) |
| AP8 | **백테스트 기간 불일치** | 최신 전략 데이터 < 3년 | 공통 기간으로 통일 또는 해당 전략 제외 |

### 점검 순서

```
1. AP1 → 카테고리 다양성 + 상관 행렬
2. AP2 → 서브전략 config에서 vol_target 확인
3. AP3 → pipeline table로 Phase 이력 확인
4. AP5 → 서브전략 short_mode 통일 여부
5. AP7 → 서브전략 TF 동일 확인
6. AP6 → 서브전략 수 확인
7. AP4 → warmup 계산
8. AP8 → 백테스트 기간 확인
```

---

## 6. YAML 등록: 단일 vs 메타 앙상블

| 항목 | 단일 전략 | 메타 앙상블 |
|------|----------|------------|
| 파일명 | `strategies/{name}.yaml` | `strategies/ens-{name}.yaml` |
| 등록 방법 | `pipeline create` CLI | 수동 작성 |
| `parameters.strategy_name` | `"{strategy-name}"` | `"ensemble"` |
| `parameters.sub_strategies` | 없음 | 필수 (리스트) |
| `meta.category` | 전략 유형 | `"메타 앙상블"` |
| Phase 1 | P1 (30점) | P1E (30점) |
| Phase 4 | P4A (Sharpe>1.0 기준) | P4E (Best 서브전략 초과 기준) |

---

## 7. YAML 예시

### Pipeline YAML (`strategies/ens-trend-vol.yaml`)

```yaml
name: ens-trend-vol
display_name: "Trend-Vol Ensemble"
status: CANDIDATE
category: "메타 앙상블"
timeframe: "1D"
short_mode: 0

rationale: |
  추세추종(CTREND) + 변동성구조(VoV-Mom) 결합.
  상관 0.15 → 분산 효과 극대화.
  Sharpe_ens ≈ sqrt(2) * 0.8 * sqrt(1-0.15) ≈ 1.04

parameters:
  strategy_name: ensemble
  sub_strategies:
    - name: "ctrend"
      params:
        lookback_long: 60
        lookback_short: 20
      weight: 1.0
    - name: "vov-mom"
      params:
        vov_window: 30
      weight: 1.0
  aggregation: "inverse_volatility"
  vol_lookback: 63
  vol_target: 0.35
  vol_window: 30
  short_mode: 0

phases:
  P1: {score: null, status: null, date: null}
  P1E: {score: 22, status: PASS, date: "2026-02-14"}

meta:
  category: "메타 앙상블"
  created: "2026-02-14"
  author: "claude"
```

### 백테스트 Config YAML

```yaml
strategy:
  name: ensemble
  config:
    strategies:
      - name: "ctrend"
        params:
          lookback_long: 60
          lookback_short: 20
        weight: 1.0
      - name: "vov-mom"
        params:
          vov_window: 30
        weight: 1.0
    aggregation: inverse_volatility
    vol_lookback: 63
    vol_target: 0.35
    vol_window: 30
    short_mode: 0

backtest:
  symbols:
    - BTC/USDT
    - ETH/USDT
  start: "2020-01-01"
  end: "2025-12-31"
  initial_capital: 10000
  commission: 0.001

portfolio:
  stop_loss_pct: 0.05
  trailing_stop_atr: 3.0
  rebalance_threshold: 0.10
```
