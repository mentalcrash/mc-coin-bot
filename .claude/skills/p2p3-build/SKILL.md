---
name: p2p3-build
description: >
  Phase 2+3 통합 — 전략 4-file 구현 + 독립 감사(C1-C7).
  사용 시점: CANDIDATE 전략 구현, "implement" "build" 요청, p1 완료 후.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--from p3]
---

# p2p3-build: 전략 구현 + 독립 감사

> P2(구현) → 역할 전환 → P3(감사) → 판정을 **한 번의 호출**로 순차 실행한다.

---

## 인수

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `accel-conv` |
| `--from p3` | X | P2 건너뛰고 P3만 실행 (YAML P2 PASS 필수) | `--from p3` |

> kebab-case (e.g., `accel-conv`), 디렉토리명은 snake_case (e.g., `accel_conv`).

---

# Phase 2: 전략 코드 구현 (Step 0-9)

## 역할

**시니어 퀀트 개발자**로서 행동한다.
판단 기준: **"이 코드가 실제 돈을 운용하는 프로덕션 시스템에 배포되어도 안전한가?"**

핵심 원칙:

- **정확성 > 속도**: shift(1) 누락이 수백만 원 차이를 만든다
- **벡터화 필수**: pandas/numpy 벡터 연산 (루프 절대 금지)
- **일관된 패턴**: 기존 전략과 동일한 코드 구조 유지
- **테스트 우선**: 구현 → 테스트 → lint/typecheck/test 통과 → 등록

---

## Step 0: Pre-flight + 분기

### 0-0. `--from p3` 분기

`--from p3` 지정 시:

```bash
cat strategies/{strategy_name}.yaml
# phases.P2.status: PASS 확인
# P2 미완료 → "P2를 먼저 완료하세요" 중단
# P2 PASS → Step 10으로 직행
```

### 0-1. YAML에서 후보 로드

```bash
cat strategies/{strategy_name}.yaml
# meta.status=CANDIDATE, phases.P1.status=PASS 확인
# YAML 없으면: "p1-research에서 pipeline create를 먼저 실행하세요"
```

추출 항목: 전략명, 카테고리, TF, ShortMode, Phase 1 점수, 핵심 가설, 사용 지표, 시그널 로직, 차별화 포인트, 데이터 요구사항.

### 0-2. annualization_factor 결정

| TF | factor | TF | factor |
|----|--------|----|--------|
| 1D | 365.0 | 4H | 2190.0 |
| 12H | 730.0 | 1H | 8760.0 |
| 8H | 1095.0 | 6H | 1460.0 |

> **Critical**: annualization_factor 오류는 vol_scalar를 왜곡하여 포지션 사이징에 직접 영향.

### 0-3. 기존 전략과 중복 확인

```bash
uv run python -c "from src.strategy import list_strategies; print(list_strategies())"
```

### 0-4. 폐기 전략 패턴 확인

[references/implementation-checklist.md](references/implementation-checklist.md)의 anti-pattern 목록 참조.

---

## Step 1: 디렉토리 구조 생성

```bash
mkdir -p src/strategy/{name_snake}/ tests/strategy/{name_snake}/
touch tests/strategy/{name_snake}/__init__.py
```

**필수 파일**: config.py, preprocessor.py, signal.py, strategy.py, **init**.py + 테스트 4개.

---

## Step 2: config.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Config Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `from __future__ import annotations` | 상단 필수 |
| `BaseModel` + `frozen=True` | 불변 설정 |
| `ShortMode(IntEnum)` | DISABLED=0, HEDGE_ONLY=1, FULL=2 |
| 전략 파라미터 `Field()` | `ge`, `le` 검증 |
| `vol_target`, `min_volatility` | Vol-target 파라미터 |
| `annualization_factor` | TF별 연환산 (Step 0-2) |
| HEDGE_ONLY 파라미터 | `hedge_threshold`, `hedge_strength_ratio` |
| `model_validator` | `vol_target >= min_volatility` |
| `warmup_periods()` | warmup bar 수 |

**주의**: `from __future__ import annotations` 사용 시 `TYPE_CHECKING` import 활용.

### Step 2.5: 레짐 적응형 전략 (해당 시)

레짐 활용이 지정된 경우: [references/regime-implementation.md](references/regime-implementation.md) 참조.

- 접근 A: RegimeService 자동 주입 (권장, EDA live 호환)
- 접근 B: 자체 레짐 감지 (regime-tsmom 참고)

### Step 2.6: Derivatives 데이터 전략 (해당 시)

OHLCV 외 Derivatives 데이터 필요 시: [references/derivatives-implementation.md](references/derivatives-implementation.md) 참조.

### Step 2.7: On-chain / Multi-Source 데이터 전략 (해당 시)

On-chain, Macro, Options 데이터 활용 시: [references/onchain-implementation.md](references/onchain-implementation.md) 참조.

**핵심 패턴: Multi-Source Context Architecture (12H + 1D)**

```
12H OHLCV  →  가격 시그널 (빠른 진입/청산 판단)
     ↕          merge_asof(direction="backward")로 자동 병합
1D On-chain →  컨텍스트/확신도 가중 (느린 시장 상태 필터)
```

| 데이터 소스 | 해상도 | 에셋 범위 | 병합 방식 |
|-----------|:-----:|----------|---------|
| OHLCV | 전략 TF | 전체 | 기본 DataFrame |
| On-chain Global | 1D | 전체 에셋 | `merge_asof` + pub lag T+1 |
| On-chain BTC/ETH | 1D | BTC/ETH만 | `merge_asof` + pub lag T+1 |
| Derivatives | 5m→1h | 전체 | `merge_asof` |
| Macro | 1D | 글로벌 | `merge_asof` + pub lag T+1~14 |

**Graceful Degradation 필수**: `oc_*` 컬럼 부재 시 NaN → 중립(0) 처리. `required_columns`에 포함하지 않음.

---

## Step 3: preprocessor.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Preprocessor Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `preprocess(df, config) -> pd.DataFrame` | 모듈 레벨 함수 |
| `df = df.copy()` | 원본 불변 (첫 줄) |
| missing columns 검증 | `required = {"open", "high", "low", "close", "volume"}` |
| `returns` = `np.log(close / close.shift(1))` | log return |
| `realized_vol` = `returns.rolling(N).std() * np.sqrt(factor)` | 연환산 변동성 |
| `vol_scalar` = `target / realized_vol.clip(lower=min_vol)` | 포지션 스케일러 |
| 전략별 feature | 벡터화 연산만 |

**공유 지표 재사용**: `src/market/indicators/` (53개 함수). 중복 구현 금지.
`from src.market.indicators import atr, rsi, drawdown, ...`

**NaN 처리**: Rolling warmup NaN 유지, 0 나눗셈 `.clip(lower=epsilon)`, feature 간 NaN 전파 허용.

**금지 패턴**: 전체 기간 `.mean()/.std()` (rolling 필수), `inplace=True`, `apply(axis=1)`.

---

## Step 4: signal.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Signal Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `generate_signals(df, config) -> StrategySignals` | 모듈 레벨 함수 |
| **Shift(1) Rule** | 모든 feature에 `.shift(1)` — preprocessor가 아닌 signal에서 적용 |
| ShortMode 3-way 분기 | DISABLED / HEDGE_ONLY / FULL |
| `StrategySignals` 반환 | `(entries, exits, direction, strength)` |
| NaN → 0 처리 | strength `.fillna(0.0)`, direction NaN → 0 |

**Shift(1) 핵심**: preprocessor에서 계산한 feature를 signal에서 shift(1). preprocessor에서는 shift하지 않음.

**금지**: `shift(-N)` (미래 참조), 동시 long+short 미처리, strength NaN에서 entries=True.

### ML Rolling Training Loop — forward_return Look-Ahead Bias 방지 (Critical)

`forward_return[i] = (close[i + prediction_horizon] - close[i]) / close[i]`를 학습 타겟으로 사용하는 전략은
시점 t에서 **확정된 타겟만** 학습에 사용해야 한다.

```python
# 올바른 패턴 (resolved_end)
prediction_horizon = config.prediction_horizon
for t in range(loop_start, n):
    start_idx = t - training_window
    # 시점 t에서 close[j]는 j <= t만 알 수 있으므로,
    # forward_return[i]가 확정된 인덱스는 i <= t - prediction_horizon.
    resolved_end = t - prediction_horizon + 1  # exclusive upper bound
    if resolved_end <= start_idx:
        continue
    x_train = feature_matrix[start_idx:resolved_end]
    y_train = forward_returns[start_idx:resolved_end]

# 금지 패턴 (미확정 타겟 포함 — 교훈 #068)
for t in range(loop_start, n):
    x_train = feature_matrix[start_idx:t]
    y_train = forward_returns[start_idx:t]  # t-1..t-ph+1은 미확정!
```

**검증 필수**: `TestResolvedTargetOnly` — VBT(full data) vs expanding window 시그널 일치 테스트.
이 테스트가 없으면 P3 C1 FAIL 처리.

---

## Step 5: strategy.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Strategy Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `@register("kebab-name")` | Registry 등록 |
| `BaseStrategy` 상속 | 추상 메서드 구현 |
| `name`, `required_columns`, `config` | property |
| `preprocess()`, `generate_signals()` | 위임 |
| `recommended_config()` | PM 설정 권장값 |
| `from_params()` | Config → 인스턴스 factory |
| `get_startup_info()` | CLI 표시용 핵심 파라미터 |

---

## Step 6: **init**.py + Registry 등록

전략 모듈 `__init__.py`: Config, Strategy, ShortMode, preprocess, generate_signals export.

`src/strategy/__init__.py`에 알파벳 순 import 추가:

```python
from src.strategy import {name_snake}  # noqa: F401 — registry side-effect
```

등록 확인:

```bash
uv run python -c "from src.strategy import list_strategies; print('{name}' in list_strategies())"
```

---

## Step 7: 테스트 작성

전략당 4개 파일, 총 **40~60개 테스트**. 상세 테스트 클래스 목록: [references/implementation-checklist.md](references/implementation-checklist.md) Tests 섹션 참조.

**핵심 테스트 그룹**:

- **test_config.py**: ShortMode enum, 기본값, frozen, 경계값, 교차 검증, warmup
- **test_preprocessor.py**: 컬럼 존재, 길이, 불변, missing, feature 범위
- **test_signal.py**: 구조, shift(1) 첫 bar 중립, ShortMode 3종, 전략 고유 로직
- **test_strategy.py**: registry, properties, pipeline, from_params, recommended_config

레짐 적응형 전략은 `TestRegimeAdaptation` 추가 (with/without regime 컬럼).

**필수 Look-Ahead Bias 방어 테스트** (test_signal.py에 포함):

- **TestNoLookaheadBias**: 마지막 50 bar 제거 시 이전 시그널 불변 (truncation invariance)
- **TestPreprocessorImmutability**: preprocess() 호출 후 원본 df 변경 없음
- **TestResolvedTargetOnly** (ML 전략 한정): forward_return expanding window parity

> 3종 중 TestNoLookaheadBias + TestPreprocessorImmutability는 **모든 전략 필수**.
> TestResolvedTargetOnly는 forward_return/ML training loop 사용 시에만 추가.

---

## Step 8: 품질 검증 + 앙상블 호환

```bash
# Lint + Format
uv run ruff check src/strategy/{name_snake}/ tests/strategy/{name_snake}/ --fix
uv run ruff format src/strategy/{name_snake}/ tests/strategy/{name_snake}/

# Type check
uv run pyright src/strategy/{name_snake}/

# Tests
uv run pytest tests/strategy/{name_snake}/ -v
uv run pytest --tb=short -q  # 전체
```

**0 error 필수**. 최종 체크리스트: [references/implementation-checklist.md](references/implementation-checklist.md)

### 앙상블 호환성 확인

모든 `BaseStrategy` 구현체는 앙상블 자동 호환. 확인 사항:

1. `from_params()` 정상 작동
1. `warmup_periods()` 정의
1. direction 값 범위 {-1, 0, 1}

> 앙상블 편입 시 `config/ensemble-example.yaml`에 서브 전략 추가. 상관 < 0.4 + Sharpe > 0.3 기준.

---

## Step 9: P3 프리스캔 + P2 YAML 기록

구현 완료 후, P3 본감사 전에 C1-C7 핵심 패턴을 자동+수동 스캔한다.
이 단계에서 FAIL이면 Step 2-7로 돌아가 수정한다.

### 9-1. 자동 스캔

```bash
bash .claude/skills/p2p3-build/scripts/scan_strategy.sh src/strategy/{name_snake}
```

Critical/High 0건이어야 통과.

### 9-2. 수동 핵심 3종 확인

| # | 항목 | 검증 방법 |
|---|------|----------|
| C1 | Look-Ahead Bias | signal.py의 모든 feature에 shift(1) 적용 확인 |
| C5 | Position Sizing | min_volatility clip + annualization_factor 정확성 |
| C7 | Entry/Exit Logic | ShortMode 3-way 분기 + direction ∈ {-1,0,1} |

> C2-C4, C6은 코드 템플릿 준수 시 자동 충족. 의심 시 전항목 검증.

### 9-3. Parameter Stability Pre-check (권장)

핵심 threshold 파라미터를 ±20% 변동했을 때 direction 50%+ 변경되면 WARNING.

### 9-4. P2 YAML 갱신

```bash
# 1. P2 phase 결과 기록 (필수)
uv run mcbot pipeline record {strategy_name} --phase P2 --verdict PASS \
  --rationale "4-file 구조 구현 완료. Registry 등록 {strategy_name}. Ruff/Pyright 0 error." \
  -d "tests={N}" -d "registry={strategy_name}" -d "files=5" \
  -d "lookahead_tests=PASS" -d "prescan_critical=0"

# 2. status: CANDIDATE → IMPLEMENTED
uv run mcbot pipeline update-status {strategy_name} --status IMPLEMENTED

# 3. Dashboard 재생성
uv run mcbot pipeline report
```

> **Critical**: `pipeline record --phase P2`를 반드시 `update-status` 이전에 실행.
> 이 단계를 누락하면 phases에 P2가 빠지고 `pipeline table`에서 next=P2로 표시됨.

---

# === 역할 전환 ===

> **여기서부터 역할이 바뀐다.**
> P2: **시니어 퀀트 개발자** (구현자, Write/Edit 자유)
> P3: **독립 감사자** (의심 우선, Read/Grep만 사용, Step 13 자동수정만 예외)

---

# Phase 3: 전략 코드 독립 감사 (Step 10-13)

## 역할

**독립 감사자(External Auditor)**로서 행동한다.
P2에서 이미 프리스캔(C1,C5,C7)과 look-ahead bias 테스트를 통과한 전략에 대해,
**구현자와 다른 시각**에서 C1-C7 전항목을 재검증한다.

판단 기준: **"P2 구현자가 놓쳤을 수 있는 결함이 있는가?"**
기본값은 **의심** — 코드가 올바름을 증명할 때까지 결함 가정.

### P2 프리스캔과의 관계

P2 Step 9에서 scan_strategy.sh + 수동 C1/C5/C7 + look-ahead bias 테스트를 이미 수행.
P3는 이를 **재확인하되, 추가로**:

- C2(Data Leakage), C3(Survivorship), C4(Vectorization), C6(Cost Model) 심층 검증
- ML 전략의 training loop 상세 추적 (resolved_end 패턴)
- Warning W1-W7 체계적 기록
- 결함의 경제적 영향 추정 (금액/비율)

---

## 판정 기준

| 판정 | 조건 |
|------|------|
| **PASS** | Critical C1~C7 **모두** 결함 없음 |
| **FAIL** | Critical 중 **1개라도** 결함 발견 -> 수정 후 재검증 |

> Warning W1~W7는 기록하되 FAIL 사유가 아님.

---

## 프로젝트 전략 코드 구조

```
src/strategy/{name}/
  __init__.py        # 모듈 export
  config.py          # Pydantic BaseModel (frozen=True), 파라미터 정의
  preprocessor.py    # preprocess(df, config) — 지표 계산 (벡터화)
  signal.py          # generate_signals(df, config) — 시그널 생성
  strategy.py        # @register 데코레이터 + BaseStrategy 상속
```

### 핵심 규칙

| 규칙 | 설명 |
|------|------|
| **Shift(1) Rule** | 모든 시그널은 `shift(1)` 적용된 전봉 데이터 기반. 당일 Close/High/Low 직접 사용 금지 |
| **Stateless** | 전략은 시그널만 생성. 포지션/잔고/리스크는 PM/RM이 관리 |
| **Vectorized** | for 루프 금지, pandas/numpy 벡터 연산만 허용 |
| **StrategySignals** | `(entries, exits, direction, strength)` 4-tuple 반환 |
| **ShortMode** | DISABLED=0, HEDGE_ONLY=1, FULL=2 — 숏 모드별 분기 필수 |
| **frozen Config** | Pydantic `frozen=True`, `model_validator` 사용 |

---

## Step 10: P3 Pre-flight + 자동 스캔

```
1. 전략 디렉토리 구조 확인 (config, preprocessor, signal, strategy 파일 존재)
2. 테스트 파일 존재 확인: tests/strategy/{name}/
3. YAML 메타데이터 확인 (필수)
   cat strategies/{strategy_name}.yaml
   # meta.status가 IMPLEMENTED이고 phases.P2.status가 PASS인지 확인
   # YAML 없음 → "P2를 먼저 완료하세요"
   # P2 미통과 → "P2를 먼저 통과하세요"
4. 자동 스캔 실행: bash .claude/skills/p2p3-build/scripts/scan_strategy.sh src/strategy/{name}
```

---

## Step 11: C1-C7 Critical 검증

### [C1] Look-Ahead Bias (CRITICAL)

> **P2 참조**: scan_strategy.sh C1-a~C1-d + 수동 C1 체크 완료 상태.
> P3에서는 **간접 look-ahead**(shift 순서 오류, 레짐 컬럼 shift 누락 등) 집중 검증.

미래 데이터를 현재 시그널 생성에 사용하는가?

**검증 대상**: `preprocessor.py`, `signal.py`

| 패턴 | 판정 |
|------|------|
| `shift(-N)`, `pct_change(-N)` | FAIL — 미래 값 직접 참조 |
| `iloc[i+N]` (N>0) | FAIL — 미래 행 접근 |
| `.min()`, `.max()`, `.mean()`, `.std()` (rolling/expanding 없이) | FAIL — 전체 기간 통계 |
| `df["high"]`, `df["low"]` (shift 없이 시그널에 사용) | FAIL — 당봉 High/Low는 미확정 |
| `df["close"]` → 시그널 → **같은 봉** 체결 | FAIL — Signal[t] at Close -> Execute at Open[t+1] 위반 |

**필수 확인**: `signal.py`에서 시그널 생성에 사용되는 **모든** 컬럼이 `shift(1)` 이상 적용되었는지 한 줄씩 추적.

> 상세 코드 패턴: [references/critical-checklist.md](references/critical-checklist.md) C1절

### [C2] Data Leakage (CRITICAL)

IS/OOS 경계를 넘어 학습하는가?

**검증 대상**: `preprocessor.py`, `signal.py`

| 패턴 | 판정 |
|------|------|
| `scaler.fit(전체데이터)`, `fit_transform(전체)` | FAIL |
| `train_test_split` (무작위 분할) | FAIL — 시계열은 시간순 분할 필수 |
| rolling/expanding 윈도우가 OOS 데이터 포함 | FAIL |
| HMM/ML 모델의 `fit()`이 전체 데이터 사용 | FAIL |

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C2절

### [C3] Survivorship Bias (CRITICAL)

상폐/유동성 고갈 종목이 제외되었는가?

**검증 대상**: `config.py`, 전략 문서

| 패턴 | 판정 |
|------|------|
| 하드코딩된 에셋 리스트에 상폐 코인 포함 | FAIL |
| 백테스트 기간 전체에 걸쳐 데이터 존재 미확인 | FAIL |
| SOL 등 2020-01 이전 미상장 종목의 NaN 미처리 | WARNING |

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C3절

### [C4] Signal Vectorization (CRITICAL)

벡터 연산이 올바르게 적용되었는가?

**검증 대상**: `preprocessor.py`, `signal.py`

| 패턴 | 판정 |
|------|------|
| `for i in range(len(df))` — DataFrame 루프 | FAIL |
| pandas 인덱스 불일치 (다른 길이 Series 연산) | FAIL |
| `fillna(0)` 부적절 사용 (NaN이 의미 있는 경우) | FAIL |
| `iloc` vs `loc` 오용으로 인덱스 미스매치 | FAIL |
| `.values` 없이 numpy/pandas 혼합 연산 | WARNING |

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C4절

### [C5] Position Sizing (CRITICAL)

> **P2 참조**: annualization_factor + min_volatility 확인 완료 상태.
> P3에서는 **vol_scalar 공식 정확성 + strength 부호 일치** 집중 검증.

vol-target, leverage cap이 올바르게 적용되는가?

**검증 대상**: `preprocessor.py` (vol_scalar 계산), `signal.py` (strength 출력)

| 패턴 | 판정 |
|------|------|
| `target_vol / realized_vol` — 0 나눗셈 미방어 | FAIL |
| vol_scalar에 leverage cap(`max_leverage_cap`) 미적용 | FAIL |
| `min_volatility` 하한 미적용 | FAIL |
| strength 출력이 direction과 부호 불일치 | FAIL |
| annualization_factor 오류 (크립토: 365, 주식: 252) | FAIL |

**vol_scalar 정확성 공식 검증**:

```
realized_vol = returns.rolling(vol_lookback).std() * sqrt(annualization_factor)
realized_vol = max(realized_vol, min_volatility)
vol_scalar = target_vol / realized_vol
vol_scalar = clip(vol_scalar, 0, max_leverage_cap)  # PM에서 처리될 수도 있음
```

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C5절

### [C6] Cost Model (CRITICAL)

거래 비용이 누락/과소 적용되지 않는가?

**검증 대상**: `strategy.py` (recommended_config)

| 패턴 | 판정 |
|------|------|
| `CostModel` 미사용 또는 fee=0 | FAIL |
| 전략이 BacktestEngine 비용 파라미터를 우회 | FAIL |
| 고빈도(연 200회+) 전략에서 비용 미고려 시그널 | WARNING |

> 이 프로젝트에서 비용은 BacktestEngine/EDA 레벨에서 적용됨. 전략 코드 자체에서
> 비용을 하드코딩하거나 0으로 오버라이드하는지 확인.

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C6절

### [C7] Entry/Exit Logic (CRITICAL)

> **P2 참조**: ShortMode 3-way 분기 확인 완료 상태.
> P3에서는 **edge case**(동시 long+short, strength=NaN+entries=True 등) 집중 검증.

진입/청산 조건에 논리적 모순이 없는가?

**검증 대상**: `signal.py`

| 패턴 | 판정 |
|------|------|
| 동시 Long + Short 시그널 (동일 bar) | FAIL |
| entries=True + exits=True 동시 발생 처리 미정의 | FAIL |
| ShortMode별 분기 누락 (DISABLED일 때 숏 시그널 잔존) | FAIL |
| direction 값이 {-1, 0, 1} 범위 밖 | FAIL |
| strength가 NaN인 bar에서 entries=True | FAIL |
| hedge_threshold/hedge_strength_ratio 미적용 (HEDGE_ONLY 모드) | FAIL |

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C7절

---

## Step 12: Warning 항목 (W1-W7) + ML 심층

### W1-W7

FAIL 사유는 아니나 반드시 기록:

| ID | 항목 | 확인 내용 |
|----|------|----------|
| W1 | Warmup Period | `warmup_periods()` 반환값 >= 실제 필요 bar 수 |
| W2 | Parameter Count | Trades/Params > 10 (자유 파라미터 과다 방지) |
| W3 | Regime Concentration | 특정 레짐(2020-2021 상승장)에 수익 집중 가능성 |
| W4 | Turnover | 연간 회전율이 비용 대비 합리적인가 |
| W5 | Correlation | 기존 활성 전략과 수익률 상관 < 0.7 |
| W6 | Derivatives NaN | derivatives 컬럼 `ffill()` 처리 여부 |
| W7 | Shared Indicators | `src/market/indicators/` 54개 지표 중복 재구현 방지 |

> 상세: [references/warning-checklist.md](references/warning-checklist.md)

### ML 전략 심층 검증 (forward_return 사용 시에만)

| # | 항목 | 검증 |
|---|------|------|
| ML-1 | resolved_end 패턴 | `resolved_end = t - prediction_horizon + 1` 사용 확인 |
| ML-2 | Training window 경계 | `x_train = features[start:resolved_end]` (t가 아닌 resolved_end) |
| ML-3 | Feature engineering 시점 | 학습 feature에 미래 데이터 미사용 |
| ML-4 | Scaler fit 범위 | `scaler.fit(X_train[:resolved_end])` — 전체 fit 금지 |
| ML-5 | Expanding Window Parity Test | TestResolvedTargetOnly 존재 + PASS 확인 |

> ML-1~ML-5 중 1개 FAIL → Phase 3 FAIL.
> 교훈 #068: ML 전략 4개가 look-ahead bias 결함으로 전량 RETIRED.

---

## Step 13: 판정 + 자동수정 + 재감사

### 13-1. C1-C7 판정 집계

- **전부 PASS** → Step S
- **FAIL 있음** → 13-2로

### 13-2. 자동수정 분류

| 수정 유형 | 예시 | 자동수정 |
|----------|------|:-------:|
| **Mechanical Fix** | shift(1) 누락, fillna 추가, min_volatility clip | O |
| **Logic Fix** | 시그널 로직 오류, 설계 결함 | X → Step F |

### 13-3. Mechanical Fix 적용

> **예외적으로 Write/Edit 사용 허용** (감사자 역할의 유일한 예외).

- shift 누락 → shift(1) 추가
- NaN 방어 → fillna/clip 추가
- annualization_factor 오류 → 정확한 값으로 교체
- ShortMode 분기 누락 → 분기 추가

수정 후 lint/typecheck/test 재실행:

```bash
uv run ruff check src/strategy/{name_snake}/ tests/strategy/{name_snake}/ --fix
uv run ruff format src/strategy/{name_snake}/ tests/strategy/{name_snake}/
uv run pyright src/strategy/{name_snake}/
uv run pytest tests/strategy/{name_snake}/ -v
```

### 13-4. C1-C7 전체 재감사 (1회 한정)

자동수정 후 Step 11 C1-C7 **전체** 재검증.

- **재감사 PASS** → Step S
- **여전히 FAIL** → Step F

> 재감사는 **1회만** 허용. 2차 FAIL 시 무조건 Step F.

---

## Step S: PASS 처리

### S-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P3 --verdict PASS \
  --detail "C1=PASS" --detail "C2=PASS" --detail "C3=PASS" \
  --detail "C4=PASS" --detail "C5=PASS" --detail "C6=PASS" --detail "C7=PASS" \
  --detail "warnings={N}" \
  --rationale "C1-C7 전항목 PASS. Warning {N}건"
```

### S-2. Dashboard + 통합 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

```
============================================================
  P2+P3 BUILD REPORT
  전략: {DisplayName} ({registry-key})
  구현일: {YYYY-MM-DD} | TF: {TF} | ShortMode: {mode}
============================================================
  [P2] 파일: src/strategy/{name_snake}/ (5 files)
       테스트: tests/strategy/{name_snake}/ (4 files)
       Ruff: PASS | Pyright: PASS | Tests: PASS ({N} tests)
       Registry: {registry-key} ({total} strategies)
       프리스캔: PASS (Critical 0건)
------------------------------------------------------------
  [P3] 판정: PASS
       C1-C7: ALL PASS
       Warnings: {N}건 (상세 아래)
       자동수정: {있음/없음}
------------------------------------------------------------
  다음: /p4-backtest → /p5p6-validate → 앙상블 편입 검토
============================================================
```

---

## Step F: FAIL 처리

### F-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P3 --verdict FAIL --no-retire \
  --rationale "{C항목} FAIL: {사유}"
```

> `--no-retire`: P3 FAIL은 폐기가 아닌 **수정 후 재검증**이므로 auto-retire를 방지한다. status는 IMPLEMENTED 유지.

### F-2. 교훈 기록 (새로운 패턴 시)

**반복되는 결함 패턴**이 발견되면 교훈 데이터에 기록:

```bash
uv run mcbot pipeline lessons-list -c strategy-design
uv run mcbot pipeline lessons-add \
  --title "{결함 패턴 제목}" \
  --body "{구체적 설명 + 영향 + 방지법}" \
  -c strategy-design -s {strategy_name}
```

> 기존 교훈과 겹치면 추가하지 않는다.

### F-3. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

---

## Anti-Pattern 체크리스트

| # | Anti-Pattern |
|---|-------------|
| 1 | `shift(-N)` 사용 (미래 참조) |
| 2 | 전체 기간 `.mean()/.std()` (rolling 없이) |
| 3 | `for i in range(len(df))` |
| 4 | `iterrows()`, `itertuples()` |
| 5 | `apply(axis=1)` |
| 6 | `inplace=True` |
| 7 | `fillna(0)` 부적절 사용 |
| 8 | `except:` (광범위 예외) |
| 9 | 매직 넘버 (config 파라미터로 추출) |
| 10 | annualization_factor 하드코딩 |
| 11 | `# noqa`, `# type: ignore` 남용 |
| 12 | ML training loop에서 `forward_returns[start_idx:t]` (미확정 타겟 포함 — resolved_end 패턴 필수) |
| 13 | forward_return 사용 전략에 Expanding Window Parity Test 누락 |

---

## 심각도 기준

| 등급 | 정의 | 영향 |
|------|------|------|
| **CRITICAL** | 자금 손실 직접 유발 또는 백테스트 신뢰도 완전 훼손 | **FAIL** |
| **HIGH** | 백테스트 결과 과대/과소 평가 | **FAIL** (C1-C7 해당 시) |
| **MEDIUM** | 성과 왜곡 가능성 있으나 방향은 올바름 | 기록 |
| **LOW** | 코드 품질/유지보수 개선 | 기록 |

## 감사 원칙

1. **shift(1) 한 줄씩 검증** — signal.py에서 시그널 계산에 사용되는 **모든** 변수의 shift 여부를 개별 확인
1. **엣지 케이스 우선** — 급등, 급락, 유동성 고갈, NaN 구간, vol=0에서의 동작
1. **수정안 반드시 제시** — 문제 지적만으로 끝내지 않음. 구체적 코드 수정안 포함

---

## 참조 문서

| 문서 | 용도 |
|------|------|
| [references/code-templates.md](references/code-templates.md) | 4-file 코드 템플릿 |
| [references/implementation-checklist.md](references/implementation-checklist.md) | 구현 체크리스트 + 폐기 패턴 + 테스트 목록 |
| [references/regime-implementation.md](references/regime-implementation.md) | 레짐 적응형 구현 가이드 |
| [references/derivatives-implementation.md](references/derivatives-implementation.md) | Derivatives 데이터 구현 가이드 |
| [references/onchain-implementation.md](references/onchain-implementation.md) | On-chain / Multi-Source 데이터 구현 가이드 |
| [references/critical-checklist.md](references/critical-checklist.md) | C1-C7 상세 검증 패턴 |
| [references/warning-checklist.md](references/warning-checklist.md) | W1-W7 경고 체크리스트 |
| [references/report-template.md](references/report-template.md) | Phase 3 리포트 형식 |
| [.claude/rules/strategy.md](../../rules/strategy.md) | 전략 개발 규칙 |
| [src/strategy/base.py](../../../src/strategy/base.py) | BaseStrategy ABC |

---

## Phase Completion Protocol

Phase 완료 후 반드시 수행:

1. 현황 확인: `uv run mcbot pipeline next --name {strategy_name}`
1. 사용자에게 다음 Phase 진행 여부 질문:
   "P2/P3 결과: {요약}. 다음 Phase {next}로 진행하시겠습니까?"
1. 승인 시 다음 스킬 즉시 호출 (`pipeline next` 출력의 skill 명령 참조)
