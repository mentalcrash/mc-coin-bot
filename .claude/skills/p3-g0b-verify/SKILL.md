---
name: p3-g0b-verify
description: >
  Gate 0B 전략 코드 검증 — C1~C7 Critical + W1~W7 Warning 판정.
  사용 시점: 구현 완료 후 백테스트 전, "검증" "verify" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
argument-hint: <strategy-name-or-directory>
---

# Gate 0B: 전략 코드 검증 (p3-g0b-verify)

## 역할

**비판적 시니어 퀀트 개발자**로서 행동한다.
판단 기준은 단 하나: **"이 코드로 실제 돈을 운용해도 되는가?"**

- 칭찬보다 **결함 탐지**를 우선한다
- 의심이 기본값 — 코드가 올바름을 **증명할 때까지** 결함 가정
- 모든 지적에 **구체적 수치와 수정안**을 제시한다
- **돈과 직결**되는 이슈는 절대 관대하게 넘기지 않는다

---

## 판정 기준 (gates/criteria.yaml 기준)

| 판정 | 조건 |
|------|------|
| **PASS** | Critical C1~C7 **모두** 결함 없음 |
| **FAIL** | Critical 중 **1개라도** 결함 발견 -> 수정 후 재검증 |

> Warning W1~W5는 기록하되 FAIL 사유가 아님.

---

## 프로젝트 전략 코드 구조

이 프로젝트의 전략은 아래 구조를 따른다. 검증 시 각 파일의 역할을 인지하고 분석한다.

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

## 검증 프로세스

### 0단계: 대상 파악

```
1. 전략 디렉토리 구조 확인 (config, preprocessor, signal, strategy 파일 존재)
2. 테스트 파일 존재 확인: tests/strategy/{name}/
3. YAML 메타데이터 확인 (필수)
   cat strategies/{strategy_name}.yaml
   # meta.status가 IMPLEMENTED이고 gates.G0A.status가 PASS인지 확인
   # YAML 없음 → "/p2-implement를 먼저 실행하세요"
   # G0A 미통과 → "p1-g0a-discover에서 G0A를 먼저 통과하세요"
4. 자동 스캔 실행: bash .claude/skills/p3-g0b-verify/scripts/scan_strategy.sh src/strategy/{name}
```

### 1단계: [C1] Look-Ahead Bias (CRITICAL)

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

### 2단계: [C2] Data Leakage (CRITICAL)

IS/OOS 경계를 넘어 학습하는가?

**검증 대상**: `preprocessor.py`, `signal.py`

| 패턴 | 판정 |
|------|------|
| `scaler.fit(전체데이터)`, `fit_transform(전체)` | FAIL |
| `train_test_split` (무작위 분할) | FAIL — 시계열은 시간순 분할 필수 |
| rolling/expanding 윈도우가 OOS 데이터 포함 | FAIL |
| HMM/ML 모델의 `fit()`이 전체 데이터 사용 | FAIL |

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C2절

### 3단계: [C3] Survivorship Bias (CRITICAL)

상폐/유동성 고갈 종목이 제외되었는가?

**검증 대상**: `config.py`, 전략 문서

| 패턴 | 판정 |
|------|------|
| 하드코딩된 에셋 리스트에 상폐 코인 포함 | FAIL |
| 백테스트 기간 전체에 걸쳐 데이터 존재 미확인 | FAIL |
| SOL 등 2020-01 이전 미상장 종목의 NaN 미처리 | WARNING |

> 상세: [references/critical-checklist.md](references/critical-checklist.md) C3절

### 4단계: [C4] Signal Vectorization (CRITICAL)

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

### 5단계: [C5] Position Sizing (CRITICAL)

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

### 6단계: [C6] Cost Model (CRITICAL)

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

### 7단계: [C7] Entry/Exit Logic (CRITICAL)

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

### 8단계: Warning 항목 (W1-W7)

FAIL 사유는 아니나 반드시 기록:

| ID | 항목 | 확인 내용 |
|----|------|----------|
| W1 | Warmup Period | `warmup_periods()` 반환값 >= 실제 필요 bar 수 |
| W2 | Parameter Count | Trades/Params > 10 (자유 파라미터 과다 방지) |
| W3 | Regime Concentration | 특정 레짐(2020-2021 상승장)에 수익 집중 가능성 |
| W4 | Turnover | 연간 회전율이 비용 대비 합리적인가 |
| W5 | Correlation | 기존 활성 전략과 수익률 상관 < 0.7 |
| W6 | Derivatives NaN | derivatives 컬럼 `ffill()` 처리 여부 |
| W7 | Shared Indicators | `src/market/indicators/` 53개 지표 중복 재구현 방지 |

> 상세: [references/warning-checklist.md](references/warning-checklist.md)

---

## 리포트 출력 형식

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

---

## 문서 갱신

검증 완료 후 **판정 결과에 따라** YAML 메타데이터를 갱신한다 (Single Source of Truth).

### YAML 갱신 (필수 — `strategies/{strategy_name}.yaml`)

**PASS 시**:

```bash
uv run mcbot pipeline record {strategy_name} \
  --gate G0B --verdict PASS \
  --detail "C1=PASS" --detail "C2=PASS" --detail "C3=PASS" \
  --detail "C4=PASS" --detail "C5=PASS" --detail "C6=PASS" --detail "C7=PASS" \
  --detail "warnings=0" \
  --rationale "C1-C7 전항목 PASS. Warning N건"
```

> `--detail key=value`로 C1-C7 개별 결과를 구조화하여 저장. YAML의 `gates.G0B.details`에 기록됨.

**FAIL 시**:

```bash
uv run mcbot pipeline record {strategy_name} \
  --gate G0B --verdict FAIL --no-retire \
  --rationale "{C항목} FAIL: {사유}"
```

> `--no-retire`: G0B FAIL은 폐기가 아닌 **수정 후 재검증**이므로 auto-retire를 방지한다. status는 IMPLEMENTED 유지.

### Dashboard 자동 생성

```bash
uv run mcbot pipeline report
```

---

## 교훈 기록 (FAIL 시)

**반복되는 결함 패턴**이 발견되면 교훈 데이터에 기록하여 향후 전략 개발에서 회피한다.

| 상황 | 교훈 기록 | 이유 |
|------|:--------:|------|
| 이전에 발견된 적 없는 새로운 결함 패턴 | O | 미래 방지 |
| 기존 교훈에 이미 포함된 패턴 | X | 중복 방지 |
| 전략 고유 버그 (일반화 불가) | X | 교훈이 아닌 버그 수정 |

새로운 결함 패턴 발견 시:

```bash
uv run mcbot pipeline lessons-add \
  --title "{결함 패턴 제목}" \
  --body "{구체적 설명 + 영향 + 방지법}" \
  -c strategy-design -s {strategy_name}
```

기존 교훈과 겹치면 추가하지 않는다. `uv run mcbot pipeline lessons-list -c strategy-design` 검색 후 판단.

---

## 심각도 기준

| 등급 | 정의 | Gate 0B 영향 |
|------|------|-------------|
| **CRITICAL** | 자금 손실 직접 유발 또는 백테스트 신뢰도 완전 훼손 | **FAIL** |
| **HIGH** | 백테스트 결과 과대/과소 평가 | **FAIL** (C1-C7 해당 시) |
| **MEDIUM** | 성과 왜곡 가능성 있으나 방향은 올바름 | 기록 |
| **LOW** | 코드 품질/유지보수 개선 | 기록 |

## 감사 원칙

1. **shift(1) 한 줄씩 검증** — signal.py에서 시그널 계산에 사용되는 **모든** 변수의 shift 여부를 개별 확인
2. **엣지 케이스 우선** — 급등, 급락, 유동성 고갈, NaN 구간, vol=0에서의 동작
3. **수정안 반드시 제시** — 문제 지적만으로 끝내지 않음. 구체적 코드 수정안 포함
