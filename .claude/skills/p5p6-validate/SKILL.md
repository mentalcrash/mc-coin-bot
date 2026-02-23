---
name: p5p6-validate
description: >
  Phase 5+6 통합 검증 — 파라미터 최적화/안정성 + WFA/CPCV/PBO/DSR/MC.
  사용 시점: P4 PASS 전략의 로버스트니스 + 심층검증, "robustness" "validate" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--from p5a|p5b|p6a|p6b]
---

# Phase 5+6: 파라미터 최적화 + 심층검증

## 역할

**시니어 퀀트 리서처 -- 파라미터 로버스트니스 + 과적합 검증 전문**으로서 행동한다.

핵심 원칙:

- 단순 threshold 비교가 아닌 **경제적 의미 해석** -- 숫자 뒤의 이유를 찾는다
- FAIL 시 **구체적 사유 + 수정 방향** 제시 (단순 "FAIL" 판정 금지)
- Phase 간 **결과 일관성 추적** -- P4 Sharpe -> P5A IS Sharpe -> P5B Plateau -> P6 WFA 흐름
- Phase 간 **결과 일관성 추적** — 숫자의 절대값보다 Phase 간 감쇠 패턴이 중요

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ac-regime` |
| `--from p5a\|p5b\|p6a\|p6b` | X | 시작 단계 (기본: p5a) | `--from p6a` |

---

## Step 0: Pre-flight Check

다음 항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. YAML 메타데이터 + P4B PASS

```bash
cat strategies/{strategy_name}.yaml  # phases.P4B.status: PASS 확인
```

P4B PASS 없으면 `/p4-backtest` 먼저 실행.

### 0-2. P4 결과 복원

YAML에서 P4 결과 복원:

| 항목 | 소스 |
|------|------|
| Best Asset | P4A 결과 |
| Sharpe | P4A Best Sharpe |
| CAGR | P4A Best CAGR |
| IS/OOS Sharpe | P4B 결과 |
| Decay | P4B Decay |

### 0-3. --from 복원 (해당 시)

| --from | 복원 대상 | 전제 |
|--------|----------|------|
| `p5b` | P5A 결과. `results/phase5_opt_{strategy}.json` 존재 확인 | P5A PASS |
| `p6a` | P5A + P5B 결과. YAML `P5B.status: PASS` 확인 | P5B PASS |
| `p6b` | P5A + P5B + P6A 결과. YAML `P6A.status: PASS` 확인 | P6A PASS |

### 0-4. Silver 데이터 존재

```bash
ls data/silver/BTC_USDT_1D.parquet data/silver/ETH_USDT_1D.parquet \
   data/silver/BNB_USDT_1D.parquet data/silver/SOL_USDT_1D.parquet \
   data/silver/DOGE_USDT_1D.parquet data/silver/LINK_USDT_1D.parquet \
   data/silver/ADA_USDT_1D.parquet data/silver/AVAX_USDT_1D.parquet
```

### 0-5. Phase 5B 스윕 소스 확인

Phase 5B 스윕 범위는 두 소스 중 하나에서 제공:

1. **P5A 자동 생성 (권장)**: `results/phase5_opt_{strategy}.json`의 `sweep_ranges`
1. **수동 등록 (fallback)**: `src/cli/_phase_runners.py`의 `P5_STRATEGIES` dict

P5A를 실행하면 수동 등록 불필요. P5A 건너뛴 경우에만 수동 등록 (사용자 승인).

---

## Step 1: Phase 5A -- 파라미터 최적화 (정보 전용)

> **Always PASS** -- 정보 제공 목적. 과적합 방어는 P6에서 담당.

### 목적

1. 기본 파라미터 대비 Optuna TPE 최적화로 개선 가능성 확인
1. 최적 파라미터 중심으로 Phase 5B sweep 범위 자동 생성 (수동 등록 불필요)
1. IS/OOS 분할 검증으로 과적합 경향 사전 파악

### 실행

Best Asset (P4 결과)에 대해 IS(70%)/OOS(30%) 분할 후 최적화:

```bash
uv run mcbot pipeline phase5-run {strategy_name} --n-trials 100 --seed 42 --json
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| Default Sharpe (IS) | 기본 파라미터의 IS Sharpe |
| Best Sharpe (IS) | 최적 파라미터의 IS Sharpe |
| Improvement (%) | `(Best - Default) / |Default| x 100` |
| OOS Sharpe | 최적 파라미터의 OOS 검증 Sharpe (정보 전용) |
| Search Space | 최적화 대상 파라미터 수 |
| Trials | 완료/전체 trial 수 |

### 결과 해석

| Improvement | 해석 |
|-------------|------|
| < 5% | 기본 파라미터가 근최적 -- 민감도 낮음 (좋은 신호) |
| 5~30% | 합리적 개선 -- 정상 범위 |
| > 30% | IS 과적합 가능성. OOS Sharpe 확인 필수 |

OOS Sharpe 확인:

- OOS > 0, IS 대비 Decay < 50%: 양호
- OOS <= 0: IS 과적합 의심 -- P5B/P6에서 추가 검증 (P5A 자체는 PASS)

### 출력

| 파일 | 내용 |
|------|------|
| `results/phase5_opt_{strategy}.json` | 최적화 결과 + Phase 5B sweep 범위 + top 10 trials |
| YAML `phases.P5A` | PASS + IS/OOS Sharpe + improvement (CLI 자동 갱신) |

> YAML 갱신은 P5A CLI가 자동 처리 -- 별도 `pipeline record` 불필요.

**Always PASS -> Step 2**

---

## Step 2: Phase 5B -- 파라미터 안정성

### 사전 작업: 스윕 소스 확인

**P5A 결과 우선**: `results/phase5_opt_{strategy_name}.json` 존재 확인.

- **존재** (P5A 실행됨): Phase 5B CLI가 자동으로 P5A sweep 범위 사용. 수동 등록 불필요.
- **미존재** (P5A 건너뜀): `src/cli/_phase_runners.py`의 `P5_STRATEGIES` dict에 수동 등록 (사용자 승인).
  핵심 파라미터 3~5개, +/-20% + 넓은 범위 그리드. vol_target 필수, short_mode 제외.

### 실행

```bash
uv run mcbot pipeline phase5-stability {strategy_name} --json
```

> P5A JSON 존재 시 최적화 sweep 범위 자동 사용. 미존재 시 P5_STRATEGIES fallback.

### 수집 지표 (파라미터별)

| 지표 | 설명 |
|------|------|
| Plateau 존재 | Best Sharpe의 70% 이상인 값 >= 2개 |
| Plateau Count / Range | 고원 값 수 / 범위 (min~max) |
| +/-20% Stable / Sharpe | 기본값 +/-20%에서 Sharpe 양수 유지 / min~max |

### 판정 기준

| 조건 | 기준 |
|------|------|
| 고원 존재 | **모든** 핵심 파라미터에서 60%+ (plateau_count/total >= 60%) |
| +/-20% 안정 | **모든** 핵심 파라미터에서 Sharpe 부호 양수 유지 |

**전체 PASS**: 모든 파라미터가 (고원 존재 AND +/-20% 안정)

### 퀀트 해석

[references/quant-interpretation-guide.md](../p4-backtest/references/quant-interpretation-guide.md) 참조.
핵심: 넓은 고원 = 로버스트, 좁은 봉우리 = 과적합 위험. vol_target Sharpe 불변은 정상 (레버리지 스케일링).

판정 기준: [references/phase-criteria.md](../p4-backtest/references/phase-criteria.md) 참조.

**FAIL -> Step F** | **PASS -> Step 3**

---

## Step 3: Phase 6A -- WFA (Walk-Forward Analysis)

### 실행

Best Asset에 대해 WFA (5-fold expanding window) 실행:

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m milestone -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| WFA OOS Sharpe (평균) | 전 fold OOS Sharpe 평균 |
| WFA Decay (%) | `(1 - WFA_OOS_Sharpe / WFA_IS_Sharpe) x 100` |
| WFA Consistency | OOS fold 양수 비율 |
| Fold별 IS/OOS | 각 fold의 IS Sharpe, OOS Sharpe, 기간 |

### 판정 기준

| 조건 | 기준 |
|------|------|
| WFA OOS Sharpe | >= 0.3 |
| WFA Decay | < 50% |
| WFA Consistency | >= 50% |

### 퀀트 해석

- WFA Decay vs P4B Decay 일관성 확인 (차이 > 20%p이면 CV 방법론 민감도 경고)
- 최근 fold OOS가 평균보다 낮으면 시장 적응 문제 경고
- P4B OOS Sharpe 양수인데 WFA OOS 음수이면 IS 구간 의존 경고
- Reference: [references/quant-interpretation-guide.md](../p4-backtest/references/quant-interpretation-guide.md)

### YAML 갱신 (Phase 6A)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P6A --verdict PASS \
  --detail "wfa_oos_sharpe={X.XX}" --detail "wfa_decay={XX}%" \
  --rationale "WFA OOS Sharpe X.XX, Decay XX%, Consistency XX%"
```

**FAIL -> Step F** | **PASS -> Step 4**

---

## Step 4: Phase 6B -- CPCV + PBO + DSR + Monte Carlo

### 실행

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m final -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| CPCV 평균 OOS Sharpe | 전 fold 평균 |
| PBO (%) | Probability of Backtest Overfitting |
| DSR | Deflated Sharpe Ratio |
| MC p-value | Monte Carlo 통계적 유의성 |
| MC 95% CI | Monte Carlo 95% 신뢰구간 |

### 판정 기준: 3단계 Triage (PASS / WATCH / FAIL)

**PASS** (OOS Sharpe >= 0.3 AND 보충 2/4 이상):

| 조건 | 기준 |
|------|------|
| CPCV OOS Sharpe | >= 0.3 |
| 보충 검증 | >= 2/4 통과 |

**WATCH** (salvageable — TESTING 유지):

| 조건 | 기준 |
|------|------|
| 경로 A | OOS Sharpe >= 0.2 AND 보충 >= 1/4 |
| 경로 B | OOS Sharpe >= 0.3 AND 보충 == 1/4 |

WATCH 시 `improvement_hints`에 미통과 항목별 개선 방향 기록.

**FAIL** (terminal — RETIRED):

- OOS Sharpe < 0.2 OR 보충 == 0/4

**보충 4개 항목**:

| # | 조건 | 기준 |
|---|------|------|
| 1 | Sharpe Decay | <= 50% |
| 2 | PBO | 이중 경로 (아래 참조) |
| 3 | DSR | > 0.5 |
| 4 | MC p-value | < 0.10 |

### PBO 이중 경로

PBO는 다음 **두 경로 중 하나**를 충족하면 PASS:

| 경로 | 조건 | 설명 |
|:----:|------|------|
| **A** | PBO < 40% | 기본 경로 -- 과적합 위험 낮음 |
| **B** | PBO < 80% AND CPCV 전 fold OOS > 0 AND MC p < 0.05 | 보조 경로 -- 파라미터 순위 역전이 있으나 기저 alpha 견고 |

**근거**: PBO는 "IS 최적 파라미터가 OOS에서도 최적 순위를 유지하는가"를 측정한다.
그러나 실전에서 중요한 것은 **"어떤 파라미터를 골라도 OOS에서 수익이 나는가"** (CPCV robustness).
경로 B는 파라미터 과적합이 있지만, 기저 전략 alpha가 견고한 경우를 구제한다.

**적용 예시**: Anchor-Mom (PBO 80%, 전 fold OOS 양수, MC p=0.000) -> 경로 B PASS (PBO < 80% 충족).

**FAIL -> Step F** | **WATCH -> TESTING 유지 (재시도)** | **PASS -> Step S**

---

## Step F: 실패 처리

Phase FAIL 시 다음을 순차 실행한다.

### F-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase {P5B|P6A|P6B} --verdict FAIL \
  --rationale "{구체적 FAIL 사유}"
```

> `pipeline record`가 FAIL 시 자동으로 status -> RETIRED 변경.

### F-2. 교훈 기록 (새로운 패턴 시)

기존 교훈 검색 후, 새로운 실패 패턴이면 추가:

```bash
uv run mcbot pipeline lessons-list -c {category} -t {관련키워드}
uv run mcbot pipeline lessons-add \
  --title "{패턴 한줄 요약}" --body "{구체적 사유}" \
  -c {category} -t {태그1} -t {태그2} -s {strategy_name} --tf {TF}
```

| FAIL 유형 | 카테고리 |
|----------|----------|
| 파라미터 고원 부재 | `strategy-design` |
| +/-20% 범위 Sharpe 음수 | `strategy-design` |
| IS 과적합 (Improvement > 30% + OOS <= 0) | `pipeline-process` |
| WFA Decay > 50% | `pipeline-process` |
| PBO > 80% (경로 A+B 모두 FAIL) | `pipeline-process` |
| DSR < 0.5 | `pipeline-process` |
| MC p-value > 0.10 | `strategy-design` |
| WFA Consistency < 50% | `market-structure` |

> 기존 교훈과 겹치면 추가하지 않는다.

### F-3. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](../p4-backtest/references/report-template.md) 참조.

---

## Step S: 성공 처리 (Phase 6B PASS)

### S-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P6B --verdict PASS \
  --detail "wfa_oos_sharpe={X.XX}" --detail "pbo={XX}" \
  --detail "dsr={X.XX}" --detail "mc_pvalue={X.XXX}" \
  --rationale "P5+P6 전 단계 PASS — WFA/CPCV/PBO/DSR/MC 검증 완료"
```

상태 `TESTING` 유지 (Phase 7 EDA Parity 대기).

### S-2. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](../p4-backtest/references/report-template.md) 참조.

### S-3. 다음 단계

Next: `/p7-live` (Phase 7: EDA Parity 검증)

---

## Phase 간 일관성 체크

파이프라인 진행 중 아래 일관성을 확인한다:

| 비교 | 기대 | 경고 조건 |
|------|------|----------|
| P4A Sharpe -> P4B OOS Sharpe | OOS >= P4A x 0.3 | OOS < P4A x 0.3이면 과적합 의심 |
| P4B Decay -> P6A WFA Decay | 유사 (+-15%p) | 차이 > 20%p이면 CV 방법론 민감도 |
| P5A OOS -> P6A WFA OOS | 유사 방향 | P5A OOS 양수인데 P6A OOS 음수이면 IS 구간 의존 |
| P4B OOS -> P6A WFA OOS | WFA >= P4B x 0.5 | WFA < P4B x 0.5이면 window 의존 |
| P5B Sharpe -> P6B MC CI | P5B baseline 포함 CI | P5B baseline < CI 하한이면 불안정 |

---

## 참조 문서

- [references/phase-criteria.md](../p4-backtest/references/phase-criteria.md) -- Phase별 정량 기준 + CLI 명령
- [references/quant-interpretation-guide.md](../p4-backtest/references/quant-interpretation-guide.md) -- 시니어 퀀트 해석 패턴
- [references/report-template.md](../p4-backtest/references/report-template.md) -- 리포트 출력 형식
- `pipeline report` -- 전략 상황판 (CLI)
