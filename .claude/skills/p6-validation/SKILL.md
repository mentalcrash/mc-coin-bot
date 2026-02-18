---
name: p6-validation
description: >
  Phase 6 심층검증 — WFA + CPCV + PBO + DSR + Monte Carlo.
  사용 시점: P5 PASS 전략의 최종 검증, "validate" "WFA" "CPCV" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name>
---

# Phase 6: 심층검증 (WFA + CPCV + PBO + DSR)

## 역할

**시니어 퀀트 검증 엔지니어**로서 행동한다 — 과적합 검증 + 통계적 유의성 전문.

핵심 원칙:

- 단순 threshold 비교가 아닌 **경제적 의미 해석** — 숫자 뒤의 이유를 찾는다
- FAIL 시 **구체적 사유 + 수정 방향** 제시 (단순 "FAIL" 판정 금지)
- 모든 결과를 **CTREND 선례**와 비교하여 상대적 위치 파악

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ac-regime` |

---

## Step 0: Pre-flight Check

다음 항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. YAML 메타데이터 + P5 PASS

```bash
cat strategies/{strategy_name}.yaml  # phases.P5B.status: PASS 확인
```

P5B PASS 없으면 `/p5-robustness` 먼저 실행.

### 0-2. P4/P5 결과 복원

YAML에서 이전 Phase 결과를 복원한다:

| 복원 대상 | 설명 |
|----------|------|
| Best Asset | P4A에서 결정된 최적 에셋 |
| IS/OOS Sharpe | P4B IS/OOS 결과 |
| Decay | P4B Decay (Phase 6 WFA Decay와 일관성 비교용) |
| 파라미터 안정성 | P5B 결과 (plateau 유무, +/-20% 안정성) |

### 0-3. Silver 데이터 존재

```bash
ls data/silver/BTC_USDT_1D.parquet data/silver/ETH_USDT_1D.parquet \
   data/silver/BNB_USDT_1D.parquet data/silver/SOL_USDT_1D.parquet \
   data/silver/DOGE_USDT_1D.parquet
```

---

## Step 1: Phase 6A — WFA (Walk-Forward Analysis)

### 실행

Best Asset에 대해 WFA (3-fold expanding window) 실행:

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m milestone -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| WFA OOS Sharpe (평균) | 전 fold OOS Sharpe 평균 |
| WFA Decay (%) | `(1 - WFA_OOS_Sharpe / WFA_IS_Sharpe) x 100` |
| WFA Consistency | OOS fold 양수 비율 |
| Fold별 IS/OOS | 각 fold의 IS Sharpe, OOS Sharpe, 기간 |

### 판정 기준

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| WFA OOS Sharpe | >= 0.5 | 1.49 |
| WFA Decay | < 40% | 39% |
| WFA Consistency | >= 60% | 67% |

### 퀀트 해석

- WFA Decay vs P4B Decay 일관성 확인 (차이 > 20%p이면 CV 방법론 민감도 경고)
- 최근 fold OOS가 평균보다 낮으면 시장 적응 문제 경고
- P4B OOS Sharpe 양수인데 WFA OOS 음수이면 IS 구간 의존 경고
- Reference: ../p4-backtest/references/quant-interpretation-guide.md

### YAML 갱신 (Phase 6A)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P6A --verdict PASS \
  --detail "wfa_oos_sharpe={X.XX}" --detail "wfa_decay={XX}%" \
  --rationale "WFA OOS Sharpe X.XX, Decay XX%, Consistency XX%"
```

**FAIL -> Step F** | **PASS -> Step 2**

---

## Step 2: Phase 6B — CPCV + PBO + DSR + Monte Carlo

### 실행

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m final -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| CPCV 평균 OOS Sharpe | 전 fold 평균 |
| PBO (%) | Probability of Backtest Overfitting |
| DSR (batch) | Deflated Sharpe Ratio (동일 배치 기준) |
| DSR (all) | 전체 기준 DSR |
| MC p-value | Monte Carlo 통계적 유의성 |
| MC 95% CI | Monte Carlo 95% 신뢰구간 |

### 판정 기준

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| PBO | 이중 경로 (아래 참조) | 60% (경로 B PASS) |
| DSR (batch) | > 0.95 | 1.00 |
| MC p-value | < 0.05 | 0.000 |

### PBO 이중 경로

PBO는 다음 **두 경로 중 하나**를 충족하면 PASS:

| 경로 | 조건 | 설명 |
|:----:|------|------|
| **A** | PBO < 40% | 기본 경로 -- 과적합 위험 낮음 |
| **B** | PBO < 80% AND CPCV 전 fold OOS > 0 AND MC p < 0.05 | 보조 경로 -- 파라미터 순위 역전이 있으나 기저 alpha 견고 |

**근거**: PBO는 "IS 최적 파라미터가 OOS에서도 최적 순위를 유지하는가"를 측정한다.
그러나 실전에서 중요한 것은 **"어떤 파라미터를 골라도 OOS에서 수익이 나는가"** (CPCV robustness).
경로 B는 파라미터 과적합이 있지만, 기저 전략 alpha가 견고한 경우를 구제한다.

**적용 예시**: CTREND (PBO 60%, 전 fold OOS 양수, MC p=0.000) -> 경로 B PASS.
Anchor-Mom (PBO 80%, 전 fold OOS 양수, MC p=0.000) -> 경로 B PASS (PBO < 80% 충족).

### YAML 갱신 (Phase 6B)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P6B --verdict PASS \
  --detail "pbo={XX}%" --detail "dsr_batch={X.XX}" --detail "mc_pvalue={X.XXX}" \
  --rationale "CPCV/PBO/DSR/MC 모두 PASS"
```

**FAIL -> Step F** | **PASS -> Step S**

---

## 판정 기준 종합

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| WFA OOS Sharpe | >= 0.5 | 1.49 |
| WFA Decay | < 40% | 39% |
| WFA Consistency | >= 60% | 67% |
| PBO | 이중 경로 | 60% (경로 B PASS) |
| DSR (batch) | > 0.95 | 1.00 |
| MC p-value | < 0.05 | 0.000 |

---

## Phase 간 일관성 체크

파이프라인 진행 중 아래 일관성을 확인한다:

| 비교 | 기대 | 경고 조건 |
|------|------|----------|
| P4B Decay -> P6A WFA Decay | 유사 (+-15%p) | 차이 > 20%p이면 CV 방법론 민감도 |
| P4B OOS -> P6A WFA OOS | WFA >= P4B x 0.5 | WFA < P4B x 0.5이면 window 의존 |
| P5B Sharpe -> P6B MC CI | P5B baseline 포함 CI | P5B baseline < CI 하한이면 불안정 |

---

## Step F: 실패 처리

Phase FAIL 시 다음을 순차 실행한다.

### F-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P6 --verdict FAIL \
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
| WFA Decay > 40% | `pipeline-process` |
| PBO > 80% (경로 A+B 모두 FAIL) | `pipeline-process` |
| DSR < 0.95 | `pipeline-process` |
| MC p-value > 0.05 | `strategy-design` |
| WFA Consistency < 60% | `market-structure` |

> 기존 교훈과 겹치면 추가하지 않는다.

### F-3. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

---

## Step S: 성공 처리 (Phase 6 PASS)

### S-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P6 --verdict PASS \
  --detail "wfa_oos_sharpe={X.XX}" --detail "pbo={XX}" \
  --rationale "WFA/CPCV/PBO/DSR 모두 PASS"
```

상태 `TESTING` 유지 (Phase 7 EDA Parity 대기).
Phase 7 검증: **2년** (2024-01-01 ~ 2025-12-31), 구현 정합성 검증이므로 6년 불필요.

### S-2. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

### S-3. 다음 단계

Next: `/p7-live` (Phase 7: EDA Parity 검증)

---

## CTREND 참조 벤치마크

| Phase | 핵심 지표 | CTREND 결과 | 판정 |
|:-----:|----------|------------|:----:|
| P6A | WFA OOS Sharpe | 1.49 | PASS |
| P6A | WFA Decay | 39% | PASS |
| P6A | WFA Consistency | 67% | PASS |
| P6B | PBO | 60% | PASS (경로 B) |
| P6B | DSR (batch) | 1.00 | PASS |
| P6B | MC p-value | 0.000 | PASS |

> CTREND은 PBO 60%로 경로 A(< 40%) FAIL이나, 전 CPCV fold OOS 양수 + MC p=0.000으로
> 경로 B를 통해 PASS. Anchor-Mom도 PBO 80%이나 동일 경로 B PASS.

---

## 참조 문서

- [../p4-backtest/references/phase-criteria.md](../p4-backtest/references/phase-criteria.md) -- Phase별 정량 기준 + CTREND 비교 + CLI 명령
- [../p4-backtest/references/quant-interpretation-guide.md](../p4-backtest/references/quant-interpretation-guide.md) -- 시니어 퀀트 해석 패턴
