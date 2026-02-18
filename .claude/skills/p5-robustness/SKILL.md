---
name: p5-robustness
description: >
  Phase 5 파라미터 최적화 + 안정성 검증 (Optuna TPE + 고원 분석).
  사용 시점: P4 PASS 전략의 로버스트니스 검증, "robustness" "optimize" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--from p5a|p5b]
---

# Phase 5: 파라미터 최적화 + 안정성 검증

## 역할

**시니어 퀀트 리서처 -- 파라미터 로버스트니스 전문**으로서 행동한다.

핵심 원칙:

- 단순 threshold 비교가 아닌 **경제적 의미 해석** -- 숫자 뒤의 이유를 찾는다
- FAIL 시 **구체적 사유 + 수정 방향** 제시 (단순 "FAIL" 판정 금지)
- Phase 간 **결과 일관성 추적** -- P4 Sharpe -> P5A IS Sharpe -> P5B Plateau 흐름 확인
- 모든 결과를 **CTREND 선례**와 비교하여 상대적 위치 파악

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ac-regime` |
| `--from p5a\|p5b` | X | 시작 단계 (기본: p5a) | `--from p5b` |

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

### 0-3. --from 복원 (해당 시)

`--from p5b` 지정 시 YAML에서 P5A 결과 복원:

| --from | 복원 대상 |
|--------|----------|
| `p5b` | P5A 결과. `results/phase5_opt_{strategy}.json` 존재 확인 |

### 0-4. Phase 5B 스윕 소스 확인

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
- OOS <= 0: IS 과적합 의심 -- Phase 5B/P6에서 추가 검증 (P5A 자체는 PASS)

### 출력

| 파일 | 내용 |
|------|------|
| `results/phase5_opt_{strategy}.json` | 최적화 결과 + Phase 5B sweep 범위 + top 10 trials |
| YAML `phases.P5A` | PASS + IS/OOS Sharpe + improvement (CLI 자동 갱신) |

### Phase 5B 연계

P5A CLI가 `results/phase5_opt_{strategy}.json`에 `sweep_ranges`를 자동 생성.
Phase 5B CLI(`pipeline phase5-stability`)가 이 파일을 자동 감지하여 sweep 범위로 사용.
**P5A 실행 후에는 Phase 5B 수동 스윕 등록 불필요.**

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
| Plateau 존재 | Best Sharpe의 80% 이상인 값 >= 3개 |
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

CTREND 비교: [references/phase-criteria.md](../p4-backtest/references/phase-criteria.md) 참조.

**FAIL -> Step F** | **PASS -> Step S**

---

## Step F: 실패 처리

Phase FAIL 시 다음을 순차 실행한다.

### F-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase {P5A|P5B} --verdict FAIL \
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
| 특정 파라미터 절벽 형태 | `risk-management` |

> 기존 교훈과 겹치면 추가하지 않는다.

### F-3. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](../p4-backtest/references/report-template.md) 참조.

---

## Step S: 성공 처리 (P5B PASS)

### S-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P5B --verdict PASS \
  --detail "plateau_pass={N}/{total}" --detail "stability_pass={N}/{total}" \
  --rationale "전 파라미터 고원 존재 + +/-20% 안정"
```

상태 `TESTING` 유지 (P6 심층검증 대기).

### S-2. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](../p4-backtest/references/report-template.md) 참조.

### S-3. 다음 단계

P6 심층검증으로 진행: `/p6-validation`

---

## 참조 문서

- [references/phase-criteria.md](../p4-backtest/references/phase-criteria.md) -- Phase별 정량 기준 + CTREND 비교 + CLI 명령
- [references/quant-interpretation-guide.md](../p4-backtest/references/quant-interpretation-guide.md) -- 시니어 퀀트 해석 패턴
- [references/report-template.md](../p4-backtest/references/report-template.md) -- 리포트 출력 형식
- `pipeline report` -- 전략 상황판 (CLI)
