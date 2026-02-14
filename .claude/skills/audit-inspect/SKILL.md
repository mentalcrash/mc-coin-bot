---
name: audit-inspect
description: >
  Architecture Audit 실행 — 코드 품질/아키텍처/리스크 정량 분석 + Finding/Action 생성.
  사용 시점: 정기 감사, "audit" "감사" 요청, 대규모 변경 후 건강도 확인 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: "[--scope architecture,risk-safety,code-quality,data-pipeline,testing-ops,performance]"
---

# Architecture Audit Inspector

## 역할

**시니어 소프트웨어 아키텍트**로서 행동한다.
객관적 지표와 코드 리뷰를 기반으로 프로젝트 건강 상태를 진단한다.

- **정량 우선**: 모든 판단에 수치 근거 제시
- **비교 관점**: 이전 스냅샷 대비 변화를 추적
- **실행 가능**: 등급 채점 + 구체적 Finding/Action 생성
- **중복 방지**: 기존 Finding과 location+title 비교하여 새것만 등록

---

## 인수

- `--scope`: 쉼표 구분 카테고리 (기본: 전체 6카테고리)
  - `architecture`, `risk-safety`, `code-quality`, `data-pipeline`, `testing-ops`, `performance`

---

## 워크플로우

### Step 0: Preflight

```bash
# Git SHA 확인
git rev-parse --short HEAD

# 이전 감사 로드
uv run mcbot audit latest
uv run mcbot audit findings --status open
uv run mcbot audit actions --status pending
```

이전 스냅샷의 등급, open finding 수, pending action 수를 기록한다.

### Step 1: 정량 지표 수집

```bash
# 테스트 실행 (카운트 + 패스율)
uv run pytest --tb=no -q 2>&1 | tail -5

# Lint 에러 수
uv run ruff check . 2>&1 | tail -3

# Type 에러 수
uv run pyright src/ 2>&1 | tail -3

# 커버리지
uv run pytest --cov=src --cov-report=term-missing --tb=no -q 2>&1 | tail -30
```

결과를 `MetricsSnapshot` 구조로 정리:
- `test_count`, `test_pass_rate`, `lint_errors`, `type_errors`, `coverage_pct`

### Step 2: 모듈 건강도

8개 모듈 각각에 대해:

| 모듈 | 경로 |
|------|------|
| src/eda | EDA 이벤트 시스템 |
| src/strategy | 전략 모듈 |
| src/exchange | 거래소 연동 |
| src/data | 데이터 파이프라인 |
| src/pipeline | 전략 관리 파이프라인 |
| src/notification | 알림 시스템 |
| src/core | 핵심 이벤트/설정 |
| src/models | Pydantic 모델 |

각 모듈별:
1. 커버리지 확인 (pytest --cov 결과에서 추출)
2. 코드 리뷰 항목 확인 ([module-checklist.md](references/module-checklist.md) 참조):
   - `assert` 문이 프로덕션 코드에서 crash 위험
   - bare `except:` 사용
   - `# noqa`, `# type: ignore` 남용
   - TODO/FIXME 잔존
   - 순환 import

건강도 판정:
- **GREEN**: 커버리지 80%+, 코드 리뷰 이슈 없음
- **YELLOW**: 커버리지 70-80% 또는 경미한 이슈
- **RED**: 커버리지 70% 미만 또는 심각한 이슈

### Step 3: 전략 현황

```bash
uv run mcbot pipeline status
```

`StrategySummary` 구조로 정리: `total`, `active`, `testing`, `candidate`, `retired`

### Step 4: 등급 채점

[grading-rubric.md](references/grading-rubric.md) 기준으로 6카테고리 A~F 등급 채점.

Overall = 6등급 가중 평균:
- risk-safety: **2배** 가중
- 나머지 5개: 1배 가중

등급 점수: A+=4.3, A=4.0, A-=3.7, B+=3.3, B=3.0, B-=2.7, C+=2.3, C=2.0, C-=1.7, D=1.0, F=0.0

### Step 5: Finding 발견

코드 리뷰 결과에서 새 Finding을 식별한다.

**중복 검사**:
```bash
# 기존 Finding 목록 확인
uv run mcbot audit findings
```

기존 Finding과 `location` + `title`이 동일하면 스킵.

각 Finding에 대해:
- severity: critical / high / medium / low
- category: 6카테고리 중 하나
- location: 파일경로:라인번호
- description, impact, proposed_fix, effort
- tags

### Step 6: Action 제안

Finding별 ActionItem 생성:
- priority: P0 (critical) / P1 (high) / P2 (medium) / P3 (low)
- phase: A (즉시) / B (단기) / C (장기)
- verification: 해결 확인 방법
- related_findings: Finding ID 목록

### Step 7: YAML 저장

**Finding 저장**:
```bash
uv run mcbot audit add-finding \
  --title "제목" \
  --severity critical \
  --category risk-safety \
  --location "src/eda/oms.py:56" \
  --description "설명" \
  --impact "영향" \
  --proposed-fix "수정안" \
  --effort "2h" \
  --tag live-trading
```

**Action 저장**:
```bash
uv run mcbot audit add-action \
  --title "제목" \
  --priority P0 \
  --phase A \
  --description "설명" \
  --effort "4h" \
  --verification "검증 방법" \
  --finding 1 \
  --tag live-trading
```

**Snapshot 저장**:
1. 임시 YAML 파일 생성 (Write tool로 /tmp/audit_YYYY-MM-DD.yaml)
2. CLI로 저장:
```bash
uv run mcbot audit create-snapshot --from-yaml /tmp/audit_YYYY-MM-DD.yaml
```

스냅샷 YAML 형식:
```yaml
date: "YYYY-MM-DD"
git_sha: "<short sha>"
auditor: claude
scope:
  - architecture
  - risk-safety
  - code-quality
  - data-pipeline
  - testing-ops
metrics:
  test_count: 3072
  test_pass_rate: 1.0
  lint_errors: 0
  type_errors: 0
  coverage_pct: 0.78
module_health:
  - module: src/eda
    health: green
    coverage_pct: 0.89
    notes: "..."
strategy_summary:
  total: 50
  active: 5
  testing: 8
  candidate: 12
  retired: 25
new_findings: [11, 12]
new_actions: [7, 8]
grades:
  architecture: "B+"
  risk_safety: "C+"
  code_quality: "A+"
  data_pipeline: "B"
  testing_ops: "B-"
  overall: "B"
summary: |
  요약 텍스트
```

### Step 8: 리포트 출력

```
============================================================
  ARCHITECTURE AUDIT REPORT
  날짜: YYYY-MM-DD
  Git SHA: <sha>
  범위: [카테고리 목록]
============================================================

  등급 요약
  ──────────────────────────────────────────
  Architecture:  B+ (← B  [+0.3])
  Risk-Safety:   C+ (← C  [+0.3])
  Code-Quality:  A+ (← A+ [=])
  Data-Pipeline: B  (← B  [=])
  Testing-Ops:   B- (← B- [=])
  ──────────────────────────────────────────
  Overall:       B  (← B  [=])

  정량 지표
  ──────────────────────────────────────────
  Tests: 3100 (+28)  |  Pass: 100%
  Lint:  0           |  Type: 0
  Coverage: 80% (+2%)

  모듈 건강도
  ──────────────────────────────────────────
  [GREEN]  src/eda        89%  EventBus stable
  [YELLOW] src/exchange   65%  LiveExecutor assert risk
  ...

  새 Findings: 2건 (critical: 0, high: 1, medium: 1)
  새 Actions:  2건 (P0: 0, P1: 1, P2: 1)

  우선순위 액션
  ──────────────────────────────────────────
  [P0] #1 OMS persistence (2h) — open
  [P1] #7 Exchange error handling (4h) — pending
  ...

  다음 단계: /audit-fix {finding_id}
============================================================
```

---

## 안전 규칙

- **읽기 전용**: 코드 수정하지 않음 (Finding/Action만 생성)
- **비파괴적**: 기존 Finding/Action 수정하지 않음
- **투명성**: 모든 등급 판정에 근거 수치 제시
