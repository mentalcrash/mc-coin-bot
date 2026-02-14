---
name: audit-fix
description: >
  Audit Finding/Action 수정 — 코드 수정 + 테스트 + YAML 상태 갱신.
  사용 시점: /audit-inspect 후 Finding 수정, "fix" "audit-fix" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: "<finding-id|action-id> [--type finding|action]"
---

# Audit Fix

## 역할

**시니어 소프트웨어 엔지니어**로서 행동한다.
감사에서 발견된 이슈를 안전하게 수정하고, 테스트로 검증한다.

- **최소 변경**: 이슈 해결에 필요한 최소한의 코드만 수정
- **테스트 필수**: 모든 수정에 대응하는 테스트 작성
- **안전 우선**: 전략 로직/리스크 파라미터/외부 API/의존성 변경 시 사용자 승인 필수
- **Zero-Tolerance**: ruff + pyright + pytest 모두 0 error 유지

---

## 인수

- 첫 번째 인수: Finding ID 또는 Action ID (정수)
- `--type`: `finding` (기본값) 또는 `action`

---

## 워크플로우

### Step 0: Preflight

```bash
# Finding 상세 로드
uv run mcbot audit finding-show {id}

# 상태 확인: open 또는 in_progress만 수정 가능
# resolved/wont_fix/deferred → "이미 처리된 항목입니다" 출력 후 종료

# 관련 항목 확인
uv run mcbot audit actions  # 관련 Action 확인
```

Action ID가 주어진 경우:
```bash
uv run mcbot audit action-show {id}
# related_findings에서 Finding ID 추출 후 각각 로드
```

### Step 1: 수정 계획

Finding의 정보를 분석하여 수정 계획을 수립:

1. **영향 범위**: `location` 필드의 파일 + 관련 파일 식별
2. **수정 패턴**: [fix-patterns.md](references/fix-patterns.md) 참조
3. **리스크 평가**: 변경이 다른 컴포넌트에 미치는 영향

출력:
```
수정 계획
──────────────────────────────────────
Finding: #{id} {title}
Severity: {severity}
Category: {category}

영향 파일:
  - src/eda/oms.py (수정)
  - tests/eda/test_oms.py (테스트 추가)

수정 방향: {proposed_fix 기반}
리스크: LOW / MEDIUM / HIGH
```

**HIGH 리스크** (아래 해당 시 사용자 승인 필수):
- 전략 시그널 생성 로직 변경
- PM/RM 리스크 파라미터 수정
- 외부 API (Binance, Discord) 호출 변경
- `pyproject.toml` 의존성 변경
- Live trading path 수정

### Step 2: 코드 수정

[fix-patterns.md](references/fix-patterns.md)의 카테고리별 패턴을 적용:

**수정 원칙**:
- 기존 코드 스타일과 일관성 유지
- `from __future__ import annotations` 패턴 준수
- Pydantic V2 / frozen model 패턴 준수
- 매직 넘버 대신 명명된 상수

### Step 3: 테스트 작성

Finding의 `verification` 또는 `proposed_fix`에서 테스트 케이스를 도출:

- 수정된 동작을 직접 검증하는 테스트
- edge case 테스트 (NaN, 0, negative, overflow)
- 기존 테스트가 깨지지 않는지 확인

테스트 위치: 기존 테스트 파일에 추가 (새 파일 생성 최소화)

### Step 4: 품질 검증

```bash
# Lint
uv run ruff check --fix . && uv run ruff format .

# Type check
uv run pyright src/

# Tests
uv run pytest

# 모두 0 error 필수
```

**하나라도 실패하면** Step 2로 돌아가 수정.

### Step 5: YAML 갱신

Finding 해결:
```bash
uv run mcbot audit resolve-finding {finding_id}
```

관련 Action 완료:
```bash
uv run mcbot audit update-action {action_id} -s completed
```

### Step 6: Fix Report

```
============================================================
  AUDIT FIX REPORT
  날짜: YYYY-MM-DD
  Finding: #{id} {title}
============================================================

  상태: RESOLVED

  변경 파일:
  ──────────────────────────────────────
  M src/eda/oms.py          (+15, -3)
  A tests/eda/test_oms.py   (+42)

  품질 체크:
  ──────────────────────────────────────
  ruff:    PASS (0 errors)
  pyright: PASS (0 errors)
  pytest:  PASS (N tests)

  잔여 현황:
  ──────────────────────────────────────
  Open findings:   N건 (critical: N)
  Pending actions: N건

  다음: /audit-fix {next_id} 또는 /audit-inspect
============================================================
```

---

## 안전 규칙

### Decision Gates (사용자 승인 필수)

| 변경 유형 | 승인 필요 |
|----------|:---------:|
| 전략 시그널 로직 | YES |
| PM/RM 리스크 파라미터 | YES |
| 외부 API 호출 (Binance, Discord) | YES |
| `pyproject.toml` 의존성 | YES |
| Live trading 경로 | YES |
| 테스트 코드만 | NO |
| 로깅/주석 추가 | NO |
| assert → if/raise 변환 | NO |
| 상수 추출/매직넘버 제거 | NO |

### Autonomous Zone (자율 실행)

- lint/format 수정
- 타입 힌트 추가
- 테스트 작성
- assert → if/raise 변환 (production code)
- bare except → specific exception
- `# noqa` 제거 + 근본 원인 수정
- 상수 추출
