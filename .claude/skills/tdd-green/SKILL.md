---
name: tdd-green
description: >
  TDD Phase 2 (GREEN+REFACTOR) — 테스트 통과 구현. tests/ 읽기/수정 금지.
  사용 시점: /tdd-red 완료 후 구현 진행.
context: fork
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
hooks:
  PreToolUse:
    - matcher: "Read|Write|Edit|Grep|Glob|Bash"
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/hooks/tdd-green-guard.sh"
argument-hint: <작업 설명>
---

# tdd-green: TDD Phase 2 — GREEN + REFACTOR (구현)

## 역할

**구현 개발자**로서 행동한다.
테스트 코드(`tests/`)는 **절대 읽지도, 수정하지도 않는다**.
오직 `pytest` 실행 결과(pass/fail)만 보고 `src/` 코드를 구현한다.

---

## 절대 규칙

### NEVER (금지)

- `tests/` 하위 파일 **Read/Write/Edit** — 테스트 내용 확인 금지
- `Grep`/`Glob`으로 `tests/` 검색 — 테스트 구현 세부사항 탐색 금지
- `cat`, `head`, `tail` 등으로 테스트 파일 내용 조회 금지
- 테스트를 통과시키기 위해 테스트 코드를 수정하는 행위

### ALWAYS (필수)

- `src/` 하위 코드만 Read/Write/Edit
- `pytest` 출력(pass/fail/error message)만으로 구현 방향 판단
- 구현 후 `pytest` 실행하여 **GREEN 확인**
- GREEN 후 리팩토링 시에도 테스트 코드 미열람

---

## 워크플로우

### Step 1: 현재 실패 확인

```bash
uv run pytest tests/{대상}/ -v --tb=short --no-header 2>&1 | tail -50
```

**pytest 출력에서 알 수 있는 정보:**
- 테스트 함수 이름 → 기대하는 동작 추론
- `AssertionError` 메시지 → 기대값 vs 실제값
- `ImportError` / `AttributeError` → 필요한 모듈/함수/클래스
- `TypeError` → 함수 시그니처

### Step 2: 구현 (GREEN)

1. `src/` 코드를 읽고 수정 대상 파악
2. 테스트 이름과 에러 메시지를 기반으로 **최소한의 구현**
3. 테스트 실행 → FAIL이면 에러 메시지 분석 후 수정 반복

```bash
uv run pytest tests/{대상}/ -v --tb=short 2>&1 | tail -50
```

### Step 3: 전체 테스트 확인

대상 테스트 통과 후 전체 테스트도 확인:

```bash
uv run pytest --tb=short -q 2>&1 | tail -20
```

### Step 4: REFACTOR (BLUE)

GREEN 확인 후 리팩토링:

- 중복 코드 제거
- 네이밍 개선
- 복잡도 축소

리팩토링 후 다시 테스트 실행:

```bash
uv run pytest tests/{대상}/ -v --tb=short 2>&1 | tail -30
```

### Step 5: 품질 게이트

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/
uv run pytest --tb=short -q
```

### Step 6: 완료 선언

```
✅ Phase 2 (GREEN + REFACTOR) 완료
- 수정 파일: src/xxx/yyy.py
- 테스트 결과: N passed
- Lint/Type: PASS

→ 추가 기능이 필요하면 /tdd-red 로 새 사이클을 시작하세요.
```

---

## 테스트 수정이 필요한 경우

pytest 출력에서 **테스트 자체의 버그**가 명확한 경우에만:

1. 사용자에게 테스트 버그 의심 사항 보고
2. 사용자가 승인하면 **최소한의 수정만** 수행
3. 수정 사유를 명시

**승인 없이 절대 테스트 수정 불가.**

---

## 구현 가이드

### 최소 구현 원칙

- 테스트를 통과하는 **가장 단순한 코드**부터 시작
- 과도한 추상화, 미래 대비 설계 금지
- YAGNI (You Aren't Gonna Need It)

### pytest 출력 읽기 팁

| 출력 패턴 | 의미 |
|-----------|------|
| `FAILED test_x - AssertionError: assert 0 == 1` | 기대값 불일치 |
| `FAILED test_x - AttributeError: 'X' has no attribute 'y'` | 메서드/속성 미구현 |
| `FAILED test_x - ImportError: cannot import 'Z'` | 모듈/클래스 미존재 |
| `FAILED test_x - TypeError: f() takes 1 arg` | 시그니처 불일치 |
| `ERROR test_x - fixture 'abc' not found` | fixture 미정의 |
