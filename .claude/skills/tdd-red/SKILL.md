---
name: tdd-red
description: >
  TDD Phase 1 (RED) — 테스트 먼저 작성. src/ 코드 수정 금지.
  사용 시점: 새 기능 구현, 버그 수정, 리팩토링 전 테스트 작성.
context: fork
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
argument-hint: <작업 설명>
---

# tdd-red: TDD Phase 1 — RED (테스트 작성)

## 역할

**테스트 설계자**로서 행동한다.
구현 코드(`src/`)는 읽기만 가능하며, **절대 수정하지 않는다**.
오직 `tests/` 디렉토리의 테스트 코드만 작성/수정한다.

---

## 절대 규칙

### NEVER (금지)

- `src/` 하위 파일 Write/Edit — **읽기(Read)만 허용**
- 테스트를 통과시키기 위한 stub/mock 과잉 사용 (실제 동작 검증 목적)
- 구현 세부사항에 의존하는 테스트 (공개 인터페이스만 테스트)

### ALWAYS (필수)

- `tests/` 하위에만 파일 생성/수정
- 테스트 작성 후 반드시 실행하여 **RED (실패) 확인**
- 실패 출력을 사용자에게 보여줌
- 테스트 의도(무엇을 검증하는지) 간결히 설명

---

## 워크플로우

### Step 1: 요구사항 분석

1. 사용자의 작업 설명을 분석
2. 관련 `src/` 코드를 **Read**로 읽어 현재 인터페이스 파악
3. 테스트 대상 모듈/함수/클래스 식별

### Step 2: 테스트 설계

테스트 케이스 목록을 먼저 사용자에게 제시:

```
테스트 계획:
1. test_xxx_기본동작 — 정상 입력 시 기대 결과
2. test_xxx_경계값 — 엣지 케이스
3. test_xxx_에러 — 잘못된 입력 시 예외
```

### Step 3: 테스트 코드 작성

- `tests/` 하위에 테스트 파일 작성/수정
- 기존 conftest.py, fixture 패턴 준수
- `pytest` + `unittest.mock` 사용

### Step 4: RED 확인 (필수)

```bash
uv run pytest tests/{대상}/ -v --tb=short --no-header 2>&1 | tail -30
```

**반드시 FAILED 출력을 확인하고 사용자에게 보여준다.**

### Step 5: 완료 선언

```
✅ Phase 1 (RED) 완료
- 작성된 테스트: N개
- 실패 확인: N FAILED
- 테스트 파일: tests/xxx/test_yyy.py

→ /tdd-green 으로 구현을 진행하세요.
```

**여기서 반드시 멈춘다. 구현으로 넘어가지 않는다.**

---

## 테스트 작성 가이드

### 좋은 테스트 특성

- **독립적**: 다른 테스트에 의존하지 않음
- **명확한 이름**: `test_함수명_시나리오_기대결과`
- **하나의 assertion**: 테스트당 하나의 검증 포인트
- **인터페이스 기반**: 내부 구현이 아닌 공개 API 테스트

### 프로젝트 테스트 컨벤션

- `AsyncMock`으로 외부 API mock
- `event_bus.flush()` 필수 (EDA 테스트)
- `pytest.fixture`로 공통 setup 분리
- `-n auto --dist worksteal` 병렬 실행 호환
