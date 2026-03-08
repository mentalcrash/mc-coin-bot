---
paths:
  - "src/**"
  - "tests/**"
---

# TDD Workflow (Red-Green-Refactor)

## 엄격한 2-Phase TDD — 에이전트 분리

모든 구현/수정 작업은 두 개의 **분리된 에이전트(skill)**로 진행한다.

### /tdd-red (Phase 1: 테스트 작성 에이전트)

- `tests/` 만 Write/Edit 가능
- `src/` 는 Read만 가능 (인터페이스 파악용)
- 테스트 작성 후 RED(실패) 확인 필수
- **구현으로 넘어가지 않고 멈춤**

### /tdd-green (Phase 2: 구현 에이전트)

- `src/` 만 Read/Write/Edit 가능
- `tests/` 는 **읽기도 금지** — pytest 출력(pass/fail)만 참조
- 테스트 통과(GREEN) 후 리팩토링(BLUE)
- 테스트 수정 필요 시 사용자 승인 필수

### 사용법

```
사용자: /tdd-red EventBus에 priority 기능 추가
에이전트: 테스트 작성 → RED 확인 → 멈춤

사용자: /tdd-green EventBus에 priority 기능 추가
에이전트: pytest 출력만 보고 구현 → GREEN → REFACTOR
```

### 일반 작업 시 (skill 미사용)

사용자가 `/tdd-red`, `/tdd-green` 없이 직접 구현 요청하면 TDD 프로세스를 건너뛴다.
단, "TDD로" 또는 "테스트 먼저"라고 명시하면 `/tdd-red`부터 시작을 안내한다.
