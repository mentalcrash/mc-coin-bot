---
paths:
  - "src/**"
  - "tests/**"
---

# Lint & Type Standards

## Zero-Tolerance Policy

모든 코드는 Ruff/Pyright 에러 0개를 유지해야 합니다.

> `# noqa`, `# type: ignore` 사용 금지 (정당한 사유 없이)

## Ruff Checklist

| # | Rule | Check |
|---|------|-------|
| 1 | Import 순서: StdLib → Third Party → `src.*` | |
| 2 | 미사용 import/변수 없음 (F401, F841) | |
| 3 | Double quotes (`"`) 사용 | |
| 4 | `inplace=True` 금지 (PD002) | |
| 5 | `except:` 금지 → 구체적 예외 (TRY002) | |
| 6 | `async def` 내 블로킹 호출 금지 (ASYNC) | |
| 7 | `os.path` 대신 `pathlib.Path` (PTH) | |
| 8 | 모든 함수에 타입 힌트 | |

## Pyright Checklist (Strict Mode)

| # | Rule |
|---|------|
| 1 | 모든 함수 인자/반환 타입 명시 |
| 2 | `X \| None` 문법 사용 (Optional 아님) |
| 3 | 내장 제네릭: `list[]`, `dict[]` |
| 4 | None 가능 타입은 narrowing 후 사용 |
| 5 | 금융 데이터는 `Decimal` (float 금지) |

## Common Violations & Fixes

| Rule | Fix |
|------|-----|
| PLR0912 (branches > 12) | 서브메서드 추출 |
| PLR0913 (args > 5) | Config/dataclass로 묶기 |
| C901 (complexity > 10) | 함수 분할, early return |
| PLR0911 (returns 과다) | Guard clause → early return |

## Data Quality Rules (SSOT)

- `float` for prices/amounts → `Decimal` 필수
- `iterrows()`, loops on DataFrame → vectorized ops
- `inplace=True` → immutable operations
- `except:` → specific exceptions

## Workflow

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/
uv run pytest --cov=src
```
