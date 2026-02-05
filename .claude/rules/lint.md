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
| 1 | Import 순서: StdLib → Third Party → `src.*` | ✓ |
| 2 | 미사용 import/변수 없음 (F401, F841) | ✓ |
| 3 | Double quotes (`"`) 사용 | ✓ |
| 4 | 암시적 문자열 연결 금지 (ISC001) | ✓ |
| 5 | `inplace=True` 금지 (PD002) | ✓ |
| 6 | `except:` 금지 → 구체적 예외 (TRY002) | ✓ |
| 7 | `async def` 내 블로킹 호출 금지 (ASYNC) | ✓ |
| 8 | `os.path` 대신 `pathlib.Path` (PTH) | ✓ |
| 9 | `len(x) > 0` → `if x:` (SIM) | ✓ |
| 10 | 모든 함수에 타입 힌트 | ✓ |

## Pyright Checklist (Strict Mode)

| # | Rule | Check |
|---|------|-------|
| 1 | 모든 함수 인자/반환 타입 명시 | ✓ |
| 2 | `X \| None` 문법 사용 (Optional 아님) | ✓ |
| 3 | 내장 제네릭: `list[]`, `dict[]` | ✓ |
| 4 | None 가능 타입은 narrowing 후 사용 | ✓ |
| 5 | 금융 데이터는 `Decimal` (float 금지) | ✓ |

## Example

```python
# ✅ Good
from decimal import Decimal
from pathlib import Path

from loguru import logger

from src.models import Order

async def process_order(order_id: str, price: Decimal | None) -> None:
    if price is None:
        logger.warning(f"Order {order_id}: price is None")
        return
    await asyncio.sleep(0.1)
    logger.info(f"Processing: {order_id} at {price}")


# ❌ Bad
import time  # ASYNC101
from src.models import Order  # I001: wrong order

def process(id):  # missing types
    time.sleep(1)  # blocking in async context
    print("done")  # use logger
```

## Workflow

```bash
# 1. Auto-fix style
uv run ruff check --fix . && uv run ruff format .

# 2. Type check
uv run pyright src/

# 3. Run tests
uv run pytest --cov=src
```
