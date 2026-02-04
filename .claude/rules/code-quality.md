# 🧹 Code Quality Standards: Ruff & Pyright

## ⚠️ CRITICAL: Zero-Tolerance Lint Policy

**모든 코드 변경은 다음 린트 도구의 에러가 0개여야 합니다.**

- **Ruff:** `pyproject.toml`에 정의된 모든 규칙 준수 필수
- **Pyright (VSCode Pylance):** `strict` 모드 수준의 타입 체크 통과 필수

### 검사 실행 방식
```bash
# Ruff 검사 (CLI에서 실행)
uv run ruff check .
uv run ruff format .

# Pyright 검사 (VSCode Pylance가 자동으로 실행)
# .vscode/settings.json: python.analysis.typeCheckingMode = "strict"
```

> [!CAUTION]
> **`# noqa`, `# ruff: noqa`, `# type: ignore` 사용 절대 금지**
>
> 린트/타입 체커를 주석으로 무력화하는 것은 최악의 최악의 최악의 상황에서만 허용됩니다.
> 가능하면 **코드를 수정하여 규칙을 준수**하는 방향으로 작성하십시오.

---

## 1. Ruff Compliance Standards

### Code Generation Checklist (출력 전 필수)

코드 생성 시 다음을 **반드시** 확인한 후 출력하십시오:

| # | 규칙 | 확인 |
|---|------|------|
| 1 | Import: StdLib → Third Party → `from src.*` 순서, `combine-as-imports` | ✓ |
| 2 | 미사용 import/변수 없음 (F401, F841) | ✓ |
| 3 | 문자열: **Double quotes (`"`)** 만 사용 | ✓ |
| 4 | **ISC001:** 암시적 문자열 연결 금지 (`"a" "b"` ❌ → `"a" + "b"` 또는 `textwrap.dedent`) | ✓ |
| 5 | `inplace=True` 금지 (PD002), Pandas는 불변 연산 | ✓ |
| 6 | `except:` / `except Exception:` 금지 → 구체적 예외 명시 (TRY002) | ✓ |
| 7 | `async def` 내부에 `time.sleep()`, `requests.get()` 등 블로킹 호출 금지 (ASYNC101/102) | ✓ |
| 8 | `len(x) > 0` → `if x:` 등 SIM 규칙 적용 | ✓ |
| 9 | 경로: `os.path` 대신 `pathlib.Path` (PTH) | ✓ |
| 10 | 타입 힌트: 모든 함수 인자·반환값에 명시 | ✓ |

### 적용 규칙셋 (2026 최신)
**활성화:** E, W, F, I, B, UP, N, SIM, C4, ASYNC, S, RUF, PERF, LOG, TC, PTH, PD, TRY, PL, ISC, **FURB, SLOT**

**2026년 추가 규칙:**
- **FURB** (refurb) - 최신 Python 리팩토링 제안
- **SLOT** (flake8-slots) - `__slots__` 메모리 최적화

**무시됨 (신경 쓰지 않아도 됨):**
- `E501` (줄 길이) — formatter가 처리
- `B008` (함수 호출 인자) — FastAPI Depends 등
- `S101` (assert) — 테스트에서 허용
- `S311` (random) — 보안 불필요 시

### 주요 규칙 상세

#### 🐼 Pandas (PD)
- **PD002:** `inplace=True` 금지 → `df = df.fillna(0)`
- **PD901:** `df` 변수명 지양 → `prices_df`, `ohlcv` 등 구체적 이름
- 벡터화: `iterrows`, `itertuples` 사용 금지

#### 🛡️ 예외 처리 (TRY, B)
- **TRY002:** `except:` 금지 → `except ValueError:` 등 구체적 예외
- **B904:** `raise X from e` 형태로 체인 유지

#### ⚡ 비동기 (ASYNC)
- **ASYNC101/102:** `async def` 내 `time.sleep()`, `requests.get()` 금지
- → `await asyncio.sleep()`, `aiohttp` 등 사용

#### 🧹 Import (I, F)
- **I001:** StdLib → Third Party → First Party (`src`) 순서
- **combine-as-imports:** `from foo import a, b` 한 줄로
- **F401:** 미사용 import 금지
- **F841:** 미사용 변수 금지

#### 📜 스타일 (N, SIM, ISC)
- **N802:** 함수명 `snake_case`
- **N806:** 변수명 `snake_case`
- **SIM:** `if len(x) > 0` → `if x`, `x == None` → `x is None`
- **ISC001:** `"a" "b"` 암시적 연결 금지 → `"a" + "b"` 또는 `textwrap.dedent`

---

## 2. Pyright (VSCode Pylance) Typing Standards

### Code Generation Checklist (출력 전 필수)

| # | 규칙 | 확인 |
|---|------|------|
| 1 | **모든 함수/메서드 인자에 타입 힌트** | ✓ |
| 2 | **모든 함수/메서드에 반환 타입** `-> None` 포함 | ✓ |
| 3 | **내장 제네릭 사용:** `list[]`, `dict[]`, `tuple[]` | ✓ |
| 4 | **Union:** `Union[X,Y]` 대신 파이프 `str \| None` | ✓ |
| 5 | **Optional 처리:** None 가능 타입 사용 전 narrowing | ✓ |
| 6 | **암시적 문자열 연결 금지** (Ruff ISC001과 동일) | ✓ |
| 7 | **금융 데이터:** `Decimal` 사용, `float` 혼용 시 명시적 변환 | ✓ |
| 8 | **Self 반환:** `-> Self` (typing.Self) | ✓ |
| 9 | **타입 별칭:** `type OrderID = str` (Python 3.12+) | ✓ |

### 설정 요약
- **Mode:** `typeCheckingMode = "strict"`
- **Version:** `pythonVersion = "3.13"`
- **Scope:** `src/`만 검사, `tests/`, `research/` 제외

### 주요 규칙 상세

#### 📝 타입 힌트 필수
- **reportMissingParameterType:** `def foo(x):` ❌ → `def foo(x: int) -> None:` ✅
- **reportReturnType:** 반환값 없으면 `-> None` 명시
- `self`, `cls`는 예외

#### 🔤 Python 3.13 문법
- **제네릭:** `list[str]`, `dict[str, int]`, `tuple[int, str]`
- **Union:** `str | None`, `int | float`
- **Type Alias:** `type Price = Decimal`
- **Self:** `def copy(self) -> Self:`

#### ⚠️ Optional 안전 처리
- `x: str | None`일 때 `x.upper()` ❌
- 먼저 `if x is not None:` 등으로 narrowing

#### 💰 Decimal & 금융 데이터
- 가격, 수량 등은 `Decimal`
- `float` → `Decimal`: `Decimal(str(val))`

#### 🧩 Async
- `await` 누락 시 `reportUnusedCoroutine`
- 코루틴 반환값은 반드시 await

---

## 3. Example Patterns

### ✅ 준수 코드
```python
import asyncio
from decimal import Decimal
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from src.models import Order

type OrderID = str

async def process_order(order_id: OrderID, price: Decimal | None) -> None:
    """주문 처리 (타입 안전, Ruff/Pyright 준수)"""
    # Guard Clause (Fail Fast)
    if price is None:
        logger.warning(f"Order {order_id}: price is None")
        return

    # Optional narrowing 후 사용
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")

    try:
        await asyncio.sleep(0.1)
        logger.info(f"Processing order: {order_id} at {price}")
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise
```

### ❌ 위반 코드
```python
import time  # ASYNC101
from src.models import Order  # I001: First party should be last

def process(id):  # reportMissingParameterType, reportReturnType
    try:
        time.sleep(1)  # ASYNC101
        print("done")  # LOG: logger 사용
    except:  # TRY002: 구체적 예외 명시 필요
        pass

    # 암시적 문자열 연결 (ISC001)
    message = "Order " "processed"  # ❌
```

---

## 4. Ruff vs Pyright 역할 분담

- **Ruff:** Import 정리, 미사용 변수, 스타일, PD/TRY/ASYNC, FURB/SLOT 등
- **Pyright (VSCode Pylance):** 타입 호환성, Optional, 반환 타입, 암시적 문자열 연결
