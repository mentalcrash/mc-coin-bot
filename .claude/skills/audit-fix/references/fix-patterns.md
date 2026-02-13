# Fix Patterns by Category

카테고리별 일반적인 수정 패턴.

## risk-safety

### RS-1: assert → if/raise

Production 코드에서 `assert`는 `-O` 플래그로 비활성화되어 crash risk.

**Before**:
```python
assert position.size > 0, "Position size must be positive"
```

**After**:
```python
if position.size <= 0:
    msg = f"Position size must be positive, got {position.size}"
    raise ValueError(msg)
```

### RS-2: Cash Negative Guard

음수 잔고 방지.

**Pattern**:
```python
if self._cash < order_cost:
    msg = f"Insufficient cash: {self._cash} < {order_cost}"
    raise InsufficientFundsError(msg)
```

### RS-3: State Persistence

OMS/CB 상태가 메모리에만 존재하면 재시작 시 유실.

**Pattern**: SQLite 또는 YAML 파일에 상태 저장.
```python
def _persist_state(self) -> None:
    """Persist critical state to disk."""
    state = {
        "pending_orders": [o.model_dump(mode="json") for o in self._pending],
        "circuit_breaker_active": self._cb_active,
    }
    self._state_path.write_text(
        yaml.dump(state, default_flow_style=False),
        encoding="utf-8",
    )
```

### RS-4: Decimal Precision

금융 계산에서 float 대신 Decimal 사용.

**Before**:
```python
fee = price * quantity * 0.001
```

**After**:
```python
from decimal import Decimal
fee = Decimal(str(price)) * Decimal(str(quantity)) * Decimal("0.001")
```

### RS-5: Rate Limit Guard

API 호출 빈도 제한.

**Pattern**:
```python
async def _retry_with_backoff(self, coro, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await coro()
        except ccxt.RateLimitExceeded:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
        except ccxt.NetworkError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)
```

---

## architecture

### ARCH-1: Import Direction Fix

의존성 방향 위반 수정.

**규칙**: `CLI -> Strategy/Backtest -> Data/Exchange/Portfolio -> Models/Core -> Config`

하위 레이어가 상위 레이어를 import하면 위반.

**수정**: Protocol/Port 패턴으로 의존성 역전.
```python
# core/ports.py
class DataFeedPort(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

# eda/runner.py (상위 레이어)
def __init__(self, data_feed: DataFeedPort): ...
```

### ARCH-2: Circular Import

순환 import 해소.

**수정 방법**:
1. TYPE_CHECKING 블록으로 이동 (타입 힌트 전용)
2. 공통 모듈 추출 (양쪽이 사용하는 것을 별도 모듈로)
3. Protocol 패턴 (인터페이스 분리)

---

## code-quality

### CQ-1: noqa 제거

`# noqa` 주석을 제거하고 근본 원인 수정.

**단계**:
1. noqa가 억제하는 규칙 확인
2. 해당 규칙 위반의 근본 원인 파악
3. 코드 수정으로 해결 (noqa 없이도 통과하도록)

### CQ-2: Magic Number → Named Constant

**Before**:
```python
if drawdown > 0.05:
    self._trigger_circuit_breaker()
```

**After**:
```python
_SYSTEM_STOP_LOSS_PCT = 0.05

if drawdown > _SYSTEM_STOP_LOSS_PCT:
    self._trigger_circuit_breaker()
```

### CQ-3: Bare Except → Specific Exception

**Before**:
```python
try:
    result = await exchange.create_order(...)
except:
    logger.error("Order failed")
```

**After**:
```python
try:
    result = await exchange.create_order(...)
except (ccxt.NetworkError, ccxt.ExchangeError) as e:
    logger.error(f"Order failed: {e}")
    raise OrderExecutionError(str(e)) from e
```

### CQ-4: type: ignore 제거

`# type: ignore` 제거하고 올바른 타입 힌트 추가.

**패턴**:
- pandas `100 * Series` → `pd.Series` annotation + 필요시 정당 사유 유지
- Union type 미지원 → `|` 연산자 사용 (Python 3.10+)

---

## data-pipeline

### DP-1: Gap Detection

데이터 갭 탐지 + 보간.

**Pattern**:
```python
def detect_gaps(df: pd.DataFrame, freq: str = "1min") -> pd.DatetimeIndex:
    expected = pd.date_range(df.index[0], df.index[-1], freq=freq)
    return expected.difference(df.index)
```

### DP-2: Timezone Consistency

모든 timestamp를 UTC로 통일.

**Pattern**:
```python
df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
```

---

## testing-ops

### TO-1: Missing Test

누락된 테스트 추가.

**원칙**:
- 기존 테스트 파일에 추가 (새 파일 최소화)
- 기존 fixture 재사용
- AAA 패턴 (Arrange-Act-Assert)
- edge case 포함 (0, None, empty, max)

### TO-2: Coverage Improvement

커버리지 부족 모듈에 테스트 추가.

**단계**:
1. `pytest --cov=src/{module} --cov-report=term-missing` 실행
2. Missing lines 확인
3. 해당 브랜치/경로를 커버하는 테스트 작성

---

## performance

### PERF-1: Loop → Vectorized

for 루프를 벡터화 연산으로 변환.

**Before**:
```python
for i in range(len(df)):
    if df.iloc[i]["rsi"] > 70:
        signals[i] = -1
```

**After**:
```python
signals = np.where(df["rsi"] > 70, -1, 0)
```

### PERF-2: iterrows → Vectorized

**Before**:
```python
for idx, row in df.iterrows():
    result.append(row["close"] * row["weight"])
```

**After**:
```python
result = df["close"] * df["weight"]
```
