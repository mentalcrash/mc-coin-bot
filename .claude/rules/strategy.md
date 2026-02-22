---
paths:
  - "src/strategy/**"
---

# Strategy Development Rules

## BaseStrategy Interface

```python
class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def required_columns(self) -> list[str]: ...

    @property
    def config(self) -> BaseModel | None: ...

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals: ...

    # Optional overrides
    @classmethod
    def from_params(cls, **params) -> BaseStrategy: ...
    @classmethod
    def recommended_config(cls) -> dict[str, Any]: ...
    def run_incremental(self, df) -> tuple[DataFrame, StrategySignals]: ...
    def get_startup_info(self) -> dict[str, str]: ...
```

## Zero Loop Policy (Critical)

**벡터화 연산만 허용**, 루프 금지:

```python
# Bad
for i in range(len(df)):
    if df["close"].iloc[i] > df["sma_20"].iloc[i]:
        signals.iloc[i] = 1

# Good
signals = np.where(df["close"] > df["sma_20"], 1, 0)
```

**금지:** `for` DataFrame 순회, `iterrows()`, `itertuples()`, `apply(axis=1)`

## Shift(1) Rule (Lookahead Bias Prevention) — SSOT

현재 봉 데이터로 시그널 생성 시 **반드시** `.shift(1)`:

```python
# Bad (lookahead bias)
signal = (df["close"] > df["sma_20"]).astype(int)

# Good
signal = (df["close"].shift(1) > df["sma_20"].shift(1)).astype(int)
```

## Directory Structure

```
src/strategy/my_strategy/
├── config.py         # Pydantic 설정 모델
├── preprocessor.py   # 지표 계산 (벡터화)
├── signal.py         # 시그널 생성 로직
└── strategy.py       # @register 메인 클래스
```

## Strategy Registry

```python
from src.strategy.registry import register

@register("my-strategy")
class MyStrategy(BaseStrategy):
    ...
```

## Common Gotchas

- **TYPE_CHECKING import**: `if TYPE_CHECKING:` 블록 안의 import는 런타임 사용 불가
- **DataFrame 컬럼 타입**: `df["close"]`는 `Series | DataFrame` — pyright 만족 시 `pd.Series` 캐스팅
- **from_params()**: Config 모델 거치는 전략은 반드시 오버라이드
- **df.copy()**: 여러 전략에 동일 df 전달 시 `.copy()` 필수 (컬럼 충돌 방지)
