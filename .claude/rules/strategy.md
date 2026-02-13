---
paths:
  - "src/strategy/**"
---

# Strategy Development Rules

## BaseStrategy Interface

모든 전략은 `BaseStrategy`를 상속하고 다음 메서드를 구현해야 합니다:

```python
class BaseStrategy(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 (벡터화 연산만)"""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """시그널 생성 (-1: 매도, 0: 관망, 1: 매수)"""

    @abstractmethod
    def get_config(self) -> BaseConfig:
        """전략 설정 반환"""
```

## Zero Loop Policy (Critical)

**벡터화 연산만 허용**, 루프 금지:

```python
# ❌ Bad (100x slower)
for i in range(len(df)):
    if df['close'].iloc[i] > df['sma_20'].iloc[i]:
        signals.iloc[i] = 1

# ✅ Good (vectorized)
signals = np.where(df['close'] > df['sma_20'], 1, 0)
```

**금지 목록:**
- `for` 루프로 DataFrame 순회
- `iterrows()`, `itertuples()`
- `apply(axis=1)` (가능하면 벡터화)

## Shift(1) Rule (Lookahead Bias Prevention)

현재 봉 데이터로 시그널 생성 시 **반드시** `.shift(1)` 사용:

```python
# ❌ Bad (lookahead bias - uses future data)
signal = (df['close'] > df['sma_20']).astype(int)

# ✅ Good (no lookahead)
signal = (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int)
```

## Directory Structure

새 전략 추가 시 다음 구조를 따릅니다:

```
src/strategy/my_strategy/
├── config.py         # Pydantic 설정 모델
├── preprocessor.py   # 지표 계산 (벡터화)
├── signal.py         # 시그널 생성 로직
└── strategy.py       # @register_strategy 메인 클래스
```

## Strategy Registry

```python
from src.strategy.registry import register_strategy

@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    ...
```

CLI에서 자동 조회:
```bash
uv run mcbot backtest strategies
```

## Log Returns

내부 계산은 로그 수익률 사용:

```python
returns = np.log(close / close.shift(1))
```

리포트 생성 시에만 단순 수익률로 변환합니다.
