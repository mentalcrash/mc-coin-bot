---
paths:
  - "src/strategy/**"
---

# Strategy Development Rules

## BaseStrategy Interface

모든 전략은 `BaseStrategy`를 상속하고 다음을 구현해야 합니다:

```python
class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """전략 고유 이름 (예: "VW-TSMOM")"""

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """필수 DataFrame 컬럼 (예: ["close", "volume"])"""

    @property
    def config(self) -> BaseModel | None:
        """전략 설정 (Pydantic 모델). 기본 None"""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 (벡터화 연산만)"""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """시그널 생성 (-1: 매도, 0: 관망, 1: 매수)"""

    # 선택적 메서드 (오버라이드 가능)
    @classmethod
    def from_params(cls, **params) -> BaseStrategy:
        """파라미터 딕셔너리로 인스턴스 생성 (parameter sweep 용)"""

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        """권장 PortfolioManagerConfig 설정 반환"""

    def run_incremental(self, df) -> tuple[DataFrame, StrategySignals]:
        """최신 시그널만 효율적으로 계산 (기본: run() 위임)"""

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널 정보 반환"""
```

## Zero Loop Policy (Critical)

**벡터화 연산만 허용**, 루프 금지:

```python
# Bad (100x slower)
for i in range(len(df)):
    if df['close'].iloc[i] > df['sma_20'].iloc[i]:
        signals.iloc[i] = 1

# Good (vectorized)
signals = np.where(df['close'] > df['sma_20'], 1, 0)
```

**금지 목록:**
- `for` 루프로 DataFrame 순회
- `iterrows()`, `itertuples()`
- `apply(axis=1)` (가능하면 벡터화)

## Shift(1) Rule (Lookahead Bias Prevention)

현재 봉 데이터로 시그널 생성 시 **반드시** `.shift(1)` 사용:

```python
# Bad (lookahead bias)
signal = (df['close'] > df['sma_20']).astype(int)

# Good (no lookahead)
signal = (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int)
```

## Directory Structure

새 전략 추가 시 다음 구조를 따릅니다:

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

## Common Gotchas

- **TYPE_CHECKING import**: `if TYPE_CHECKING:` 블록 안의 import는 런타임에 사용 불가. 런타임 필요 시 블록 밖으로 이동
- **DataFrame 컬럼 타입**: `df["close"]`는 `Series | DataFrame` — pyright 만족시키려면 `pd.Series` 캐스팅 필요
- **from_params() 패턴**: Config 모델을 거쳐 생성해야 하는 전략은 반드시 `from_params()` 오버라이드
- **df.copy()**: Ensemble 등에서 여러 전략에 동일 df 전달 시 `.copy()` 필수 (컬럼 충돌 방지)
