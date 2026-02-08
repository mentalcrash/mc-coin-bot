---
name: new-strategy
description: >
  새 트레이딩 전략을 프로젝트 표준 4-파일 구조로 스캐폴딩한다.
  config.py, preprocessor.py, signal.py, strategy.py + 테스트 4개 + YAML 설정을 생성.
  사용 시점: 전략 구현, 전략 추가, 새 알고리즘 포팅, 리서치 문서의 전략 구현 시.
argument-hint: <strategy-name>
---

# New Strategy — 전략 스캐폴딩

새 트레이딩 전략을 프로젝트 표준 패턴에 맞게 생성한다.

## Pre-flight 체크

1. **이름 중복 확인**: `src/strategy/` 하위에 동일 디렉토리 존재 여부
2. **ShortMode 결정**: Long-Only(DISABLED=0) / Hedge(HEDGE_ONLY=1) / Full Short(FULL=2)
3. **필요 지표 식별**: 사용할 기술적 지표와 라이브러리 (pandas, numpy, numba)
4. **레지스트리 이름**: kebab-case (e.g., `"vol-regime"`, `"bb-rsi"`)

## 생성 파일 목록 (9개)

### 소스 파일 (5개)

| # | 파일 | 역할 |
|---|------|------|
| 1 | `src/strategy/{name}/__init__.py` | Public exports |
| 2 | `src/strategy/{name}/config.py` | Pydantic frozen config + sweep ranges |
| 3 | `src/strategy/{name}/preprocessor.py` | Vectorized indicator 계산 |
| 4 | `src/strategy/{name}/signal.py` | shift(1) 적용 + StrategySignals 반환 |
| 5 | `src/strategy/{name}/strategy.py` | `@register()`, `from_params()`, `recommended_config()` |

### 테스트 파일 (4개)

| # | 파일 | 역할 |
|---|------|------|
| 6 | `tests/strategy/{name}/test_config.py` | Config validation + sweep ranges |
| 7 | `tests/strategy/{name}/test_preprocessor.py` | Indicator 정확성 |
| 8 | `tests/strategy/{name}/test_signal.py` | Signal 타이밍 + ShortMode |
| 9 | `tests/strategy/{name}/test_strategy.py` | Registry + pipeline + factory |

## 생성 순서

### Step 1: `config.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.strategy.tsmom.config import ShortMode  # 공통 ShortMode 재사용

if TYPE_CHECKING:
    pass


class {Name}Config(BaseModel):
    """
    {StrategyName} 전략 설정.

    Attributes:
        short_mode: ShortMode enum (DISABLED/HEDGE_ONLY/FULL)
        # ... 전략 특화 파라미터
    """

    model_config = {"frozen": True}

    short_mode: ShortMode = ShortMode.DISABLED
    # TODO: 전략 특화 파라미터 추가

    # --- Sweep Ranges (classmethod) ---
    @classmethod
    def sweep_ranges(cls) -> dict[str, list[object]]:
        """파라미터 스윕용 범위 정의."""
        return {
            "short_mode": [ShortMode.DISABLED, ShortMode.HEDGE_ONLY, ShortMode.FULL],
            # TODO: 전략 특화 범위 추가
        }
```

**필수 패턴:**
- `model_config = {"frozen": True}` — immutable config
- `ShortMode` enum 사용 (DISABLED=0, HEDGE_ONLY=1, FULL=2)
- `sweep_ranges()` classmethod 포함
- 모든 필드에 합리적 기본값

### Step 2: `preprocessor.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from src.strategy.{name}.config import {Name}Config


def preprocess(df: pd.DataFrame, config: {Name}Config) -> pd.DataFrame:
    """
    벡터화된 지표 계산.

    주의: 이 함수에서 shift() 사용 금지.
    shift(1)은 signal.py에서만 적용.
    """
    result = df.copy()
    # TODO: 지표 계산 (vectorized operations only)
    return result
```

**필수 패턴:**
- `df.copy()` 후 작업 (원본 불변)
- `shift()` 절대 사용 금지 (signal.py 전담)
- vectorized ops only (`iterrows()` 금지)
- NaN 초기 윈도우 인지

### Step 3: `signal.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from src.models.strategy import StrategySignals
from src.strategy.{name}.config import {Name}Config, ShortMode


def generate_signals(df: pd.DataFrame, config: {Name}Config) -> StrategySignals:
    """
    시그널 생성.

    핵심: signal_shifted = raw_signal.shift(1)
    Signal[t] at Close → Execute at Open[t+1]
    """
    import pandas as pd as _pd

    # 1. Raw signal 계산
    raw_signal: pd.Series = ...  # TODO

    # 2. shift(1) 적용 (미래 참조 방지)
    signal_shifted: pd.Series = raw_signal.shift(1).fillna(0.0)

    # 3. ShortMode 적용
    if config.short_mode == ShortMode.DISABLED:
        signal_shifted = signal_shifted.clip(lower=0.0)

    # 4. Entry/Exit 생성
    direction = np.sign(signal_shifted).astype(int)
    entries = (direction != 0) & (direction != direction.shift(1))
    exits = (direction == 0) & (direction.shift(1) != 0)

    return StrategySignals(
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        direction=direction,
        strength=signal_shifted.abs(),
    )
```

**필수 패턴:**
- `shift(1)` 반드시 적용
- `ShortMode` 분기 처리
- `StrategySignals` 반환 (entries, exits, direction, strength)
- NaN → `fillna()` 처리

### Step 4: `strategy.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from src.models.strategy import StrategySignals
from src.strategy.base import BaseStrategy
from src.strategy.{name}.config import {Name}Config
from src.strategy.{name}.preprocessor import preprocess
from src.strategy.{name}.signal import generate_signals
from src.strategy.registry import register


@register("{registry-name}")
class {Name}Strategy(BaseStrategy):
    """
    {StrategyName} 전략.

    설명: ...
    """

    def __init__(self, config: {Name}Config | None = None) -> None:
        self._config = config or {Name}Config()

    @classmethod
    def from_params(cls, **params: object) -> {Name}Strategy:
        config = {Name}Config(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "{StrategyName}"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> {Name}Config:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        preprocessed = self.preprocess(df)
        return generate_signals(preprocessed, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
        }

    def warmup_periods(self) -> int:
        # TODO: max(사용 지표의 window 크기) + 여유분
        return 50

    def get_startup_info(self) -> dict[str, str]:
        return {
            "Strategy": self.name,
            # TODO: 핵심 파라미터 추가
        }

    @classmethod
    def conservative(cls) -> {Name}Strategy:
        """보수적 설정 팩토리."""
        return cls({Name}Config(
            # TODO: 보수적 파라미터
        ))
```

**필수 패턴:**
- `@register("{registry-name}")` 데코레이터
- `from_params(**params)` classmethod
- `recommended_config()` classmethod → PM 설정 권장값
- `warmup_periods()` → NaN 구간 크기
- `generate_signals()`에서 `preprocess()` 먼저 호출

### Step 5: `__init__.py`

```python
from src.strategy.{name}.config import {Name}Config
from src.strategy.{name}.strategy import {Name}Strategy

__all__ = ["{Name}Config", "{Name}Strategy"]
```

### Step 6: 레지스트리 등록

`src/strategy/__init__.py`에 side-effect import 추가:

```python
import src.strategy.{name}  # noqa: F401 — register() 트리거
```

### Step 7: 테스트 생성

[references/strategy-template.md](references/strategy-template.md) 참조.

4개 테스트 파일 생성, 각각 `pytest.fixture`로 `sample_ohlcv_df` 공유.

## 검증 단계

생성 완료 후 **반드시** 실행:

```bash
# 1. Lint + Format
uv run ruff check --fix src/strategy/{name}/ tests/strategy/{name}/
uv run ruff format src/strategy/{name}/ tests/strategy/{name}/

# 2. Type check
uv run pyright src/strategy/{name}/

# 3. 테스트
uv run pytest tests/strategy/{name}/ -v

# 4. 레지스트리 확인
uv run python -c "from src.strategy import list_strategies; print(list_strategies())"

# 5. EDA 실행 확인 (선택)
uv run python main.py eda run {registry-name} BTC/USDT --start 2024-01-01 --end 2025-01-01
```

## 체크리스트

- [ ] `@register()` 이름이 kebab-case
- [ ] `from_params()` 에서 Config 생성 가능
- [ ] `signal.py`에 `shift(1)` 적용됨
- [ ] `preprocessor.py`에 `shift()` 없음
- [ ] `ShortMode` 3가지 모드 테스트
- [ ] `recommended_config()` 반환값에 `max_leverage_cap` 포함
- [ ] `__init__.py` side-effect import 추가됨
- [ ] ruff + pyright + pytest 0 error
