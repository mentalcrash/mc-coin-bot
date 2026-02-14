# Code Templates — p2-implement

> 각 파일의 완전한 코드 템플릿. `{placeholder}`는 전략별 값으로 대체.

---

## 1. config.py Template

```python
"""{StrategyDisplayName} 전략 설정.

{한줄 설명 — 핵심 가설 요약}.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class {StrategyConfig}(BaseModel):
    """{StrategyDisplayName} 전략 설정.

    Attributes:
        {param1}: {설명}
        {param2}: {설명}
        vol_target: 연환산 변동성 타겟 (0~1)
        vol_window: 변동성 계산 rolling window
        min_volatility: 변동성 하한 (0 나눗셈 방지)
        annualization_factor: TF별 연환산 계수
        short_mode: 숏 포지션 허용 모드
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0)
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    {param1}: {type} = Field(default={default}, ge={min}, le={max})
    {param2}: {type} = Field(default={default}, ge={min}, le={max})

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default={TF_FACTOR})

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.{DEFAULT_MODE})
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> {StrategyConfig}:
        if self.vol_target < self.min_volatility:
            msg = f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max({largest_window}, self.vol_window) + 10
```

### 참고: annualization_factor 값

| TF | Factor |
|----|--------|
| 1D | `365.0` |
| 12H | `730.0` |
| 8H | `1095.0` |
| 6H | `1460.0` |
| 4H | `2190.0` |
| 1H | `8760.0` |

---

## 2. preprocessor.py Template

```python
"""{StrategyDisplayName} 전처리 모듈.

OHLCV 데이터에서 전략 feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.{name_snake}.config import {StrategyConfig}

# 공통 지표 라이브러리 (중복 구현 금지, 패키지 구조)
from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)


_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: {StrategyConfig}) -> pd.DataFrame:
    """{StrategyDisplayName} feature 계산.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    open_: pd.Series = df["open"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Strategy-Specific Features ---
    # TODO: 전략별 feature 구현
    # df["{feature_name}"] = ...

    # --- Drawdown (HEDGE_ONLY용) ---
    # cumulative = (1 + returns).cumprod()
    # running_max = cumulative.cummax()
    # df["drawdown"] = (cumulative - running_max) / running_max

    # --- ATR (trailing stop용) ---
    # tr = pd.concat([
    #     high - low,
    #     (high - close.shift(1)).abs(),
    #     (low - close.shift(1)).abs(),
    # ], axis=1).max(axis=1)
    # df["atr"] = tr.rolling(14).mean()

    return df
```

### 공통 지표 함수 시그니처 (src/market/indicators/)

패키지 구조로 재구성되었으며, 53개 함수 사용 가능. 주요 함수:

```python
# Returns
def log_returns(close: pd.Series) -> pd.Series: ...
def simple_returns(close: pd.Series) -> pd.Series: ...
def rolling_return(close: pd.Series, window: int) -> pd.Series: ...

# Volatility
def realized_volatility(
    returns: pd.Series,
    window: int = 30,
    annualization_factor: float = 365.0,
    min_periods: int | None = None,
) -> pd.Series: ...
def volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float = 0.35,
    min_volatility: float = 0.05,
) -> pd.Series: ...
def parkinson_volatility(high: pd.Series, low: pd.Series, window: int, annualization_factor: float) -> pd.Series: ...
def garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int, annualization_factor: float) -> pd.Series: ...

# Trend
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series: ...
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series: ...
def sma(close: pd.Series, period: int) -> pd.Series: ...
def ema(close: pd.Series, period: int) -> pd.Series: ...
def kama(close: pd.Series, period: int = 10, fast_ema: int = 2, slow_ema: int = 30) -> pd.Series: ...

# Oscillators
def rsi(close: pd.Series, period: int = 14) -> pd.Series: ...
def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series: ...
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]: ...

# Channels
def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]: ...
def donchian_channel(high: pd.Series, low: pd.Series, period: int) -> tuple[pd.Series, pd.Series]: ...

# Composite
def drawdown(close: pd.Series) -> pd.Series: ...
def rolling_zscore(close: pd.Series, window: int) -> pd.Series: ...
def trend_strength(close: pd.Series, period: int = 20) -> pd.Series: ...

# Derivatives
def funding_rate_ma(funding_rate: pd.Series, window: int) -> pd.Series: ...
def funding_zscore(funding_rate: pd.Series, window: int) -> pd.Series: ...
def oi_momentum(open_interest: pd.Series, window: int) -> pd.Series: ...
```

전체 목록은 `src/market/indicators/__init__.py` 참조.
```

---

## 2.5 Derivatives-Aware Preprocessor Template

Derivatives 데이터를 사용하는 전략의 preprocessor 템플릿.
참조 구현: `src/strategy/funding_carry/preprocessor.py`

```python
"""{StrategyDisplayName} 전처리 모듈 (Derivatives).

OHLCV + Derivatives 데이터에서 전략 feature를 계산한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.{name_snake}.config import {StrategyConfig}

from src.market.indicators import (
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})


def preprocess(df: pd.DataFrame, config: {StrategyConfig}) -> pd.DataFrame:
    """{StrategyDisplayName} feature 계산 (Derivatives 포함)."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Derivatives Feature ---
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    funding_rate = funding_rate.ffill()  # merge_asof 후 NaN 처리
    df["avg_funding_rate"] = funding_rate.rolling(config.lookback).mean()
    rolling_mean = df["avg_funding_rate"].rolling(config.zscore_window).mean()
    rolling_std = df["avg_funding_rate"].rolling(config.zscore_window).std()
    df["funding_zscore"] = (df["avg_funding_rate"] - rolling_mean) / rolling_std.clip(lower=1e-10)

    return df
```

---

## 3. signal.py Template

```python
"""{StrategyDisplayName} 시그널 생성.

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.{name_snake}.config import {StrategyConfig}, ShortMode


def generate_signals(df: pd.DataFrame, config: {StrategyConfig}) -> StrategySignals:
    """{StrategyDisplayName} 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.{name_snake}.config import ShortMode

    # --- Shift(1): 전봉 기준 시그널 ---
    # feature_a = df["{feature_a}"].shift(1)
    # feature_b = df["{feature_b}"].shift(1)
    vol_scalar = df["vol_scalar"].shift(1)

    # --- Signal Logic ---
    # long_signal = (feature_a > config.{threshold})
    # short_signal = (feature_a < -config.{threshold})

    # --- Direction (ShortMode 분기) ---
    direction = _compute_direction(
        # long_signal=long_signal,
        # short_signal=short_signal,
        df=df,
        config=config,
    )

    # --- Strength ---
    strength = direction.astype(float) * vol_scalar.fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # --- Entries / Exits ---
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_dir)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _compute_direction(
    # long_signal: pd.Series,
    # short_signal: pd.Series,
    df: pd.DataFrame,
    config: {StrategyConfig},
) -> pd.Series:
    """ShortMode 3-way 분기로 direction 계산."""
    from src.strategy.{name_snake}.config import ShortMode

    # TODO: 전략별 시그널 로직 구현
    long_signal = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)

    if config.short_mode == ShortMode.DISABLED:
        raw = np.where(long_signal, 1, 0)

    elif config.short_mode == ShortMode.HEDGE_ONLY:
        drawdown = df["drawdown"].shift(1)
        hedge_active = drawdown < config.hedge_threshold
        raw = np.where(
            long_signal,
            1,
            np.where(short_signal & hedge_active, -1, 0),
        )

    else:  # FULL
        raw = np.where(long_signal, 1, np.where(short_signal, -1, 0))

    return pd.Series(raw, index=df.index, dtype=int)
```

---

## 3.5 Regime-Adaptive signal.py Template (해당 시)

RegimeService 자동 주입 컬럼을 활용하는 signal.py 추가 패턴.
기본 signal.py Template의 확장이며, 레짐 적응형 전략에만 적용.

```python
"""{StrategyDisplayName} 시그널 생성 (레짐 적응형).

RegimeService가 주입하는 regime 컬럼을 활용하여
vol_target/threshold 등을 레짐 확률에 따라 적응적으로 조절합니다.
regime 컬럼이 없으면 기본 config.vol_target으로 fallback합니다.

Shift(1) Rule: 모든 feature + regime 컬럼에 shift(1) 적용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.{name_snake}.config import {StrategyConfig}, ShortMode


# RegimeService가 주입하는 컬럼 목록
_REGIME_COLUMNS = ("p_trending", "p_ranging", "p_volatile", "regime_label",
                   "trend_direction", "trend_strength")


def _has_regime_columns(df: pd.DataFrame) -> bool:
    """DataFrame에 regime 컬럼이 있는지 확인."""
    return "p_trending" in df.columns


def _compute_adaptive_vol_target(
    df: pd.DataFrame,
    config: {StrategyConfig},
) -> pd.Series:
    """레짐 확률 가중 adaptive vol_target 계산.

    regime 컬럼이 없으면 config.vol_target 상수 반환.
    """
    if not _has_regime_columns(df):
        return pd.Series(config.vol_target, index=df.index)

    p_trending = df["p_trending"].shift(1).fillna(1 / 3)
    p_ranging = df["p_ranging"].shift(1).fillna(1 / 3)
    p_volatile = df["p_volatile"].shift(1).fillna(1 / 3)

    adaptive: pd.Series = (  # type: ignore[assignment]
        p_trending * config.trending_vol_target
        + p_ranging * config.ranging_vol_target
        + p_volatile * config.volatile_vol_target
    )
    return adaptive


def generate_signals(df: pd.DataFrame, config: {StrategyConfig}) -> StrategySignals:
    """{StrategyDisplayName} 시그널 생성 (레짐 적응형)."""
    from src.strategy.{name_snake}.config import ShortMode

    # --- Regime-Adaptive Vol Target ---
    adaptive_vol_target = _compute_adaptive_vol_target(df, config)

    # --- Vol Scalar (adaptive) ---
    realized_vol = df["realized_vol"].shift(1)
    clamped_vol = realized_vol.clip(lower=config.min_volatility)
    vol_scalar = adaptive_vol_target / clamped_vol

    # --- Strategy-Specific Signal Logic ---
    # feature_a = df["{feature_a}"].shift(1)
    # long_signal = ...
    # short_signal = ...

    # --- Direction (ShortMode 분기) ---
    # direction = _compute_direction(...)

    # --- Strength ---
    # strength = direction.astype(float) * vol_scalar.fillna(0)

    # ... (이하 기본 signal.py Template과 동일)
```

### 레짐 적응형 config.py 추가 필드

```python
    # --- 레짐별 적응적 파라미터 (RegimeService 자동 주입 컬럼 활용) ---
    trending_vol_target: float = Field(
        default=0.40, ge=0.05, le=1.0,
        description="trending 레짐에서의 vol_target (공격적)",
    )
    ranging_vol_target: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="ranging 레짐에서의 vol_target (보수적)",
    )
    volatile_vol_target: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="volatile 레짐에서의 vol_target (초보수)",
    )
```

### 레짐 적응형 test_signal.py 추가 테스트

```python
class TestRegimeAdaptation:
    """RegimeService 컬럼 활용 테스트."""

    def test_with_regime_columns(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        """regime 컬럼이 있을 때 adaptive vol_target 사용."""
        df = preprocessed_df.copy()
        df["p_trending"] = 0.8
        df["p_ranging"] = 0.1
        df["p_volatile"] = 0.1
        df["regime_label"] = "trending"
        df["trend_direction"] = 1
        df["trend_strength"] = 0.7
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)

    def test_without_regime_columns(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        """regime 컬럼 없이도 fallback으로 정상 작동."""
        signals = generate_signals(preprocessed_df, config)
        assert len(signals.entries) == len(preprocessed_df)

    def test_trending_more_aggressive(
        self, preprocessed_df: pd.DataFrame
    ) -> None:
        """trending에서 더 높은 평균 |strength| 확인."""
        config_default = {StrategyConfig}()
        df_trending = preprocessed_df.copy()
        df_trending["p_trending"] = 1.0
        df_trending["p_ranging"] = 0.0
        df_trending["p_volatile"] = 0.0
        df_trending["regime_label"] = "trending"
        df_trending["trend_direction"] = 1
        df_trending["trend_strength"] = 0.8

        df_volatile = preprocessed_df.copy()
        df_volatile["p_trending"] = 0.0
        df_volatile["p_ranging"] = 0.0
        df_volatile["p_volatile"] = 1.0
        df_volatile["regime_label"] = "volatile"
        df_volatile["trend_direction"] = 0
        df_volatile["trend_strength"] = 0.0

        sig_t = generate_signals(df_trending, config_default)
        sig_v = generate_signals(df_volatile, config_default)

        # trending의 평균 |strength|가 volatile보다 커야 함
        avg_t = sig_t.strength.abs().mean()
        avg_v = sig_v.strength.abs().mean()
        assert avg_t >= avg_v
```

---

## 4. strategy.py Template

```python
"""{StrategyDisplayName} 전략.

{한줄 설명}.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.{name_snake}.config import {StrategyConfig}
from src.strategy.{name_snake}.preprocessor import preprocess
from src.strategy.{name_snake}.signal import generate_signals


@register("{registry-key}")
class {StrategyClass}(BaseStrategy):
    """{StrategyDisplayName} 전략 구현.

    {핵심 가설 1줄 요약}.
    """

    def __init__(self, config: {StrategyConfig} | None = None) -> None:
        self._config = config or {StrategyConfig}()

    @property
    def name(self) -> str:
        return "{registry-key}"

    @property
    def required_columns(self) -> list[str]:
        # Derivatives 전략: ["close", "high", "low", "volume", "funding_rate"] 등 추가
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BaseModel:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = {StrategyConfig}(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "{param1}": str(self._config.{param1}),
            "{param2}": str(self._config.{param2}),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
```

---

## 5. __init__.py Template

```python
"""{StrategyDisplayName}: {한줄 설명}."""

from src.strategy.{name_snake}.config import {StrategyConfig}, ShortMode
from src.strategy.{name_snake}.preprocessor import preprocess
from src.strategy.{name_snake}.signal import generate_signals
from src.strategy.{name_snake}.strategy import {StrategyClass}

__all__ = [
    "{StrategyConfig}",
    "{StrategyClass}",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
```

---

## 6. Test Templates

### test_config.py

```python
"""Tests for {StrategyDisplayName} config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.{name_snake}.config import {StrategyConfig}, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class Test{StrategyConfig}:
    def test_default_values(self) -> None:
        config = {StrategyConfig}()
        assert config.{param1} == {default1}
        assert config.vol_target == 0.35
        assert config.annualization_factor == {TF_FACTOR}
        assert config.short_mode == ShortMode.{DEFAULT_MODE}

    def test_frozen(self) -> None:
        config = {StrategyConfig}()
        with pytest.raises(ValidationError):
            config.{param1} = 999  # type: ignore[misc]

    def test_{param1}_range(self) -> None:
        with pytest.raises(ValidationError):
            {StrategyConfig}({param1}={too_low})
        with pytest.raises(ValidationError):
            {StrategyConfig}({param1}={too_high})

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            {StrategyConfig}(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = {StrategyConfig}()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = {StrategyConfig}()
        assert config.annualization_factor == {TF_FACTOR}

    def test_custom_params(self) -> None:
        config = {StrategyConfig}({param1}={custom_val})
        assert config.{param1} == {custom_val}
```

### test_preprocessor.py

```python
"""Tests for {StrategyDisplayName} preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.{name_snake}.config import {StrategyConfig}
from src.strategy.{name_snake}.preprocessor import preprocess


@pytest.fixture
def config() -> {StrategyConfig}:
    return {StrategyConfig}()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="{freq}"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {"returns", "realized_vol", "vol_scalar"}  # + 전략별 feature
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: {StrategyConfig}) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
```

### test_signal.py

```python
"""Tests for {StrategyDisplayName} signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.{name_snake}.config import {StrategyConfig}, ShortMode
from src.strategy.{name_snake}.preprocessor import preprocess
from src.strategy.{name_snake}.signal import generate_signals


@pytest.fixture
def config() -> {StrategyConfig}:
    return {StrategyConfig}()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="{freq}"),
    )


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame, config: {StrategyConfig}
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: {StrategyConfig}
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = {StrategyConfig}(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = {StrategyConfig}(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # FULL 모드에서는 -1이 가능해야 함 (데이터에 따라)
        assert signals.direction.dtype == int
```

### test_strategy.py

```python
"""Tests for {StrategyDisplayName} strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.{name_snake}.config import {StrategyConfig}
from src.strategy.{name_snake}.strategy import {StrategyClass}


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="{freq}"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "{registry-key}" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("{registry-key}")
        assert cls is {StrategyClass}


class Test{StrategyClass}:
    def test_name(self) -> None:
        strategy = {StrategyClass}()
        assert strategy.name == "{registry-key}"

    def test_required_columns(self) -> None:
        strategy = {StrategyClass}()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = {StrategyClass}()
        assert isinstance(strategy.config, {StrategyConfig})

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = {StrategyClass}()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = {StrategyClass}()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = {StrategyClass}()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = {StrategyClass}.from_params({param1}={custom_val})
        assert isinstance(strategy, {StrategyClass})

    def test_recommended_config(self) -> None:
        config = {StrategyClass}.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = {StrategyClass}()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = {StrategyConfig}({param1}={custom_val})
        strategy = {StrategyClass}(config=config)
        assert strategy._config.{param1} == {custom_val}  # noqa: SLF001

    def test_params_property(self) -> None:
        strategy = {StrategyClass}()
        params = strategy.params
        assert isinstance(params, dict)
        assert "{param1}" in params

    def test_repr(self) -> None:
        strategy = {StrategyClass}()
        assert "{registry-key}" in strategy.name
        assert repr(strategy)  # truthy (not empty)
```
