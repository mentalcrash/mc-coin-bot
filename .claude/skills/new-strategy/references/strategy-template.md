# 전략 테스트 템플릿

TSMOM/Breakout 패턴 기반 표준 테스트 구조.

---

## test_config.py 템플릿

```python
from __future__ import annotations

import pytest

from src.strategy.{name}.config import {Name}Config, ShortMode


class Test{Name}Config:
    """Config validation 테스트."""

    def test_default_config(self) -> None:
        config = {Name}Config()
        assert config.short_mode == ShortMode.DISABLED
        # TODO: 기본값 검증

    def test_frozen_model(self) -> None:
        config = {Name}Config()
        with pytest.raises(Exception):  # ValidationError
            config.short_mode = ShortMode.FULL  # type: ignore[misc]

    def test_custom_config(self) -> None:
        config = {Name}Config(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_sweep_ranges(self) -> None:
        ranges = {Name}Config.sweep_ranges()
        assert "short_mode" in ranges
        assert len(ranges) >= 1

    @pytest.mark.parametrize(
        "short_mode",
        [ShortMode.DISABLED, ShortMode.HEDGE_ONLY, ShortMode.FULL],
    )
    def test_all_short_modes(self, short_mode: ShortMode) -> None:
        config = {Name}Config(short_mode=short_mode)
        assert config.short_mode == short_mode
```

---

## test_preprocessor.py 템플릿

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.{name}.config import {Name}Config
from src.strategy.{name}.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """100일치 합성 OHLCV 데이터."""
    np.random.seed(42)
    n = 100
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


class Test{Name}Preprocessor:
    def test_returns_dataframe(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = {Name}Config()
        result = preprocess(sample_ohlcv_df, config)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = {Name}Config()
        result = preprocess(sample_ohlcv_df, config)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_no_shift_in_preprocessor(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """preprocessor에서 shift 사용하지 않는지 소스코드 레벨 확인."""
        import inspect
        source = inspect.getsource(preprocess)
        # shift(1)은 signal.py에서만 적용
        assert "shift(1)" not in source or "shift(-" not in source

    def test_adds_indicator_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = {Name}Config()
        result = preprocess(sample_ohlcv_df, config)
        # TODO: 추가된 지표 컬럼 확인
        assert len(result.columns) > len(sample_ohlcv_df.columns)

    def test_does_not_modify_input(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = {Name}Config()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
```

---

## test_signal.py 템플릿

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.{name}.config import {Name}Config, ShortMode
from src.strategy.{name}.preprocessor import preprocess
from src.strategy.{name}.signal import generate_signals


@pytest.fixture
def preprocessed_df() -> pd.DataFrame:
    """전처리 완료된 DataFrame."""
    np.random.seed(42)
    n = 100
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))
    return preprocess(df, {Name}Config())


class Test{Name}Signal:
    def test_returns_strategy_signals(self, preprocessed_df: pd.DataFrame) -> None:
        signals = generate_signals(preprocessed_df, {Name}Config())
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_signal_shift_applied(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 행의 시그널은 반드시 NaN → fillna(0)."""
        signals = generate_signals(preprocessed_df, {Name}Config())
        # shift(1) 적용 → 첫 행은 0이어야 함
        assert signals.direction.iloc[0] == 0

    def test_long_only_mode(self, preprocessed_df: pd.DataFrame) -> None:
        config = {Name}Config(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)
        assert (signals.direction >= 0).all()

    def test_full_short_mode(self, preprocessed_df: pd.DataFrame) -> None:
        config = {Name}Config(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)
        # Short 방향(-1) 시그널이 존재할 수 있음
        assert signals.direction.min() >= -1

    def test_entries_exits_boolean(self, preprocessed_df: pd.DataFrame) -> None:
        signals = generate_signals(preprocessed_df, {Name}Config())
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame) -> None:
        signals = generate_signals(preprocessed_df, {Name}Config())
        unique = set(signals.direction.unique())
        assert unique.issubset({-1, 0, 1})
```

---

## test_strategy.py 템플릿

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.{name}.config import {Name}Config
from src.strategy.{name}.strategy import {Name}Strategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


class Test{Name}Strategy:
    def test_strategy_registration(self) -> None:
        assert "{registry-name}" in list_strategies()

    def test_get_strategy(self) -> None:
        strategy_cls = get_strategy("{registry-name}")
        assert strategy_cls is {Name}Strategy

    def test_strategy_properties(self) -> None:
        strategy = {Name}Strategy()
        assert strategy.name == "{StrategyName}"
        assert "close" in strategy.required_columns
        assert isinstance(strategy.config, {Name}Config)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = {Name}Strategy()
        signals = strategy.run(sample_ohlcv_df)
        assert hasattr(signals, "entries")
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = {Name}Strategy.from_params(short_mode=1)
        assert strategy.config.short_mode.value == 1

    def test_recommended_config(self) -> None:
        rc = {Name}Strategy.recommended_config()
        assert "max_leverage_cap" in rc
        assert rc["max_leverage_cap"] > 0

    def test_warmup_periods(self) -> None:
        strategy = {Name}Strategy()
        assert strategy.warmup_periods() > 0

    def test_factory_conservative(self) -> None:
        strategy = {Name}Strategy.conservative()
        assert isinstance(strategy, {Name}Strategy)
```

---

## 네이밍 규칙

| 항목 | 규칙 | 예시 |
|------|------|------|
| 디렉토리명 | snake_case | `vol_regime`, `bb_rsi` |
| 레지스트리명 | kebab-case | `"vol-regime"`, `"bb-rsi"` |
| 클래스명 | PascalCase | `VolRegimeStrategy`, `BBRSIStrategy` |
| Config명 | PascalCase + Config | `VolRegimeConfig`, `BBRSIConfig` |
