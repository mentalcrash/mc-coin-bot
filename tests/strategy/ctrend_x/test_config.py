"""Tests for CTREND-X config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ctrend_x.config import CTRENDXConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCTRENDXConfig:
    def test_default_values(self) -> None:
        config = CTRENDXConfig()
        assert config.training_window == 252
        assert config.prediction_horizon == 5
        assert config.n_estimators == 100
        assert config.max_depth == 3
        assert config.learning_rate == 0.05
        assert config.subsample == 0.8
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CTRENDXConfig()
        with pytest.raises(ValidationError):
            config.training_window = 999  # type: ignore[misc]

    def test_training_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CTRENDXConfig(training_window=10)
        with pytest.raises(ValidationError):
            CTRENDXConfig(training_window=1000)

    def test_n_estimators_range(self) -> None:
        with pytest.raises(ValidationError):
            CTRENDXConfig(n_estimators=1)
        with pytest.raises(ValidationError):
            CTRENDXConfig(n_estimators=1000)

    def test_max_depth_range(self) -> None:
        with pytest.raises(ValidationError):
            CTRENDXConfig(max_depth=0)
        with pytest.raises(ValidationError):
            CTRENDXConfig(max_depth=20)

    def test_learning_rate_range(self) -> None:
        with pytest.raises(ValidationError):
            CTRENDXConfig(learning_rate=0.0001)
        with pytest.raises(ValidationError):
            CTRENDXConfig(learning_rate=1.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CTRENDXConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CTRENDXConfig()
        assert config.warmup_periods() == 302  # 252 + 50

    def test_annualization_factor(self) -> None:
        config = CTRENDXConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = CTRENDXConfig(training_window=120, n_estimators=50)
        assert config.training_window == 120
        assert config.n_estimators == 50
