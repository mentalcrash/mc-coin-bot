"""Tests for Vol-Term ML config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_term_ml.config import ShortMode, VolTermMLConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolTermMLConfig:
    def test_default_values(self) -> None:
        config = VolTermMLConfig()
        assert config.training_window == 252
        assert config.prediction_horizon == 5
        assert config.ridge_alpha == 1.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = VolTermMLConfig()
        with pytest.raises(ValidationError):
            config.training_window = 999  # type: ignore[misc]

    def test_training_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VolTermMLConfig(training_window=10)
        with pytest.raises(ValidationError):
            VolTermMLConfig(training_window=1000)

    def test_ridge_alpha_range(self) -> None:
        with pytest.raises(ValidationError):
            VolTermMLConfig(ridge_alpha=0.001)
        with pytest.raises(ValidationError):
            VolTermMLConfig(ridge_alpha=200.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolTermMLConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolTermMLConfig()
        assert config.warmup_periods() == 322  # 252 + 70

    def test_annualization_factor(self) -> None:
        config = VolTermMLConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = VolTermMLConfig(training_window=120, ridge_alpha=5.0)
        assert config.training_window == 120
        assert config.ridge_alpha == 5.0
