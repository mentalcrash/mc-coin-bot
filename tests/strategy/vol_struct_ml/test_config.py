"""Tests for Volatility Structure ML config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_struct_ml.config import ShortMode, VolStructMLConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolStructMLConfig:
    def test_default_values(self) -> None:
        config = VolStructMLConfig()
        assert config.training_window == 252
        assert config.prediction_horizon == 6
        assert config.alpha == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = VolStructMLConfig()
        with pytest.raises(ValidationError):
            config.training_window = 999  # type: ignore[misc]

    def test_training_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VolStructMLConfig(training_window=10)
        with pytest.raises(ValidationError):
            VolStructMLConfig(training_window=1000)

    def test_alpha_range(self) -> None:
        with pytest.raises(ValidationError):
            VolStructMLConfig(alpha=0.0)
        with pytest.raises(ValidationError):
            VolStructMLConfig(alpha=1.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolStructMLConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolStructMLConfig()
        assert config.warmup_periods() >= config.training_window

    def test_annualization_factor(self) -> None:
        config = VolStructMLConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = VolStructMLConfig(training_window=120, alpha=0.3)
        assert config.training_window == 120
        assert config.alpha == 0.3

    def test_feature_params(self) -> None:
        config = VolStructMLConfig()
        assert config.vol_estimator_window == 20
        assert config.fractal_period == 30
        assert config.hurst_window == 50
        assert config.er_period == 10
        assert config.adx_period == 14
        assert config.vov_window == 20
