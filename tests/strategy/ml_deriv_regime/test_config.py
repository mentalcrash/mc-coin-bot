"""Tests for ML Derivatives Regime config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMlDerivRegimeConfig:
    def test_default_values(self) -> None:
        config = MlDerivRegimeConfig()
        assert config.training_window == 252
        assert config.prediction_horizon == 5
        assert config.alpha == 0.5
        assert config.fr_lookback_short == 3
        assert config.fr_lookback_long == 21
        assert config.fr_zscore_window == 63
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = MlDerivRegimeConfig()
        with pytest.raises(ValidationError):
            config.training_window = 999  # type: ignore[misc]

    def test_training_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MlDerivRegimeConfig(training_window=30)
        with pytest.raises(ValidationError):
            MlDerivRegimeConfig(training_window=600)

    def test_alpha_range(self) -> None:
        with pytest.raises(ValidationError):
            MlDerivRegimeConfig(alpha=0.0)
        with pytest.raises(ValidationError):
            MlDerivRegimeConfig(alpha=1.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MlDerivRegimeConfig(vol_target=0.01, min_volatility=0.05)

    def test_fr_lookback_short_lt_long(self) -> None:
        with pytest.raises(ValidationError):
            MlDerivRegimeConfig(fr_lookback_short=30, fr_lookback_long=10)

    def test_warmup_periods(self) -> None:
        config = MlDerivRegimeConfig()
        assert config.warmup_periods() >= config.training_window

    def test_annualization_factor(self) -> None:
        config = MlDerivRegimeConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = MlDerivRegimeConfig(training_window=126, alpha=0.8)
        assert config.training_window == 126
        assert config.alpha == 0.8
