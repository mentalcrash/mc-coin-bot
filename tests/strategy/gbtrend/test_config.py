"""Tests for GBTrend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.gbtrend.config import GBTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestGBTrendConfig:
    def test_default_values(self) -> None:
        config = GBTrendConfig()
        assert config.training_window == 180
        assert config.prediction_horizon == 5
        assert config.n_estimators == 80
        assert config.max_depth == 3
        assert config.learning_rate == 0.05
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = GBTrendConfig()
        with pytest.raises(ValidationError):
            config.training_window = 999  # type: ignore[misc]

    def test_training_window_range(self) -> None:
        with pytest.raises(ValidationError):
            GBTrendConfig(training_window=10)
        with pytest.raises(ValidationError):
            GBTrendConfig(training_window=1000)

    def test_n_estimators_range(self) -> None:
        with pytest.raises(ValidationError):
            GBTrendConfig(n_estimators=1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            GBTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = GBTrendConfig()
        assert config.warmup_periods() == 250  # 180 + 70

    def test_annualization_factor(self) -> None:
        config = GBTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = GBTrendConfig(training_window=120, n_estimators=50)
        assert config.training_window == 120
        assert config.n_estimators == 50
