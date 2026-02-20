"""Tests for Complexity-Filtered Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.complexity_trend.config import ComplexityTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestComplexityTrendConfig:
    def test_default_values(self) -> None:
        config = ComplexityTrendConfig()
        assert config.hurst_window == 60
        assert config.fractal_period == 30
        assert config.er_period == 21
        assert config.trend_window == 42
        assert config.hurst_threshold == 0.55
        assert config.fractal_threshold == 1.45
        assert config.er_threshold == 0.15
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = ComplexityTrendConfig()
        with pytest.raises(ValidationError):
            config.hurst_window = 999  # type: ignore[misc]

    def test_hurst_window_range(self) -> None:
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(hurst_window=10)
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(hurst_window=500)

    def test_fractal_period_range(self) -> None:
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(fractal_period=5)
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(fractal_period=200)

    def test_er_period_range(self) -> None:
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(er_period=3)
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(er_period=100)

    def test_hurst_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(hurst_threshold=0.30)
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(hurst_threshold=0.90)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ComplexityTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = ComplexityTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        # fractal_dimension uses 2*period internally
        assert config.warmup_periods() >= 2 * config.fractal_period

    def test_annualization_factor(self) -> None:
        config = ComplexityTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = ComplexityTrendConfig(hurst_window=80, er_period=30)
        assert config.hurst_window == 80
        assert config.er_period == 30
