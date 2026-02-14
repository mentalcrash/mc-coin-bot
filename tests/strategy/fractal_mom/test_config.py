"""Tests for Fractal-Filtered Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fractal_mom.config import FractalMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFractalMomConfig:
    def test_default_values(self) -> None:
        config = FractalMomConfig()
        assert config.fractal_period == 30
        assert config.fractal_threshold == 1.5
        assert config.mom_fast == 12
        assert config.mom_slow == 48
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FractalMomConfig()
        with pytest.raises(ValidationError):
            config.fractal_period = 999  # type: ignore[misc]

    def test_fractal_period_range(self) -> None:
        with pytest.raises(ValidationError):
            FractalMomConfig(fractal_period=5)
        with pytest.raises(ValidationError):
            FractalMomConfig(fractal_period=200)

    def test_fractal_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            FractalMomConfig(fractal_threshold=1.0)
        with pytest.raises(ValidationError):
            FractalMomConfig(fractal_threshold=2.0)

    def test_mom_fast_must_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            FractalMomConfig(mom_fast=50, mom_slow=50)
        with pytest.raises(ValidationError):
            FractalMomConfig(mom_fast=60, mom_slow=50)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FractalMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FractalMomConfig()
        assert config.warmup_periods() >= config.mom_slow

    def test_annualization_factor(self) -> None:
        config = FractalMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = FractalMomConfig(fractal_period=50, fractal_threshold=1.4)
        assert config.fractal_period == 50
        assert config.fractal_threshold == 1.4
