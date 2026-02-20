"""Tests for Residual Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.residual_mom.config import ResidualMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestResidualMomConfig:
    def test_default_values(self) -> None:
        config = ResidualMomConfig()
        assert config.regression_window == 60
        assert config.residual_lookback == 21
        assert config.entry_threshold == 1.0
        assert config.exit_threshold == 0.3
        assert config.zscore_window == 60
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = ResidualMomConfig()
        with pytest.raises(ValidationError):
            config.regression_window = 999  # type: ignore[misc]

    def test_regression_window_range(self) -> None:
        with pytest.raises(ValidationError):
            ResidualMomConfig(regression_window=5)
        with pytest.raises(ValidationError):
            ResidualMomConfig(regression_window=500)

    def test_residual_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            ResidualMomConfig(residual_lookback=2)
        with pytest.raises(ValidationError):
            ResidualMomConfig(residual_lookback=200)

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            ResidualMomConfig(entry_threshold=0.0)
        with pytest.raises(ValidationError):
            ResidualMomConfig(entry_threshold=5.0)

    def test_exit_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            ResidualMomConfig(exit_threshold=-0.1)
        with pytest.raises(ValidationError):
            ResidualMomConfig(exit_threshold=3.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ResidualMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_exit_lt_entry_threshold(self) -> None:
        """exit_threshold must be strictly less than entry_threshold."""
        with pytest.raises(ValidationError):
            ResidualMomConfig(entry_threshold=1.0, exit_threshold=1.0)
        with pytest.raises(ValidationError):
            ResidualMomConfig(entry_threshold=0.5, exit_threshold=0.6)

    def test_warmup_periods(self) -> None:
        config = ResidualMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.regression_window

    def test_annualization_factor(self) -> None:
        config = ResidualMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = ResidualMomConfig(
            regression_window=90,
            residual_lookback=30,
            entry_threshold=1.5,
            exit_threshold=0.5,
        )
        assert config.regression_window == 90
        assert config.residual_lookback == 30
        assert config.entry_threshold == 1.5
        assert config.exit_threshold == 0.5

    def test_hedge_only_params(self) -> None:
        config = ResidualMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
