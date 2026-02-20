"""Tests for VRP-Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vrp_trend.config import ShortMode, VrpTrendConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVrpTrendConfig:
    def test_default_values(self) -> None:
        config = VrpTrendConfig()
        assert config.rv_window == 30
        assert config.vrp_ma_window == 14
        assert config.vrp_zscore_window == 90
        assert config.vrp_entry_z == 0.5
        assert config.vrp_exit_z == -0.5
        assert config.trend_sma_window == 50
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = VrpTrendConfig()
        with pytest.raises(ValidationError):
            config.rv_window = 999  # type: ignore[misc]

    def test_rv_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(rv_window=4)
        with pytest.raises(ValidationError):
            VrpTrendConfig(rv_window=121)

    def test_vrp_ma_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_ma_window=2)
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_ma_window=61)

    def test_vrp_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_zscore_window=19)
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_zscore_window=366)

    def test_vrp_entry_z_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_entry_z=-0.1)
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_entry_z=3.1)

    def test_vrp_exit_z_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_exit_z=-3.1)
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_exit_z=0.1)

    def test_trend_sma_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(trend_sma_window=9)
        with pytest.raises(ValidationError):
            VrpTrendConfig(trend_sma_window=201)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VrpTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_vrp_entry_z_must_exceed_exit_z(self) -> None:
        """vrp_entry_z must be strictly greater than vrp_exit_z."""
        # Same value (0.0 == 0.0) → fail
        with pytest.raises(ValidationError):
            VrpTrendConfig(vrp_entry_z=0.0, vrp_exit_z=0.0)

    def test_vrp_entry_exit_z_cross_validation(self) -> None:
        """vrp_entry_z == vrp_exit_z should fail."""
        # entry_z=0.5, exit_z=0.0 → 0.5 > 0.0 → OK
        config = VrpTrendConfig(vrp_entry_z=0.5, vrp_exit_z=0.0)
        assert config.vrp_entry_z > config.vrp_exit_z

    def test_warmup_periods(self) -> None:
        config = VrpTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert (
            config.warmup_periods()
            >= config.rv_window + config.vrp_ma_window + config.vrp_zscore_window
        )

    def test_annualization_factor(self) -> None:
        config = VrpTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = VrpTrendConfig(rv_window=20, vrp_ma_window=7)
        assert config.rv_window == 20
        assert config.vrp_ma_window == 7

    def test_hedge_params(self) -> None:
        config = VrpTrendConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
