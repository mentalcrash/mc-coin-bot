"""Tests for Vol-of-Vol Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vov_mom.config import ShortMode, VovMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVovMomConfig:
    def test_default_values(self) -> None:
        config = VovMomConfig()
        assert config.gk_window == 20
        assert config.vov_window == 20
        assert config.vov_percentile_window == 120
        assert config.vov_threshold == 0.5
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VovMomConfig()
        with pytest.raises(ValidationError):
            config.gk_window = 999  # type: ignore[misc]

    def test_gk_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VovMomConfig(gk_window=4)
        with pytest.raises(ValidationError):
            VovMomConfig(gk_window=101)

    def test_vov_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VovMomConfig(vov_window=4)
        with pytest.raises(ValidationError):
            VovMomConfig(vov_window=101)

    def test_vov_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            VovMomConfig(vov_threshold=0.0)
        with pytest.raises(ValidationError):
            VovMomConfig(vov_threshold=1.1)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            VovMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            VovMomConfig(mom_lookback=101)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VovMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VovMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.gk_window + config.vov_window

    def test_annualization_factor(self) -> None:
        config = VovMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = VovMomConfig(gk_window=30, vov_window=30)
        assert config.gk_window == 30
        assert config.vov_window == 30

    def test_hedge_params(self) -> None:
        config = VovMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
