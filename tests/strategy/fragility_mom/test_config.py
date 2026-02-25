"""Tests for Fragility-Aware Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fragility_mom.config import FragilityMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFragilityMomConfig:
    def test_default_values(self) -> None:
        config = FragilityMomConfig()
        assert config.gk_window == 20
        assert config.vov_window == 20
        assert config.vov_percentile_window == 252
        assert config.vov_threshold == 0.35
        assert config.gk_vol_percentile_window == 252
        assert config.mom_lookback == 42
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = FragilityMomConfig()
        with pytest.raises(ValidationError):
            config.gk_window = 999  # type: ignore[misc]

    def test_gk_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FragilityMomConfig(gk_window=2)
        with pytest.raises(ValidationError):
            FragilityMomConfig(gk_window=200)

    def test_vov_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FragilityMomConfig(vov_window=2)
        with pytest.raises(ValidationError):
            FragilityMomConfig(vov_window=200)

    def test_vov_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            FragilityMomConfig(vov_threshold=0.0)
        with pytest.raises(ValidationError):
            FragilityMomConfig(vov_threshold=1.5)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            FragilityMomConfig(mom_lookback=1)
        with pytest.raises(ValidationError):
            FragilityMomConfig(mom_lookback=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FragilityMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FragilityMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.gk_window + config.vov_window
        assert config.warmup_periods() >= config.vov_percentile_window

    def test_annualization_factor(self) -> None:
        config = FragilityMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = FragilityMomConfig(gk_window=30, mom_lookback=40)
        assert config.gk_window == 30
        assert config.mom_lookback == 40

    def test_hedge_params(self) -> None:
        config = FragilityMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
