"""Tests for Composite Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.comp_mom.config import CompMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCompMomConfig:
    def test_default_values(self) -> None:
        config = CompMomConfig()
        assert config.mom_period == 20
        assert config.mom_zscore_window == 60
        assert config.vol_zscore_window == 60
        assert config.gk_window == 20
        assert config.gk_zscore_window == 60
        assert config.composite_threshold == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CompMomConfig()
        with pytest.raises(ValidationError):
            config.mom_period = 999  # type: ignore[misc]

    def test_mom_period_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(mom_period=4)
        with pytest.raises(ValidationError):
            CompMomConfig(mom_period=121)

    def test_mom_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(mom_zscore_window=9)
        with pytest.raises(ValidationError):
            CompMomConfig(mom_zscore_window=201)

    def test_vol_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(vol_zscore_window=9)

    def test_gk_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(gk_window=4)
        with pytest.raises(ValidationError):
            CompMomConfig(gk_window=101)

    def test_gk_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(gk_zscore_window=9)

    def test_composite_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(composite_threshold=-0.1)
        with pytest.raises(ValidationError):
            CompMomConfig(composite_threshold=5.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CompMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.mom_period + config.mom_zscore_window

    def test_annualization_factor(self) -> None:
        config = CompMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = CompMomConfig(mom_period=30, composite_threshold=1.0)
        assert config.mom_period == 30
        assert config.composite_threshold == 1.0

    def test_short_mode_hedge_params(self) -> None:
        config = CompMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5

    def test_hedge_threshold_must_be_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(hedge_threshold=0.01)

    def test_atr_period_range(self) -> None:
        with pytest.raises(ValidationError):
            CompMomConfig(atr_period=4)
        with pytest.raises(ValidationError):
            CompMomConfig(atr_period=51)
