"""Tests for Trend Quality Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.trend_quality_mom.config import ShortMode, TrendQualityMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestTrendQualityMomConfig:
    def test_default_values(self) -> None:
        config = TrendQualityMomConfig()
        assert config.regression_lookback == 30
        assert config.r2_threshold == 0.3
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = TrendQualityMomConfig()
        with pytest.raises(ValidationError):
            config.regression_lookback = 999  # type: ignore[misc]

    def test_regression_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(regression_lookback=5)
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(regression_lookback=200)

    def test_r2_threshold_range(self) -> None:
        # Valid edges
        TrendQualityMomConfig(r2_threshold=0.0)
        TrendQualityMomConfig(r2_threshold=0.9)
        # Invalid
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(r2_threshold=-0.1)
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(r2_threshold=0.95)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(mom_lookback=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            TrendQualityMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = TrendQualityMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.regression_lookback

    def test_annualization_factor(self) -> None:
        config = TrendQualityMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = TrendQualityMomConfig(regression_lookback=50, r2_threshold=0.5)
        assert config.regression_lookback == 50
        assert config.r2_threshold == 0.5

    def test_hedge_params(self) -> None:
        config = TrendQualityMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
