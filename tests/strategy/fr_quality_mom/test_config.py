"""Tests for FR Quality Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fr_quality_mom.config import FrQualityMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFrQualityMomConfig:
    def test_default_values(self) -> None:
        config = FrQualityMomConfig()
        assert config.momentum_window == 21
        assert config.fr_lookback == 7
        assert config.fr_zscore_window == 90
        assert config.fr_crowd_threshold == 1.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = FrQualityMomConfig()
        with pytest.raises(ValidationError):
            config.momentum_window = 999  # type: ignore[misc]

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            FrQualityMomConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            FrQualityMomConfig(fr_lookback=50)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FrQualityMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FrQualityMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = FrQualityMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = FrQualityMomConfig(fr_lookback=14, fr_crowd_threshold=2.0)
        assert config.fr_lookback == 14
        assert config.fr_crowd_threshold == 2.0
