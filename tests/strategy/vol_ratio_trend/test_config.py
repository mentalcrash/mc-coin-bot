"""Tests for Volatility Ratio Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_ratio_trend.config import ShortMode, VolRatioTrendConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolRatioTrendConfig:
    def test_default_values(self) -> None:
        config = VolRatioTrendConfig()
        assert config.short_vol_window == 10
        assert config.long_vol_window == 60
        assert config.ratio_smooth_window == 5
        assert config.contango_threshold == 0.90
        assert config.backwardation_threshold == 1.20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = VolRatioTrendConfig()
        with pytest.raises(ValidationError):
            config.short_vol_window = 999  # type: ignore[misc]

    def test_short_vol_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VolRatioTrendConfig(short_vol_window=1)
        with pytest.raises(ValidationError):
            VolRatioTrendConfig(short_vol_window=50)

    def test_short_gte_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VolRatioTrendConfig(short_vol_window=30, long_vol_window=30)

    def test_contango_gte_backwardation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VolRatioTrendConfig(contango_threshold=1.0, backwardation_threshold=1.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolRatioTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolRatioTrendConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VolRatioTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = VolRatioTrendConfig(short_vol_window=7, long_vol_window=90)
        assert config.short_vol_window == 7
        assert config.long_vol_window == 90
