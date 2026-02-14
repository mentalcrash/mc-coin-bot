"""Tests for Multi-Scale Volatility Ratio config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ms_vol_ratio.config import MSVolRatioConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMSVolRatioConfig:
    def test_default_values(self) -> None:
        config = MSVolRatioConfig()
        assert config.short_vol_window == 6
        assert config.long_vol_window == 48
        assert config.ratio_upper == 1.3
        assert config.ratio_lower == 0.7
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = MSVolRatioConfig()
        with pytest.raises(ValidationError):
            config.short_vol_window = 999  # type: ignore[misc]

    def test_short_vol_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MSVolRatioConfig(short_vol_window=1)
        with pytest.raises(ValidationError):
            MSVolRatioConfig(short_vol_window=50)

    def test_short_must_lt_long(self) -> None:
        with pytest.raises(ValidationError):
            MSVolRatioConfig(short_vol_window=48, long_vol_window=48)
        with pytest.raises(ValidationError):
            MSVolRatioConfig(short_vol_window=30, long_vol_window=20)

    def test_ratio_upper_gt_lower(self) -> None:
        with pytest.raises(ValidationError):
            MSVolRatioConfig(ratio_upper=0.5, ratio_lower=0.7)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MSVolRatioConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = MSVolRatioConfig()
        assert config.warmup_periods() >= config.long_vol_window

    def test_annualization_factor(self) -> None:
        config = MSVolRatioConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = MSVolRatioConfig(short_vol_window=10, long_vol_window=60)
        assert config.short_vol_window == 10
        assert config.long_vol_window == 60
