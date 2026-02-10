"""Tests for Acceleration-Skewness Signal config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.accel_skew.config import AccelSkewConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAccelSkewConfig:
    def test_default_values(self) -> None:
        config = AccelSkewConfig()
        assert config.acc_smooth_window == 12
        assert config.skew_window == 30
        assert config.skew_threshold == 0.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = AccelSkewConfig()
        with pytest.raises(ValidationError):
            config.acc_smooth_window = 999  # type: ignore[misc]

    def test_acc_smooth_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelSkewConfig(acc_smooth_window=2)
        with pytest.raises(ValidationError):
            AccelSkewConfig(acc_smooth_window=61)

    def test_skew_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelSkewConfig(skew_window=9)
        with pytest.raises(ValidationError):
            AccelSkewConfig(skew_window=121)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AccelSkewConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AccelSkewConfig()
        assert config.warmup_periods() >= config.skew_window

    def test_annualization_factor(self) -> None:
        config = AccelSkewConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = AccelSkewConfig(skew_window=50)
        assert config.skew_window == 50
