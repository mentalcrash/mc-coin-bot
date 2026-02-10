"""Tests for Acceleration-Conviction Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.accel_conv.config import AccelConvConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAccelConvConfig:
    def test_default_values(self) -> None:
        config = AccelConvConfig()
        assert config.smooth_window == 12
        assert config.signal_threshold == 0.01
        assert config.vol_target == 0.35
        assert config.annualization_factor == 1460.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = AccelConvConfig()
        with pytest.raises(ValidationError):
            config.smooth_window = 999  # type: ignore[misc]

    def test_smooth_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelConvConfig(smooth_window=2)
        with pytest.raises(ValidationError):
            AccelConvConfig(smooth_window=61)

    def test_signal_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelConvConfig(signal_threshold=0.0)
        with pytest.raises(ValidationError):
            AccelConvConfig(signal_threshold=0.11)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AccelConvConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AccelConvConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = AccelConvConfig()
        assert config.annualization_factor == 1460.0

    def test_custom_params(self) -> None:
        config = AccelConvConfig(smooth_window=20)
        assert config.smooth_window == 20
