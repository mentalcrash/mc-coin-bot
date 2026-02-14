"""Tests for OBV Acceleration Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.obv_accel.config import ObvAccelConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestObvAccelConfig:
    def test_default_values(self) -> None:
        config = ObvAccelConfig()
        assert config.obv_smooth == 10
        assert config.accel_window == 10
        assert config.accel_threshold == 0.5
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = ObvAccelConfig()
        with pytest.raises(ValidationError):
            config.obv_smooth = 999  # type: ignore[misc]

    def test_obv_smooth_range(self) -> None:
        with pytest.raises(ValidationError):
            ObvAccelConfig(obv_smooth=2)
        with pytest.raises(ValidationError):
            ObvAccelConfig(obv_smooth=51)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ObvAccelConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = ObvAccelConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = ObvAccelConfig(obv_smooth=15, accel_window=8)
        assert config.obv_smooth == 15
        assert config.accel_window == 8
