"""Tests for Momentum Acceleration config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.mom_accel.config import MomAccelConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMomAccelConfig:
    def test_default_values(self) -> None:
        config = MomAccelConfig()
        assert config.fast_roc == 10
        assert config.slow_roc == 30
        assert config.accel_window == 5
        assert config.momentum_window == 21
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = MomAccelConfig()
        with pytest.raises(ValidationError):
            config.fast_roc = 999  # type: ignore[misc]

    def test_fast_roc_range(self) -> None:
        with pytest.raises(ValidationError):
            MomAccelConfig(fast_roc=1)
        with pytest.raises(ValidationError):
            MomAccelConfig(fast_roc=50)

    def test_fast_gte_slow_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MomAccelConfig(fast_roc=30, slow_roc=30)
        with pytest.raises(ValidationError):
            MomAccelConfig(fast_roc=30, slow_roc=20)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MomAccelConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = MomAccelConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = MomAccelConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = MomAccelConfig(fast_roc=5, slow_roc=20)
        assert config.fast_roc == 5
        assert config.slow_roc == 20
