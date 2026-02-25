"""Tests for Z-Momentum (MACD-V) config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.z_mom.config import ShortMode, ZMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestZMomConfig:
    def test_default_values(self) -> None:
        config = ZMomConfig()
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.atr_period == 14
        assert config.flat_zone == 0.3
        assert config.mom_lookback == 5
        assert config.vol_target == 0.50
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = ZMomConfig()
        with pytest.raises(ValidationError):
            config.macd_fast = 999  # type: ignore[misc]

    def test_macd_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(macd_fast=2)
        with pytest.raises(ValidationError):
            ZMomConfig(macd_fast=31)

    def test_macd_slow_range(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(macd_slow=9)
        with pytest.raises(ValidationError):
            ZMomConfig(macd_slow=61)

    def test_macd_signal_range(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(macd_signal=2)
        with pytest.raises(ValidationError):
            ZMomConfig(macd_signal=31)

    def test_atr_period_range(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(atr_period=4)
        with pytest.raises(ValidationError):
            ZMomConfig(atr_period=51)

    def test_flat_zone_range(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(flat_zone=-0.1)
        with pytest.raises(ValidationError):
            ZMomConfig(flat_zone=5.1)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            ZMomConfig(mom_lookback=61)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_macd_slow_gt_macd_fast(self) -> None:
        with pytest.raises(ValidationError):
            ZMomConfig(macd_fast=26, macd_slow=26)
        with pytest.raises(ValidationError):
            ZMomConfig(macd_fast=26, macd_slow=20)

    def test_warmup_periods(self) -> None:
        config = ZMomConfig()
        warmup = config.warmup_periods()
        assert warmup >= config.vol_window
        assert warmup >= config.macd_slow + config.macd_signal
        assert warmup >= config.atr_period
        assert warmup >= config.mom_lookback

    def test_annualization_factor(self) -> None:
        config = ZMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = ZMomConfig(macd_fast=8, macd_slow=21, flat_zone=1.0)
        assert config.macd_fast == 8
        assert config.macd_slow == 21
        assert config.flat_zone == 1.0

    def test_hedge_params(self) -> None:
        config = ZMomConfig(hedge_threshold=-0.10, hedge_strength_ratio=0.5)
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5

    def test_flat_zone_zero_allowed(self) -> None:
        config = ZMomConfig(flat_zone=0.0)
        assert config.flat_zone == 0.0
