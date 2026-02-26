"""Tests for Donchian Cascade MTF config."""

from __future__ import annotations

import pytest

from src.strategy.donch_cascade.config import DonchCascadeConfig, ShortMode


class TestDefaults:
    def test_default_values(self) -> None:
        config = DonchCascadeConfig()
        assert config.lookback_short == 20
        assert config.lookback_mid == 40
        assert config.lookback_long == 80
        assert config.entry_threshold == 0.34
        assert config.htf_multiplier == 3
        assert config.confirm_ema_period == 5
        assert config.max_wait_bars == 3
        assert config.vol_target == 0.35
        assert config.vol_window == 90
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = DonchCascadeConfig()
        with pytest.raises(Exception):  # noqa: B017
            config.lookback_short = 10  # type: ignore[misc]


class TestActualLookbacks:
    def test_default_multiplier(self) -> None:
        config = DonchCascadeConfig()
        assert config.actual_lookbacks() == (60, 120, 240)

    def test_custom_multiplier(self) -> None:
        config = DonchCascadeConfig(htf_multiplier=6)
        assert config.actual_lookbacks() == (120, 240, 480)

    def test_custom_lookbacks(self) -> None:
        config = DonchCascadeConfig(lookback_short=10, lookback_mid=30, lookback_long=50)
        assert config.actual_lookbacks() == (30, 90, 150)


class TestWarmupPeriods:
    def test_warmup_based_on_max_lookback(self) -> None:
        config = DonchCascadeConfig()
        # max(80*3, 90) + 10 = max(240, 90) + 10 = 250
        assert config.warmup_periods() == 250

    def test_warmup_based_on_vol_window(self) -> None:
        config = DonchCascadeConfig(
            lookback_short=5, lookback_mid=10, lookback_long=20, vol_window=300
        )
        # max(20*3, 300) + 10 = max(60, 300) + 10 = 310
        assert config.warmup_periods() == 310


class TestCrossFieldValidation:
    def test_lookback_order_valid(self) -> None:
        config = DonchCascadeConfig(lookback_short=10, lookback_mid=30, lookback_long=50)
        assert config.lookback_short < config.lookback_mid < config.lookback_long

    def test_lookback_order_invalid(self) -> None:
        with pytest.raises(ValueError, match="lookback_short"):
            DonchCascadeConfig(lookback_short=50, lookback_mid=30, lookback_long=80)

    def test_lookback_equal_invalid(self) -> None:
        with pytest.raises(ValueError, match="lookback_short"):
            DonchCascadeConfig(lookback_short=40, lookback_mid=40, lookback_long=80)

    def test_vol_target_below_min_volatility(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            DonchCascadeConfig(vol_target=0.01, min_volatility=0.05)


class TestFieldBounds:
    def test_htf_multiplier_bounds(self) -> None:
        DonchCascadeConfig(htf_multiplier=2)  # min
        DonchCascadeConfig(htf_multiplier=6)  # max
        with pytest.raises(ValueError):
            DonchCascadeConfig(htf_multiplier=1)
        with pytest.raises(ValueError):
            DonchCascadeConfig(htf_multiplier=7)

    def test_confirm_ema_bounds(self) -> None:
        DonchCascadeConfig(confirm_ema_period=2)  # min
        DonchCascadeConfig(confirm_ema_period=30)  # max
        with pytest.raises(ValueError):
            DonchCascadeConfig(confirm_ema_period=1)

    def test_max_wait_bars_bounds(self) -> None:
        DonchCascadeConfig(max_wait_bars=1)  # min
        DonchCascadeConfig(max_wait_bars=10)  # max
        with pytest.raises(ValueError):
            DonchCascadeConfig(max_wait_bars=0)


class TestShortMode:
    def test_all_modes(self) -> None:
        for mode in ShortMode:
            config = DonchCascadeConfig(short_mode=mode)
            assert config.short_mode == mode

    def test_int_enum_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2
