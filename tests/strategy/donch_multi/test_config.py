"""Tests for Donchian Multi-Scale config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.donch_multi.config import DonchMultiConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestDonchMultiConfig:
    def test_default_values(self) -> None:
        config = DonchMultiConfig()
        assert config.lookback_short == 20
        assert config.lookback_mid == 40
        assert config.lookback_long == 80
        assert config.entry_threshold == 0.34
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = DonchMultiConfig()
        with pytest.raises(ValidationError):
            config.lookback_short = 999  # type: ignore[misc]

    def test_lookback_short_range(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_short=2)
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_short=101, lookback_mid=200, lookback_long=400)

    def test_lookback_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_mid=5)

    def test_lookback_long_range(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_long=10)

    def test_lookback_ordering(self) -> None:
        """lookback_short < lookback_mid < lookback_long 검증."""
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_short=40, lookback_mid=40, lookback_long=80)
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_short=20, lookback_mid=80, lookback_long=80)
        with pytest.raises(ValidationError):
            DonchMultiConfig(lookback_short=80, lookback_mid=40, lookback_long=20)

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(entry_threshold=-0.1)
        with pytest.raises(ValidationError):
            DonchMultiConfig(entry_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = DonchMultiConfig()
        assert config.warmup_periods() >= config.lookback_long
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = DonchMultiConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = DonchMultiConfig(
            lookback_short=10,
            lookback_mid=30,
            lookback_long=60,
            entry_threshold=0.5,
        )
        assert config.lookback_short == 10
        assert config.lookback_mid == 30
        assert config.lookback_long == 60
        assert config.entry_threshold == 0.5

    def test_hedge_params(self) -> None:
        config = DonchMultiConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_hedge_threshold_le_zero(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(hedge_threshold=0.1)

    def test_hedge_strength_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            DonchMultiConfig(hedge_strength_ratio=0.0)
        with pytest.raises(ValidationError):
            DonchMultiConfig(hedge_strength_ratio=1.1)
