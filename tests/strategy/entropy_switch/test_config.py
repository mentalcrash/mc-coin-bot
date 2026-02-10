"""Tests for EntropySwitchConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.entropy_switch.config import EntropySwitchConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self) -> None:
        config = EntropySwitchConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestEntropySwitchConfig:
    def test_default_values(self) -> None:
        config = EntropySwitchConfig()

        assert config.entropy_window == 120
        assert config.entropy_bins == 10
        assert config.entropy_low_threshold == 1.8
        assert config.entropy_high_threshold == 2.2
        assert config.mom_lookback == 20
        assert config.adx_period == 14
        assert config.adx_threshold == 20.0
        assert config.use_adx_filter is True
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self) -> None:
        config = EntropySwitchConfig()
        with pytest.raises(ValidationError):
            config.entropy_window = 200  # type: ignore[misc]

    def test_entropy_window_range(self) -> None:
        config = EntropySwitchConfig(entropy_window=30)
        assert config.entropy_window == 30

        config = EntropySwitchConfig(entropy_window=500)
        assert config.entropy_window == 500

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_window=29)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_window=501)

    def test_entropy_bins_range(self) -> None:
        config = EntropySwitchConfig(entropy_bins=5)
        assert config.entropy_bins == 5

        config = EntropySwitchConfig(entropy_bins=30)
        assert config.entropy_bins == 30

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_bins=4)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_bins=31)

    def test_entropy_low_threshold_range(self) -> None:
        config = EntropySwitchConfig(entropy_low_threshold=0.5, entropy_high_threshold=1.0)
        assert config.entropy_low_threshold == 0.5

        config = EntropySwitchConfig(entropy_low_threshold=3.0, entropy_high_threshold=3.5)
        assert config.entropy_low_threshold == 3.0

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_low_threshold=0.4)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_low_threshold=3.1)

    def test_entropy_high_threshold_range(self) -> None:
        config = EntropySwitchConfig(entropy_high_threshold=1.0, entropy_low_threshold=0.5)
        assert config.entropy_high_threshold == 1.0

        config = EntropySwitchConfig(entropy_high_threshold=3.5)
        assert config.entropy_high_threshold == 3.5

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_high_threshold=0.9)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_high_threshold=3.6)

    def test_entropy_threshold_ordering(self) -> None:
        """entropy_low_threshold < entropy_high_threshold 검증."""
        config = EntropySwitchConfig(entropy_low_threshold=1.5, entropy_high_threshold=2.5)
        assert config.entropy_low_threshold < config.entropy_high_threshold

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_low_threshold=2.5, entropy_high_threshold=2.5)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(entropy_low_threshold=2.5, entropy_high_threshold=2.0)

    def test_mom_lookback_range(self) -> None:
        config = EntropySwitchConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = EntropySwitchConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            EntropySwitchConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(mom_lookback=61)

    def test_adx_period_range(self) -> None:
        config = EntropySwitchConfig(adx_period=5)
        assert config.adx_period == 5

        config = EntropySwitchConfig(adx_period=50)
        assert config.adx_period == 50

        with pytest.raises(ValidationError):
            EntropySwitchConfig(adx_period=4)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(adx_period=51)

    def test_adx_threshold_range(self) -> None:
        config = EntropySwitchConfig(adx_threshold=10.0)
        assert config.adx_threshold == 10.0

        config = EntropySwitchConfig(adx_threshold=40.0)
        assert config.adx_threshold == 40.0

        with pytest.raises(ValidationError):
            EntropySwitchConfig(adx_threshold=9.9)

        with pytest.raises(ValidationError):
            EntropySwitchConfig(adx_threshold=40.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        config = EntropySwitchConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            EntropySwitchConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = EntropySwitchConfig(
            entropy_window=120,
            mom_lookback=20,
            adx_period=14,
            atr_period=14,
        )
        # max(120, 20, 14, 14) + 1 = 121
        assert config.warmup_periods() == 121

    def test_warmup_periods_custom(self) -> None:
        config = EntropySwitchConfig(
            entropy_window=200,
            mom_lookback=60,
            adx_period=50,
            atr_period=50,
        )
        # max(200, 60, 50, 50) + 1 = 201
        assert config.warmup_periods() == 201

    def test_warmup_periods_mom_dominant(self) -> None:
        config = EntropySwitchConfig(
            entropy_window=30,
            mom_lookback=60,
            adx_period=5,
            atr_period=5,
        )
        # max(30, 60, 5, 5) + 1 = 61
        assert config.warmup_periods() == 61
