"""Tests for OUMeanRevConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.ou_meanrev.config import OUMeanRevConfig, ShortMode


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = OUMeanRevConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestOUMeanRevConfig:
    def test_default_values(self):
        config = OUMeanRevConfig()

        assert config.ou_window == 120
        assert config.entry_zscore == 2.0
        assert config.exit_zscore == 0.5
        assert config.max_half_life == 30
        assert config.exit_timeout_bars == 30
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 2190.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = OUMeanRevConfig()
        with pytest.raises(ValidationError):
            config.ou_window = 200  # type: ignore[misc]

    def test_ou_window_range(self):
        config = OUMeanRevConfig(ou_window=40)
        assert config.ou_window == 40

        config = OUMeanRevConfig(ou_window=500)
        assert config.ou_window == 500

        with pytest.raises(ValidationError):
            OUMeanRevConfig(ou_window=39)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(ou_window=501)

    def test_entry_zscore_range(self):
        config = OUMeanRevConfig(entry_zscore=1.0)
        assert config.entry_zscore == 1.0

        config = OUMeanRevConfig(entry_zscore=4.0)
        assert config.entry_zscore == 4.0

        with pytest.raises(ValidationError):
            OUMeanRevConfig(entry_zscore=0.5)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(entry_zscore=4.5)

    def test_exit_zscore_range(self):
        config = OUMeanRevConfig(exit_zscore=0.1)
        assert config.exit_zscore == 0.1

        config = OUMeanRevConfig(exit_zscore=1.5, entry_zscore=2.0)
        assert config.exit_zscore == 1.5

        with pytest.raises(ValidationError):
            OUMeanRevConfig(exit_zscore=0.05)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(exit_zscore=1.6)

    def test_exit_zscore_lt_entry_zscore(self):
        """exit_zscore < entry_zscore 검증."""
        config = OUMeanRevConfig(exit_zscore=0.5, entry_zscore=2.0)
        assert config.exit_zscore < config.entry_zscore

        with pytest.raises(ValidationError):
            OUMeanRevConfig(exit_zscore=2.0, entry_zscore=2.0)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(exit_zscore=2.5, entry_zscore=2.0)

    def test_max_half_life_range(self):
        config = OUMeanRevConfig(max_half_life=10)
        assert config.max_half_life == 10

        config = OUMeanRevConfig(max_half_life=100)
        assert config.max_half_life == 100

        with pytest.raises(ValidationError):
            OUMeanRevConfig(max_half_life=9)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(max_half_life=101)

    def test_exit_timeout_bars_range(self):
        config = OUMeanRevConfig(exit_timeout_bars=10)
        assert config.exit_timeout_bars == 10

        config = OUMeanRevConfig(exit_timeout_bars=60)
        assert config.exit_timeout_bars == 60

        with pytest.raises(ValidationError):
            OUMeanRevConfig(exit_timeout_bars=9)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(exit_timeout_bars=61)

    def test_mom_lookback_range(self):
        config = OUMeanRevConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = OUMeanRevConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            OUMeanRevConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            OUMeanRevConfig(mom_lookback=61)

    def test_vol_target_gte_min_volatility(self):
        config = OUMeanRevConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            OUMeanRevConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = OUMeanRevConfig(
            ou_window=120,
            mom_lookback=20,
            atr_period=14,
        )
        # max(120, 20, 14) + 1 = 121
        assert config.warmup_periods() == 121

    def test_warmup_periods_custom(self):
        config = OUMeanRevConfig(
            ou_window=200,
            mom_lookback=60,
            atr_period=50,
        )
        # max(200, 60, 50) + 1 = 201
        assert config.warmup_periods() == 201

    def test_warmup_mom_lookback_dominates(self):
        """mom_lookback이 ou_window보다 클 때."""
        config = OUMeanRevConfig(
            ou_window=40,
            mom_lookback=60,
            atr_period=14,
        )
        # max(40, 60, 14) + 1 = 61
        assert config.warmup_periods() == 61
