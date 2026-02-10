"""Tests for PermEntropyMomConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig, ShortMode


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = PermEntropyMomConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL


class TestPermEntropyMomConfig:
    def test_default_values(self):
        config = PermEntropyMomConfig()

        assert config.pe_order == 3
        assert config.pe_short_window == 30
        assert config.pe_long_window == 60
        assert config.mom_lookback == 30
        assert config.noise_threshold == 0.95
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 2190.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = PermEntropyMomConfig()
        with pytest.raises(ValidationError):
            config.pe_order = 5  # type: ignore[misc]

    def test_pe_order_range(self):
        config = PermEntropyMomConfig(pe_order=2)
        assert config.pe_order == 2

        config = PermEntropyMomConfig(pe_order=7)
        assert config.pe_order == 7

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_order=1)

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_order=8)

    def test_pe_short_window_range(self):
        config = PermEntropyMomConfig(pe_short_window=10, pe_long_window=60)
        assert config.pe_short_window == 10

        config = PermEntropyMomConfig(pe_short_window=120, pe_long_window=240)
        assert config.pe_short_window == 120

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_short_window=9)

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_short_window=121)

    def test_pe_long_window_range(self):
        config = PermEntropyMomConfig(pe_short_window=10, pe_long_window=20)
        assert config.pe_long_window == 20

        config = PermEntropyMomConfig(pe_long_window=240)
        assert config.pe_long_window == 240

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_long_window=19)

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_long_window=241)

    def test_pe_long_gt_short(self):
        """pe_long_window > pe_short_window validation."""
        config = PermEntropyMomConfig(pe_short_window=20, pe_long_window=40)
        assert config.pe_long_window > config.pe_short_window

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_short_window=30, pe_long_window=30)

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(pe_short_window=50, pe_long_window=40)

    def test_mom_lookback_range(self):
        config = PermEntropyMomConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = PermEntropyMomConfig(mom_lookback=120)
        assert config.mom_lookback == 120

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(mom_lookback=121)

    def test_noise_threshold_range(self):
        config = PermEntropyMomConfig(noise_threshold=0.5)
        assert config.noise_threshold == 0.5

        config = PermEntropyMomConfig(noise_threshold=1.0)
        assert config.noise_threshold == 1.0

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(noise_threshold=0.49)

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(noise_threshold=1.01)

    def test_vol_target_gte_min_volatility(self):
        config = PermEntropyMomConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            PermEntropyMomConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = PermEntropyMomConfig(
            pe_order=3,
            pe_short_window=30,
            pe_long_window=60,
            mom_lookback=30,
            atr_period=14,
        )
        # max(60 + 3, 30, 14) + 1 = 63 + 1 = 64
        assert config.warmup_periods() == 64

    def test_warmup_periods_custom(self):
        config = PermEntropyMomConfig(
            pe_order=5,
            pe_short_window=60,
            pe_long_window=120,
            mom_lookback=100,
            atr_period=50,
        )
        # max(120 + 5, 100, 50) + 1 = 125 + 1 = 126
        assert config.warmup_periods() == 126

    def test_warmup_periods_mom_dominant(self):
        config = PermEntropyMomConfig(
            pe_order=2,
            pe_short_window=10,
            pe_long_window=20,
            mom_lookback=120,
            atr_period=14,
        )
        # max(20 + 2, 120, 14) + 1 = 120 + 1 = 121
        assert config.warmup_periods() == 121
