"""Tests for CandleRejectConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.candle_reject.config import CandleRejectConfig, ShortMode


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = CandleRejectConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestCandleRejectConfig:
    def test_default_values(self):
        config = CandleRejectConfig()

        assert config.rejection_threshold == 0.6
        assert config.volume_zscore_threshold == 1.0
        assert config.volume_zscore_window == 30
        assert config.consecutive_boost == 1.5
        assert config.consecutive_min == 2
        assert config.exit_timeout_bars == 12
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 2190.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = CandleRejectConfig()
        with pytest.raises(ValidationError):
            config.rejection_threshold = 0.7  # type: ignore[misc]

    def test_rejection_threshold_range(self):
        config = CandleRejectConfig(rejection_threshold=0.3)
        assert config.rejection_threshold == 0.3

        config = CandleRejectConfig(rejection_threshold=0.9)
        assert config.rejection_threshold == 0.9

        with pytest.raises(ValidationError):
            CandleRejectConfig(rejection_threshold=0.2)

        with pytest.raises(ValidationError):
            CandleRejectConfig(rejection_threshold=0.95)

    def test_volume_zscore_threshold_range(self):
        config = CandleRejectConfig(volume_zscore_threshold=0.5)
        assert config.volume_zscore_threshold == 0.5

        config = CandleRejectConfig(volume_zscore_threshold=3.0)
        assert config.volume_zscore_threshold == 3.0

        with pytest.raises(ValidationError):
            CandleRejectConfig(volume_zscore_threshold=0.3)

        with pytest.raises(ValidationError):
            CandleRejectConfig(volume_zscore_threshold=3.5)

    def test_volume_zscore_window_range(self):
        config = CandleRejectConfig(volume_zscore_window=10)
        assert config.volume_zscore_window == 10

        config = CandleRejectConfig(volume_zscore_window=100)
        assert config.volume_zscore_window == 100

        with pytest.raises(ValidationError):
            CandleRejectConfig(volume_zscore_window=9)

        with pytest.raises(ValidationError):
            CandleRejectConfig(volume_zscore_window=101)

    def test_consecutive_boost_range(self):
        config = CandleRejectConfig(consecutive_boost=1.0)
        assert config.consecutive_boost == 1.0

        config = CandleRejectConfig(consecutive_boost=2.0)
        assert config.consecutive_boost == 2.0

        with pytest.raises(ValidationError):
            CandleRejectConfig(consecutive_boost=0.5)

        with pytest.raises(ValidationError):
            CandleRejectConfig(consecutive_boost=2.5)

    def test_consecutive_min_range(self):
        config = CandleRejectConfig(consecutive_min=2)
        assert config.consecutive_min == 2

        config = CandleRejectConfig(consecutive_min=5)
        assert config.consecutive_min == 5

        with pytest.raises(ValidationError):
            CandleRejectConfig(consecutive_min=1)

        with pytest.raises(ValidationError):
            CandleRejectConfig(consecutive_min=6)

    def test_exit_timeout_bars_range(self):
        config = CandleRejectConfig(exit_timeout_bars=4)
        assert config.exit_timeout_bars == 4

        config = CandleRejectConfig(exit_timeout_bars=48)
        assert config.exit_timeout_bars == 48

        with pytest.raises(ValidationError):
            CandleRejectConfig(exit_timeout_bars=3)

        with pytest.raises(ValidationError):
            CandleRejectConfig(exit_timeout_bars=49)

    def test_vol_target_gte_min_volatility(self):
        config = CandleRejectConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            CandleRejectConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = CandleRejectConfig(
            volume_zscore_window=30,
            atr_period=14,
        )
        # max(30, 14) + 1 = 31
        assert config.warmup_periods() == 31

    def test_warmup_periods_custom(self):
        config = CandleRejectConfig(
            volume_zscore_window=50,
            atr_period=40,
        )
        # max(50, 40) + 1 = 51
        assert config.warmup_periods() == 51

    def test_warmup_periods_atr_dominant(self):
        config = CandleRejectConfig(
            volume_zscore_window=10,
            atr_period=50,
        )
        # max(10, 50) + 1 = 51
        assert config.warmup_periods() == 51
