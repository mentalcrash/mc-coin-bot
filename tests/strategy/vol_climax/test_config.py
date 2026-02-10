"""Tests for VolClimaxConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vol_climax.config import ShortMode, VolClimaxConfig


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = VolClimaxConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestVolClimaxConfig:
    def test_default_values(self):
        config = VolClimaxConfig()

        assert config.vol_zscore_window == 30
        assert config.climax_threshold == 2.5
        assert config.obv_lookback == 6
        assert config.divergence_boost == 1.3
        assert config.close_position_threshold == 0.3
        assert config.exit_vol_zscore == 1.0
        assert config.exit_timeout_bars == 18
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 2190.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = VolClimaxConfig()
        with pytest.raises(ValidationError):
            config.vol_zscore_window = 50  # type: ignore[misc]

    def test_vol_zscore_window_range(self):
        config = VolClimaxConfig(vol_zscore_window=10)
        assert config.vol_zscore_window == 10

        config = VolClimaxConfig(vol_zscore_window=100)
        assert config.vol_zscore_window == 100

        with pytest.raises(ValidationError):
            VolClimaxConfig(vol_zscore_window=9)

        with pytest.raises(ValidationError):
            VolClimaxConfig(vol_zscore_window=101)

    def test_climax_threshold_range(self):
        config = VolClimaxConfig(climax_threshold=1.5, exit_vol_zscore=0.5)
        assert config.climax_threshold == 1.5

        config = VolClimaxConfig(climax_threshold=5.0)
        assert config.climax_threshold == 5.0

        with pytest.raises(ValidationError):
            VolClimaxConfig(climax_threshold=1.4)

        with pytest.raises(ValidationError):
            VolClimaxConfig(climax_threshold=5.1)

    def test_obv_lookback_range(self):
        config = VolClimaxConfig(obv_lookback=3)
        assert config.obv_lookback == 3

        config = VolClimaxConfig(obv_lookback=20)
        assert config.obv_lookback == 20

        with pytest.raises(ValidationError):
            VolClimaxConfig(obv_lookback=2)

        with pytest.raises(ValidationError):
            VolClimaxConfig(obv_lookback=21)

    def test_divergence_boost_range(self):
        config = VolClimaxConfig(divergence_boost=1.0)
        assert config.divergence_boost == 1.0

        config = VolClimaxConfig(divergence_boost=2.0)
        assert config.divergence_boost == 2.0

        with pytest.raises(ValidationError):
            VolClimaxConfig(divergence_boost=0.9)

        with pytest.raises(ValidationError):
            VolClimaxConfig(divergence_boost=2.1)

    def test_close_position_threshold_range(self):
        config = VolClimaxConfig(close_position_threshold=0.1)
        assert config.close_position_threshold == 0.1

        config = VolClimaxConfig(close_position_threshold=0.5)
        assert config.close_position_threshold == 0.5

        with pytest.raises(ValidationError):
            VolClimaxConfig(close_position_threshold=0.09)

        with pytest.raises(ValidationError):
            VolClimaxConfig(close_position_threshold=0.51)

    def test_exit_vol_zscore_range(self):
        config = VolClimaxConfig(exit_vol_zscore=0.5)
        assert config.exit_vol_zscore == 0.5

        config = VolClimaxConfig(exit_vol_zscore=2.0)
        assert config.exit_vol_zscore == 2.0

        with pytest.raises(ValidationError):
            VolClimaxConfig(exit_vol_zscore=0.4)

        with pytest.raises(ValidationError):
            VolClimaxConfig(exit_vol_zscore=2.1)

    def test_exit_timeout_bars_range(self):
        config = VolClimaxConfig(exit_timeout_bars=6)
        assert config.exit_timeout_bars == 6

        config = VolClimaxConfig(exit_timeout_bars=48)
        assert config.exit_timeout_bars == 48

        with pytest.raises(ValidationError):
            VolClimaxConfig(exit_timeout_bars=5)

        with pytest.raises(ValidationError):
            VolClimaxConfig(exit_timeout_bars=49)

    def test_mom_lookback_range(self):
        config = VolClimaxConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = VolClimaxConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            VolClimaxConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            VolClimaxConfig(mom_lookback=61)

    def test_vol_target_gte_min_volatility(self):
        config = VolClimaxConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            VolClimaxConfig(vol_target=0.03, min_volatility=0.05)

    def test_exit_vol_zscore_lt_climax_threshold(self):
        """exit_vol_zscore < climax_threshold 검증."""
        config = VolClimaxConfig(climax_threshold=2.5, exit_vol_zscore=1.0)
        assert config.exit_vol_zscore < config.climax_threshold

        with pytest.raises(ValidationError):
            VolClimaxConfig(climax_threshold=2.0, exit_vol_zscore=2.0)

        with pytest.raises(ValidationError):
            VolClimaxConfig(climax_threshold=1.5, exit_vol_zscore=2.0)

    def test_warmup_periods(self):
        config = VolClimaxConfig(
            vol_zscore_window=30,
            obv_lookback=6,
            mom_lookback=20,
            atr_period=14,
        )
        # max(30, 6, 20, 14) + 1 = 31
        assert config.warmup_periods() == 31

    def test_warmup_periods_custom(self):
        config = VolClimaxConfig(
            vol_zscore_window=100,
            obv_lookback=20,
            mom_lookback=60,
            atr_period=50,
        )
        # max(100, 20, 60, 50) + 1 = 101
        assert config.warmup_periods() == 101
