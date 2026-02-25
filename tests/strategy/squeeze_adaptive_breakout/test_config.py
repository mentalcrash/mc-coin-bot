"""Tests for Squeeze-Adaptive Breakout config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.squeeze_adaptive_breakout.config import (
    ShortMode,
    SqueezeAdaptiveBreakoutConfig,
)


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestSqueezeAdaptiveBreakoutConfig:
    def test_default_values(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig()
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.kc_period == 20
        assert config.kc_atr_period == 10
        assert config.kc_mult == 1.5
        assert config.kama_er_lookback == 10
        assert config.kama_fast == 2
        assert config.kama_slow == 30
        assert config.bb_pos_period == 20
        assert config.bb_pos_std == 2.0
        assert config.bb_pos_long_threshold == 0.7
        assert config.bb_pos_short_threshold == 0.3
        assert config.squeeze_lookback == 3
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig()
        with pytest.raises(ValidationError):
            config.bb_period = 999  # type: ignore[misc]

    def test_bb_period_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_period=4)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_period=51)

    def test_bb_std_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_std=0.5)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_std=3.5)

    def test_kc_mult_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(kc_mult=0.3)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(kc_mult=3.5)

    def test_kama_er_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(kama_er_lookback=2)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(kama_er_lookback=31)

    def test_bb_pos_long_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_pos_long_threshold=0.4)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_pos_long_threshold=1.1)

    def test_bb_pos_short_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_pos_short_threshold=-0.1)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(bb_pos_short_threshold=0.6)

    def test_squeeze_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(squeeze_lookback=0)
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(squeeze_lookback=21)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            SqueezeAdaptiveBreakoutConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.bb_period
        assert config.warmup_periods() >= config.kc_period

    def test_annualization_factor(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig(
            bb_period=15,
            kc_mult=2.0,
            kama_er_lookback=15,
            squeeze_lookback=5,
        )
        assert config.bb_period == 15
        assert config.kc_mult == 2.0
        assert config.kama_er_lookback == 15
        assert config.squeeze_lookback == 5

    def test_short_mode_options(self) -> None:
        for mode in ShortMode:
            config = SqueezeAdaptiveBreakoutConfig(short_mode=mode)
            assert config.short_mode == mode

    def test_hedge_params(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig()
        assert config.hedge_threshold <= 0.0
        assert 0.0 < config.hedge_strength_ratio <= 1.0
