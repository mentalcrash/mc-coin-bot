"""Tests for Regime-Adaptive Multi-Lookback Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.regime_adaptive_mom.config import RegimeAdaptiveMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestRegimeAdaptiveMomConfig:
    def test_default_values(self) -> None:
        config = RegimeAdaptiveMomConfig()
        assert config.fast_lookback == 20
        assert config.mid_lookback == 60
        assert config.slow_lookback == 120
        assert config.trending_fast_weight == 0.6
        assert config.trending_mid_weight == 0.3
        assert config.trending_slow_weight == 0.1
        assert config.volatile_fast_weight == 0.1
        assert config.volatile_mid_weight == 0.3
        assert config.volatile_slow_weight == 0.6
        assert config.signal_threshold == 0.01
        assert config.trending_vol_target == 0.40
        assert config.ranging_vol_target == 0.20
        assert config.volatile_vol_target == 0.10
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = RegimeAdaptiveMomConfig()
        with pytest.raises(ValidationError):
            config.fast_lookback = 999  # type: ignore[misc]

    def test_fast_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(fast_lookback=4)
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(fast_lookback=61)

    def test_mid_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(mid_lookback=19)
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(mid_lookback=121)

    def test_slow_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(slow_lookback=59)
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(slow_lookback=253)

    def test_signal_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(signal_threshold=-0.01)
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(signal_threshold=0.21)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_lookback_ordering(self) -> None:
        # fast must be < mid
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(fast_lookback=60, mid_lookback=60, slow_lookback=120)
        # mid must be < slow
        with pytest.raises(ValidationError):
            RegimeAdaptiveMomConfig(fast_lookback=20, mid_lookback=120, slow_lookback=120)

    def test_warmup_periods(self) -> None:
        config = RegimeAdaptiveMomConfig()
        assert config.warmup_periods() >= config.slow_lookback
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = RegimeAdaptiveMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = RegimeAdaptiveMomConfig(
            fast_lookback=10, mid_lookback=40, slow_lookback=80, signal_threshold=0.02
        )
        assert config.fast_lookback == 10
        assert config.mid_lookback == 40
        assert config.slow_lookback == 80
        assert config.signal_threshold == 0.02

    def test_regime_weights(self) -> None:
        config = RegimeAdaptiveMomConfig(
            trending_fast_weight=0.8,
            trending_mid_weight=0.15,
            trending_slow_weight=0.05,
        )
        assert config.trending_fast_weight == 0.8
        assert config.trending_mid_weight == 0.15
        assert config.trending_slow_weight == 0.05
