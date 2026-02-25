"""Tests for T-Stat Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.t_stat_mom.config import ShortMode, TStatMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestTStatMomConfig:
    def test_default_values(self) -> None:
        config = TStatMomConfig()
        assert config.fast_lookback == 20
        assert config.mid_lookback == 40
        assert config.slow_lookback == 80
        assert config.entry_threshold == 1.0
        assert config.exit_threshold == 0.5
        assert config.tanh_scale == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = TStatMomConfig()
        with pytest.raises(ValidationError):
            config.fast_lookback = 999  # type: ignore[misc]

    def test_fast_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(fast_lookback=2)  # too low
        with pytest.raises(ValidationError):
            TStatMomConfig(fast_lookback=100)  # too high

    def test_mid_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(mid_lookback=10)  # too low
        with pytest.raises(ValidationError):
            TStatMomConfig(mid_lookback=200)  # too high

    def test_slow_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(slow_lookback=20)  # too low
        with pytest.raises(ValidationError):
            TStatMomConfig(slow_lookback=300)  # too high

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(entry_threshold=0.1)  # too low
        with pytest.raises(ValidationError):
            TStatMomConfig(entry_threshold=5.0)  # too high

    def test_tanh_scale_range(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(tanh_scale=0.0)  # must be > 0
        with pytest.raises(ValidationError):
            TStatMomConfig(tanh_scale=3.0)  # too high

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_entry_gte_exit_threshold(self) -> None:
        with pytest.raises(ValidationError):
            TStatMomConfig(entry_threshold=0.3, exit_threshold=0.5)

    def test_lookback_ordering(self) -> None:
        """fast < mid < slow must hold."""
        with pytest.raises(ValidationError):
            TStatMomConfig(fast_lookback=50, mid_lookback=40, slow_lookback=80)
        with pytest.raises(ValidationError):
            TStatMomConfig(fast_lookback=20, mid_lookback=90, slow_lookback=80)

    def test_warmup_periods(self) -> None:
        config = TStatMomConfig()
        assert config.warmup_periods() >= config.slow_lookback
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = TStatMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = TStatMomConfig(fast_lookback=10, mid_lookback=30, slow_lookback=60)
        assert config.fast_lookback == 10
        assert config.mid_lookback == 30
        assert config.slow_lookback == 60

    def test_hedge_only_params(self) -> None:
        config = TStatMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
