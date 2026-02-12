"""Tests for StochMomConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.stoch_mom.config import ShortMode, StochMomConfig


class TestStochMomConfig:
    """StochMomConfig basic tests."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        config = StochMomConfig()

        assert config.k_period == 14
        assert config.d_period == 3
        assert config.sma_period == 30
        assert config.atr_period == 14
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.min_vol_ratio == 0.30
        assert config.max_vol_ratio == 0.95
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self) -> None:
        """Frozen model does not allow attribute modification."""
        config = StochMomConfig()

        with pytest.raises(ValidationError):
            config.k_period = 20  # type: ignore[misc]

    def test_vol_target_validation(self) -> None:
        """vol_target < min_volatility raises ValidationError."""
        with pytest.raises(ValidationError, match="vol_target"):
            StochMomConfig(vol_target=0.03, min_volatility=0.10)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility is allowed."""
        config = StochMomConfig(vol_target=0.10, min_volatility=0.10)
        assert config.vol_target == config.min_volatility

    def test_min_vol_ratio_lt_max_vol_ratio(self) -> None:
        """min_vol_ratio < max_vol_ratio validation."""
        config = StochMomConfig(min_vol_ratio=0.20, max_vol_ratio=0.80)
        assert config.min_vol_ratio < config.max_vol_ratio

    def test_min_vol_ratio_ge_max_vol_ratio_raises(self) -> None:
        """min_vol_ratio >= max_vol_ratio raises ValidationError."""
        with pytest.raises(ValidationError, match="min_vol_ratio"):
            StochMomConfig(min_vol_ratio=0.80, max_vol_ratio=0.80)

    def test_min_vol_ratio_gt_max_vol_ratio_raises(self) -> None:
        """min_vol_ratio > max_vol_ratio raises ValidationError."""
        with pytest.raises(ValidationError, match="min_vol_ratio"):
            StochMomConfig(min_vol_ratio=0.80, max_vol_ratio=0.60)


class TestWarmupPeriods:
    """Warmup period calculation tests."""

    def test_warmup_default(self) -> None:
        """Default warmup = max(14, 30, 14) + 3 + 1 = 34."""
        config = StochMomConfig()
        assert config.warmup_periods() == 30 + 3 + 1

    def test_warmup_custom(self) -> None:
        """Custom parameters: warmup = max(20, 50, 20) + 5 + 1 = 56."""
        config = StochMomConfig(k_period=20, sma_period=50, atr_period=20, d_period=5)
        assert config.warmup_periods() == 50 + 5 + 1

    def test_warmup_k_dominant(self) -> None:
        """When k_period is largest: warmup = k_period + d_period + 1."""
        config = StochMomConfig(k_period=50, sma_period=10, atr_period=10, d_period=3)
        assert config.warmup_periods() == 50 + 3 + 1


class TestForTimeframe:
    """Timeframe factory tests."""

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d') uses annualization=365.0."""
        config = StochMomConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h') uses annualization=2190.0."""
        config = StochMomConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_1h(self) -> None:
        """for_timeframe('1h') uses annualization=8760.0."""
        config = StochMomConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_unknown(self) -> None:
        """Unknown timeframe uses default 365.0."""
        config = StochMomConfig.for_timeframe("7h")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe with kwargs overrides."""
        config = StochMomConfig.for_timeframe("4h", vol_target=0.30)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.30


class TestPresets:
    """Preset factory tests."""

    def test_conservative_preset(self) -> None:
        """conservative() preset values."""
        config = StochMomConfig.conservative()

        assert config.k_period == 21
        assert config.sma_period == 50
        assert config.vol_target == 0.30
        assert config.min_volatility == 0.08

    def test_aggressive_preset(self) -> None:
        """aggressive() preset values."""
        config = StochMomConfig.aggressive()

        assert config.k_period == 9
        assert config.sma_period == 20
        assert config.vol_target == 0.50
        assert config.min_volatility == 0.05


class TestShortModeConfig:
    """ShortMode configuration tests."""

    def test_default_is_disabled(self) -> None:
        """Default short_mode is DISABLED."""
        config = StochMomConfig()
        assert config.short_mode == ShortMode.DISABLED

    def test_full_mode(self) -> None:
        """FULL mode can be set."""
        config = StochMomConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

    def test_short_mode_values(self) -> None:
        """ShortMode integer values."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.FULL == 2


class TestFieldRanges:
    """Field range validation tests."""

    def test_k_period_min(self) -> None:
        """k_period minimum 5 is allowed."""
        config = StochMomConfig(k_period=5)
        assert config.k_period == 5

    def test_k_period_max(self) -> None:
        """k_period maximum 50 is allowed."""
        config = StochMomConfig(k_period=50)
        assert config.k_period == 50

    def test_k_period_below_min(self) -> None:
        """k_period < 5 raises ValidationError."""
        with pytest.raises(ValidationError):
            StochMomConfig(k_period=4)

    def test_k_period_above_max(self) -> None:
        """k_period > 50 raises ValidationError."""
        with pytest.raises(ValidationError):
            StochMomConfig(k_period=51)

    def test_d_period_min(self) -> None:
        """d_period minimum 2 is allowed."""
        config = StochMomConfig(d_period=2)
        assert config.d_period == 2

    def test_d_period_above_max(self) -> None:
        """d_period > 10 raises ValidationError."""
        with pytest.raises(ValidationError):
            StochMomConfig(d_period=11)

    def test_sma_period_min(self) -> None:
        """sma_period minimum 10 is allowed."""
        config = StochMomConfig(sma_period=10)
        assert config.sma_period == 10

    def test_sma_period_above_max(self) -> None:
        """sma_period > 100 raises ValidationError."""
        with pytest.raises(ValidationError):
            StochMomConfig(sma_period=101)
