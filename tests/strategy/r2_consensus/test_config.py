"""Tests for R2 Consensus Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.r2_consensus.config import R2ConsensusConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestR2ConsensusConfig:
    def test_default_values(self) -> None:
        config = R2ConsensusConfig()
        assert config.lookback_short == 20
        assert config.lookback_mid == 50
        assert config.lookback_long == 120
        assert config.r2_threshold == 0.3
        assert config.entry_threshold == 0.34
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = R2ConsensusConfig()
        with pytest.raises(ValidationError):
            config.lookback_short = 999  # type: ignore[misc]

    def test_lookback_short_range(self) -> None:
        with pytest.raises(ValidationError):
            R2ConsensusConfig(lookback_short=2)
        with pytest.raises(ValidationError):
            R2ConsensusConfig(lookback_short=100)

    def test_lookback_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            R2ConsensusConfig(lookback_mid=10)
        with pytest.raises(ValidationError):
            R2ConsensusConfig(lookback_mid=200)

    def test_lookback_long_range(self) -> None:
        with pytest.raises(ValidationError):
            R2ConsensusConfig(lookback_long=30)
        with pytest.raises(ValidationError):
            R2ConsensusConfig(lookback_long=400)

    def test_r2_threshold_range(self) -> None:
        R2ConsensusConfig(r2_threshold=0.0)  # valid
        R2ConsensusConfig(r2_threshold=0.9)  # valid
        with pytest.raises(ValidationError):
            R2ConsensusConfig(r2_threshold=-0.1)
        with pytest.raises(ValidationError):
            R2ConsensusConfig(r2_threshold=1.0)

    def test_entry_threshold_range(self) -> None:
        R2ConsensusConfig(entry_threshold=0.0)  # valid
        R2ConsensusConfig(entry_threshold=1.0)  # valid
        with pytest.raises(ValidationError):
            R2ConsensusConfig(entry_threshold=-0.1)
        with pytest.raises(ValidationError):
            R2ConsensusConfig(entry_threshold=1.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            R2ConsensusConfig(vol_target=0.01, min_volatility=0.05)

    def test_lookback_ordering(self) -> None:
        """lookback_short < lookback_mid < lookback_long 검증."""
        with pytest.raises(ValidationError, match="strictly increasing"):
            R2ConsensusConfig(lookback_short=50, lookback_mid=50, lookback_long=120)
        with pytest.raises(ValidationError, match="strictly increasing"):
            R2ConsensusConfig(lookback_short=20, lookback_mid=120, lookback_long=100)
        with pytest.raises(ValidationError, match="strictly increasing"):
            R2ConsensusConfig(lookback_short=60, lookback_mid=50, lookback_long=120)

    def test_warmup_periods(self) -> None:
        config = R2ConsensusConfig()
        assert config.warmup_periods() >= config.lookback_long
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = R2ConsensusConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = R2ConsensusConfig(
            lookback_short=10,
            lookback_mid=40,
            lookback_long=100,
            r2_threshold=0.5,
        )
        assert config.lookback_short == 10
        assert config.lookback_mid == 40
        assert config.lookback_long == 100
        assert config.r2_threshold == 0.5

    def test_hedge_params(self) -> None:
        config = R2ConsensusConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_atr_period_range(self) -> None:
        R2ConsensusConfig(atr_period=5)  # valid
        R2ConsensusConfig(atr_period=50)  # valid
        with pytest.raises(ValidationError):
            R2ConsensusConfig(atr_period=2)
        with pytest.raises(ValidationError):
            R2ConsensusConfig(atr_period=100)
