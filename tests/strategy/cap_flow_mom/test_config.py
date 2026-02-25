"""Tests for Capital Flow Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cap_flow_mom.config import CapFlowMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCapFlowMomConfig:
    def test_default_values(self) -> None:
        config = CapFlowMomConfig()
        assert config.fast_roc_period == 6
        assert config.slow_roc_period == 30
        assert config.roc_threshold == 0.01
        assert config.stablecoin_roc_window == 14
        assert config.stablecoin_boost == 1.3
        assert config.stablecoin_dampen == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CapFlowMomConfig()
        with pytest.raises(ValidationError):
            config.fast_roc_period = 999  # type: ignore[misc]

    def test_fast_roc_period_range(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(fast_roc_period=1, slow_roc_period=30)
        with pytest.raises(ValidationError):
            CapFlowMomConfig(fast_roc_period=31)

    def test_slow_roc_period_range(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(slow_roc_period=9)
        with pytest.raises(ValidationError):
            CapFlowMomConfig(slow_roc_period=121)

    def test_roc_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(roc_threshold=0.0005)
        with pytest.raises(ValidationError):
            CapFlowMomConfig(roc_threshold=0.11)

    def test_stablecoin_boost_range(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(stablecoin_boost=0.9)
        with pytest.raises(ValidationError):
            CapFlowMomConfig(stablecoin_boost=2.1)

    def test_stablecoin_dampen_range(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(stablecoin_dampen=0.05)
        with pytest.raises(ValidationError):
            CapFlowMomConfig(stablecoin_dampen=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_fast_lt_slow_roc(self) -> None:
        with pytest.raises(ValidationError):
            CapFlowMomConfig(fast_roc_period=30, slow_roc_period=30)
        with pytest.raises(ValidationError):
            CapFlowMomConfig(fast_roc_period=20, slow_roc_period=15)

    def test_warmup_periods(self) -> None:
        config = CapFlowMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.slow_roc_period

    def test_annualization_factor(self) -> None:
        config = CapFlowMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = CapFlowMomConfig(fast_roc_period=10, slow_roc_period=40)
        assert config.fast_roc_period == 10
        assert config.slow_roc_period == 40

    def test_hedge_threshold(self) -> None:
        config = CapFlowMomConfig()
        assert config.hedge_threshold <= 0.0

    def test_hedge_strength_ratio(self) -> None:
        config = CapFlowMomConfig()
        assert 0.0 < config.hedge_strength_ratio <= 1.0
