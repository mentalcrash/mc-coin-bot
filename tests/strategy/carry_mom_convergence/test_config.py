"""Tests for Carry-Momentum Convergence config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCarryMomConvergenceConfig:
    def test_default_values(self) -> None:
        config = CarryMomConvergenceConfig()
        assert config.mom_lookback == 20
        assert config.mom_fast == 10
        assert config.mom_slow == 30
        assert config.fr_lookback == 3
        assert config.fr_zscore_window == 90
        assert config.convergence_boost == 1.5
        assert config.divergence_penalty == 0.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CarryMomConvergenceConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(mom_lookback=121)

    def test_mom_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(mom_fast=2)
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(mom_fast=51)

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(fr_lookback=31)

    def test_convergence_boost_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(convergence_boost=0.5)
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(convergence_boost=3.5)

    def test_divergence_penalty_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(divergence_penalty=-0.1)
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(divergence_penalty=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(vol_target=0.01, min_volatility=0.05)

    def test_mom_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(mom_fast=30, mom_slow=30)
        with pytest.raises(ValidationError):
            CarryMomConvergenceConfig(mom_fast=40, mom_slow=30)

    def test_warmup_periods(self) -> None:
        config = CarryMomConvergenceConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.mom_lookback
        assert config.warmup_periods() >= config.fr_zscore_window

    def test_annualization_factor(self) -> None:
        config = CarryMomConvergenceConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = CarryMomConvergenceConfig(mom_lookback=30, fr_lookback=5, convergence_boost=2.0)
        assert config.mom_lookback == 30
        assert config.fr_lookback == 5
        assert config.convergence_boost == 2.0
