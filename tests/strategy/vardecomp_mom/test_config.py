"""Tests for Variance Decomposition Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vardecomp_mom.config import ShortMode, VardecompMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVardecompMomConfig:
    def test_default_values(self) -> None:
        config = VardecompMomConfig()
        assert config.semivar_window == 30
        assert config.mom_lookback == 20
        assert config.var_ratio_threshold == 0.55
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VardecompMomConfig()
        with pytest.raises(ValidationError):
            config.semivar_window = 999  # type: ignore[misc]

    def test_semivar_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VardecompMomConfig(semivar_window=4)
        with pytest.raises(ValidationError):
            VardecompMomConfig(semivar_window=201)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            VardecompMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            VardecompMomConfig(mom_lookback=101)

    def test_var_ratio_threshold_range(self) -> None:
        config = VardecompMomConfig(var_ratio_threshold=0.5)
        assert config.var_ratio_threshold == 0.5
        with pytest.raises(ValidationError):
            VardecompMomConfig(var_ratio_threshold=0.49)
        with pytest.raises(ValidationError):
            VardecompMomConfig(var_ratio_threshold=0.96)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VardecompMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VardecompMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.semivar_window

    def test_annualization_factor(self) -> None:
        config = VardecompMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = VardecompMomConfig(semivar_window=50, mom_lookback=30)
        assert config.semivar_window == 50
        assert config.mom_lookback == 30

    def test_hedge_params(self) -> None:
        config = VardecompMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
