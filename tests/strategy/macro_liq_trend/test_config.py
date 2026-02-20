"""Tests for Macro-Liquidity Adaptive Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMacroLiqTrendConfig:
    def test_default_values(self) -> None:
        config = MacroLiqTrendConfig()
        assert config.dxy_roc_period == 20
        assert config.vix_roc_period == 20
        assert config.spy_roc_period == 20
        assert config.stab_change_period == 14
        assert config.zscore_window == 90
        assert config.liq_long_threshold == 0.5
        assert config.liq_short_threshold == -0.5
        assert config.price_mom_period == 50
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = MacroLiqTrendConfig()
        with pytest.raises(ValidationError):
            config.dxy_roc_period = 999  # type: ignore[misc]

    def test_dxy_roc_period_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(dxy_roc_period=2)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(dxy_roc_period=200)

    def test_vix_roc_period_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(vix_roc_period=2)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(vix_roc_period=200)

    def test_spy_roc_period_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(spy_roc_period=2)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(spy_roc_period=200)

    def test_stab_change_period_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(stab_change_period=2)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(stab_change_period=100)

    def test_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(zscore_window=10)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(zscore_window=500)

    def test_price_mom_period_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(price_mom_period=5)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(price_mom_period=300)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_threshold_ordering(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(liq_long_threshold=0.0, liq_short_threshold=0.0)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(liq_long_threshold=-0.5, liq_short_threshold=0.5)

    def test_warmup_periods(self) -> None:
        config = MacroLiqTrendConfig()
        warmup = config.warmup_periods()
        assert warmup >= config.vol_window
        assert warmup >= config.zscore_window

    def test_annualization_factor(self) -> None:
        config = MacroLiqTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = MacroLiqTrendConfig(dxy_roc_period=30, zscore_window=60)
        assert config.dxy_roc_period == 30
        assert config.zscore_window == 60

    def test_hedge_params(self) -> None:
        config = MacroLiqTrendConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5

    def test_atr_period_range(self) -> None:
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(atr_period=2)
        with pytest.raises(ValidationError):
            MacroLiqTrendConfig(atr_period=60)
