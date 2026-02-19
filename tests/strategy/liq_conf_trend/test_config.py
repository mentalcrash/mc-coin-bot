"""Tests for Liquidity-Confirmed Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.liq_conf_trend.config import LiqConfTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestLiqConfTrendConfig:
    def test_default_values(self) -> None:
        config = LiqConfTrendConfig()
        assert config.mom_lookback == 20
        assert config.stablecoin_roc_window == 14
        assert config.tvl_roc_window == 14
        assert config.liq_score_threshold == 1
        assert config.fg_fear_threshold == 20
        assert config.fg_greed_threshold == 80
        assert config.fg_ma_window == 14
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = LiqConfTrendConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(mom_lookback=61)

    def test_liq_score_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(liq_score_threshold=0)
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(liq_score_threshold=3)

    def test_fg_fear_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(fg_fear_threshold=4)
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(fg_fear_threshold=36)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_fear_lt_greed(self) -> None:
        with pytest.raises(ValidationError):
            LiqConfTrendConfig(fg_fear_threshold=30, fg_greed_threshold=30)

    def test_warmup_periods(self) -> None:
        config = LiqConfTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.mom_lookback

    def test_annualization_factor(self) -> None:
        config = LiqConfTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = LiqConfTrendConfig(mom_lookback=30, liq_score_threshold=2)
        assert config.mom_lookback == 30
        assert config.liq_score_threshold == 2
