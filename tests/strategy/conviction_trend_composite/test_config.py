"""Tests for Conviction Trend Composite config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.conviction_trend_composite.config import (
    ConvictionTrendCompositeConfig,
    ShortMode,
)


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestConvictionTrendCompositeConfig:
    def test_default_values(self) -> None:
        config = ConvictionTrendCompositeConfig()
        assert config.mom_lookback == 20
        assert config.mom_fast == 10
        assert config.mom_slow == 30
        assert config.obv_fast == 10
        assert config.obv_slow == 30
        assert config.rv_short_window == 10
        assert config.rv_long_window == 60
        assert config.conviction_threshold == 0.4
        assert config.trending_vol_target == 0.40
        assert config.ranging_vol_target == 0.15
        assert config.volatile_vol_target == 0.10
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = ConvictionTrendCompositeConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(mom_lookback=121)

    def test_obv_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(obv_fast=2)
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(obv_fast=51)

    def test_rv_short_window_range(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(rv_short_window=2)
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(rv_short_window=31)

    def test_conviction_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(conviction_threshold=-0.1)
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(conviction_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(vol_target=0.01, min_volatility=0.05)

    def test_mom_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(mom_fast=30, mom_slow=30)

    def test_obv_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(obv_fast=30, obv_slow=30)

    def test_rv_short_lt_long(self) -> None:
        with pytest.raises(ValidationError):
            ConvictionTrendCompositeConfig(rv_short_window=25, rv_long_window=20)

    def test_warmup_periods(self) -> None:
        config = ConvictionTrendCompositeConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.mom_slow
        assert config.warmup_periods() >= config.rv_long_window

    def test_annualization_factor(self) -> None:
        config = ConvictionTrendCompositeConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = ConvictionTrendCompositeConfig(
            mom_lookback=30, conviction_threshold=0.6, rv_long_window=90
        )
        assert config.mom_lookback == 30
        assert config.conviction_threshold == 0.6
        assert config.rv_long_window == 90
