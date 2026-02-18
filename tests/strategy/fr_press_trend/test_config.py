"""Tests for Funding Pressure Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fr_press_trend.config import FrPressTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFrPressTrendConfig:
    def test_default_values(self) -> None:
        config = FrPressTrendConfig()
        assert config.sma_fast == 10
        assert config.sma_slow == 42
        assert config.er_window == 20
        assert config.er_threshold == 0.20
        assert config.fr_ma_window == 21
        assert config.fr_zscore_window == 42
        assert config.fr_aligned_threshold == 1.5
        assert config.fr_extreme_threshold == 2.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = FrPressTrendConfig()
        with pytest.raises(ValidationError):
            config.sma_fast = 999  # type: ignore[misc]

    def test_sma_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            FrPressTrendConfig(sma_fast=42, sma_slow=42)
        with pytest.raises(ValidationError):
            FrPressTrendConfig(sma_fast=50, sma_slow=42)

    def test_fr_aligned_lt_extreme(self) -> None:
        with pytest.raises(ValidationError):
            FrPressTrendConfig(fr_aligned_threshold=2.5, fr_extreme_threshold=2.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FrPressTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FrPressTrendConfig()
        assert config.warmup_periods() >= config.sma_slow

    def test_annualization_factor(self) -> None:
        config = FrPressTrendConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = FrPressTrendConfig(sma_fast=15)
        assert config.sma_fast == 15
