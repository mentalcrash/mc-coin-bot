"""Tests for ER Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.er_trend.config import ErTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestErTrendConfig:
    def test_default_values(self) -> None:
        config = ErTrendConfig()
        assert config.er_fast == 10
        assert config.er_mid == 21
        assert config.er_slow == 42
        assert config.w_fast == 0.25
        assert config.w_mid == 0.50
        assert config.w_slow == 0.25
        assert config.entry_threshold == 0.15
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = ErTrendConfig()
        with pytest.raises(ValidationError):
            config.er_fast = 999  # type: ignore[misc]

    def test_er_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            ErTrendConfig(er_fast=2)
        with pytest.raises(ValidationError):
            ErTrendConfig(er_fast=31)

    def test_er_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            ErTrendConfig(er_mid=9)
        with pytest.raises(ValidationError):
            ErTrendConfig(er_mid=61)

    def test_er_slow_range(self) -> None:
        with pytest.raises(ValidationError):
            ErTrendConfig(er_slow=19)
        with pytest.raises(ValidationError):
            ErTrendConfig(er_slow=121)

    def test_er_periods_strictly_increasing(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            ErTrendConfig(er_fast=21, er_mid=21, er_slow=42)
        with pytest.raises(ValidationError, match="strictly increasing"):
            ErTrendConfig(er_fast=10, er_mid=42, er_slow=42)
        with pytest.raises(ValidationError, match="strictly increasing"):
            ErTrendConfig(er_fast=30, er_mid=20, er_slow=42)

    def test_weights_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"Weights must sum to 1\.0"):
            ErTrendConfig(w_fast=0.5, w_mid=0.5, w_slow=0.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ErTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = ErTrendConfig()
        assert config.warmup_periods() >= config.er_slow
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = ErTrendConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = ErTrendConfig(er_fast=5, er_mid=15, er_slow=30)
        assert config.er_fast == 5
        assert config.er_mid == 15
        assert config.er_slow == 30

    def test_custom_weights(self) -> None:
        config = ErTrendConfig(w_fast=0.33, w_mid=0.34, w_slow=0.33)
        assert abs(config.w_fast + config.w_mid + config.w_slow - 1.0) < 1e-6

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            ErTrendConfig(entry_threshold=0.0)
        with pytest.raises(ValidationError):
            ErTrendConfig(entry_threshold=0.9)

    def test_hedge_parameters(self) -> None:
        config = ErTrendConfig()
        assert config.hedge_threshold <= 0.0
        assert 0.0 < config.hedge_strength_ratio <= 1.0
