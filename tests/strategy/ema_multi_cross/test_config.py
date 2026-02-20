"""Tests for EMA Multi-Cross config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ema_multi_cross.config import EmaMultiCrossConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestEmaMultiCrossConfig:
    def test_default_values(self) -> None:
        config = EmaMultiCrossConfig()
        assert config.pair1_fast == 8
        assert config.pair1_slow == 21
        assert config.pair2_fast == 20
        assert config.pair2_slow == 50
        assert config.pair3_fast == 50
        assert config.pair3_slow == 100
        assert config.min_votes == 2
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen(self) -> None:
        config = EmaMultiCrossConfig()
        with pytest.raises(ValidationError):
            config.pair1_fast = 999  # type: ignore[misc]

    def test_pair_ordering_validation(self) -> None:
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(pair1_fast=21, pair1_slow=21)
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(pair2_fast=50, pair2_slow=30)
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(pair3_fast=100, pair3_slow=60)

    def test_min_votes_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(min_votes=1)
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(min_votes=4)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = EmaMultiCrossConfig()
        assert config.warmup_periods() >= config.pair3_slow
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = EmaMultiCrossConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = EmaMultiCrossConfig(pair1_fast=5, pair1_slow=15)
        assert config.pair1_fast == 5
        assert config.pair1_slow == 15

    def test_min_votes_3(self) -> None:
        config = EmaMultiCrossConfig(min_votes=3)
        assert config.min_votes == 3

    def test_pair1_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(pair1_fast=2)
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(pair1_fast=31)

    def test_hedge_threshold_le_zero(self) -> None:
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(hedge_threshold=0.1)

    def test_hedge_strength_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(hedge_strength_ratio=0.0)
        with pytest.raises(ValidationError):
            EmaMultiCrossConfig(hedge_strength_ratio=1.1)
