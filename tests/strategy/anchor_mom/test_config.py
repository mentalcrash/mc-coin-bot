"""Tests for Anchored Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.anchor_mom.config import AnchorMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAnchorMomConfig:
    def test_default_values(self) -> None:
        config = AnchorMomConfig()
        assert config.nearness_lookback == 60
        assert config.mom_lookback == 30
        assert config.strong_nearness == 0.95
        assert config.weak_nearness == 0.85
        assert config.short_nearness == 0.80
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = AnchorMomConfig()
        with pytest.raises(ValidationError):
            config.nearness_lookback = 999  # type: ignore[misc]

    def test_nearness_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            AnchorMomConfig(nearness_lookback=9)
        with pytest.raises(ValidationError):
            AnchorMomConfig(nearness_lookback=201)

    def test_strong_gt_weak_gt_short(self) -> None:
        with pytest.raises(ValidationError):
            AnchorMomConfig(strong_nearness=0.85, weak_nearness=0.85)
        with pytest.raises(ValidationError):
            AnchorMomConfig(weak_nearness=0.80, short_nearness=0.80)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AnchorMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AnchorMomConfig()
        assert config.warmup_periods() >= config.nearness_lookback

    def test_annualization_factor(self) -> None:
        config = AnchorMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = AnchorMomConfig(nearness_lookback=80)
        assert config.nearness_lookback == 80
