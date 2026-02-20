"""Tests for EMA Ribbon Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ema_ribbon_mom.config import EmaRibbonMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestEmaRibbonMomConfig:
    def test_default_values(self) -> None:
        config = EmaRibbonMomConfig()
        assert config.ema_periods == (8, 13, 21, 34, 55)
        assert config.roc_period == 21
        assert config.alignment_threshold == 0.7
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen(self) -> None:
        config = EmaRibbonMomConfig()
        with pytest.raises(ValidationError):
            config.roc_period = 999  # type: ignore[misc]

    def test_ema_periods_ascending(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(ema_periods=(21, 13, 8))

    def test_ema_periods_min_3(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(ema_periods=(8, 21))

    def test_alignment_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(alignment_threshold=0.2)
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(alignment_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = EmaRibbonMomConfig()
        assert config.warmup_periods() >= max(config.ema_periods)
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = EmaRibbonMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_ema_periods(self) -> None:
        config = EmaRibbonMomConfig(ema_periods=(10, 20, 40))
        assert config.ema_periods == (10, 20, 40)

    def test_roc_period_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(roc_period=4)
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(roc_period=64)

    def test_hedge_threshold_le_zero(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(hedge_threshold=0.1)

    def test_hedge_strength_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(hedge_strength_ratio=0.0)
        with pytest.raises(ValidationError):
            EmaRibbonMomConfig(hedge_strength_ratio=1.1)
