"""Tests for Anti-Correlation Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.anti_corr_mom.config import AntiCorrMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAntiCorrMomConfig:
    def test_default_values(self) -> None:
        config = AntiCorrMomConfig()
        assert config.corr_window == 30
        assert config.corr_threshold == 0.5
        assert config.momentum_window == 21
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = AntiCorrMomConfig()
        with pytest.raises(ValidationError):
            config.corr_window = 999  # type: ignore[misc]

    def test_corr_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AntiCorrMomConfig(corr_window=5)
        with pytest.raises(ValidationError):
            AntiCorrMomConfig(corr_window=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AntiCorrMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AntiCorrMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = AntiCorrMomConfig(corr_window=50)
        assert config.corr_window == 50
