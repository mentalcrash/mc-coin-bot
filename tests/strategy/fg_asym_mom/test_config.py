"""Tests for FgAsymMom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fg_asym_mom.config import FgAsymMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestFgAsymMomConfig:
    def test_default_values(self) -> None:
        config = FgAsymMomConfig()
        assert config.fear_threshold == 25.0
        assert config.greed_threshold == 75.0
        assert config.greed_hold_threshold == 55.0
        assert config.sma_short == 10
        assert config.sma_long == 20
        assert config.fg_delta_window == 5
        assert config.greed_persist_min == 5
        assert config.vol_target == 0.35
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FgAsymMomConfig()
        with pytest.raises(ValidationError):
            config.fear_threshold = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FgAsymMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_fear_lt_greed_hold(self) -> None:
        with pytest.raises(ValidationError):
            FgAsymMomConfig(fear_threshold=60.0, greed_hold_threshold=30.0)

    def test_neutral_low_lt_high(self) -> None:
        with pytest.raises(ValidationError):
            FgAsymMomConfig(neutral_low=70.0, neutral_high=30.0)

    def test_warmup_periods(self) -> None:
        config = FgAsymMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = FgAsymMomConfig(fear_threshold=20.0, sma_short=5)
        assert config.fear_threshold == 20.0
        assert config.sma_short == 5
