"""Tests for Adaptive ROC Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.aroc_mom.config import ArocMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestArocMomConfig:
    def test_default_values(self) -> None:
        config = ArocMomConfig()
        assert config.fast_lookback == 10
        assert config.slow_lookback == 60
        assert config.vol_rank_window == 60
        assert config.mom_threshold == 0.01
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = ArocMomConfig()
        with pytest.raises(ValidationError):
            config.fast_lookback = 999  # type: ignore[misc]

    def test_fast_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            ArocMomConfig(fast_lookback=2)
        with pytest.raises(ValidationError):
            ArocMomConfig(fast_lookback=61)

    def test_slow_lookback_must_exceed_fast(self) -> None:
        with pytest.raises(ValidationError):
            ArocMomConfig(fast_lookback=30, slow_lookback=30)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ArocMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = ArocMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.slow_lookback

    def test_annualization_factor(self) -> None:
        config = ArocMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = ArocMomConfig(fast_lookback=5, slow_lookback=50)
        assert config.fast_lookback == 5
        assert config.slow_lookback == 50
