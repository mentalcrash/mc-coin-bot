"""Tests for Trend Efficiency Scorer config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.trend_eff_score.config import ShortMode, TrendEffScoreConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestTrendEffScoreConfig:
    def test_default_values(self) -> None:
        config = TrendEffScoreConfig()
        assert config.er_window == 30
        assert config.roc_short == 18
        assert config.roc_medium == 42
        assert config.roc_long == 90
        assert config.er_threshold == 0.25
        assert config.min_score == 2
        assert config.adx_period == 14
        assert config.adx_threshold == 20.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = TrendEffScoreConfig()
        with pytest.raises(ValidationError):
            config.er_window = 999  # type: ignore[misc]

    def test_er_window_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendEffScoreConfig(er_window=9)
        with pytest.raises(ValidationError):
            TrendEffScoreConfig(er_window=121)

    def test_roc_periods_strictly_increasing(self) -> None:
        with pytest.raises(ValidationError):
            TrendEffScoreConfig(roc_short=42, roc_medium=42)
        with pytest.raises(ValidationError):
            TrendEffScoreConfig(roc_medium=90, roc_long=90)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            TrendEffScoreConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = TrendEffScoreConfig()
        assert config.warmup_periods() >= config.roc_long

    def test_annualization_factor(self) -> None:
        config = TrendEffScoreConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = TrendEffScoreConfig(er_window=50)
        assert config.er_window == 50
