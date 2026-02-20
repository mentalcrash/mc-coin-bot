"""Tests for TrendPersist config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.trend_persist.config import ShortMode, TrendPersistConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestTrendPersistConfig:
    def test_default_values(self) -> None:
        config = TrendPersistConfig()
        assert config.persist_window == 21
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = TrendPersistConfig()
        with pytest.raises(ValidationError):
            config.persist_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            TrendPersistConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = TrendPersistConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = TrendPersistConfig(persist_window=30)
        assert config.persist_window == 30
