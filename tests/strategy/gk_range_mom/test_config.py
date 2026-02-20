"""Tests for GK Range Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.gk_range_mom.config import GkRangeMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestGkRangeMomConfig:
    def test_default_values(self) -> None:
        config = GkRangeMomConfig()
        assert config.range_window == 14
        assert config.gk_window == 21
        assert config.long_threshold == 0.65
        assert config.short_threshold == 0.35
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = GkRangeMomConfig()
        with pytest.raises(ValidationError):
            config.range_window = 999  # type: ignore[misc]

    def test_range_window_range(self) -> None:
        with pytest.raises(ValidationError):
            GkRangeMomConfig(range_window=1)
        with pytest.raises(ValidationError):
            GkRangeMomConfig(range_window=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            GkRangeMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = GkRangeMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = GkRangeMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = GkRangeMomConfig(range_window=20, gk_window=30)
        assert config.range_window == 20
        assert config.gk_window == 30
