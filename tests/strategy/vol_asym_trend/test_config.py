"""Tests for vol-asym-trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_asym_trend.config import ShortMode, VolAsymTrendConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolAsymTrendConfig:
    def test_default_values(self) -> None:
        config = VolAsymTrendConfig()
        assert config.asym_window == 30
        assert config.vol_target == 0.35
        assert config.annualization_factor == 1460.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VolAsymTrendConfig()
        with pytest.raises(ValidationError):
            config.asym_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolAsymTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolAsymTrendConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VolAsymTrendConfig()
        assert config.annualization_factor == 1460.0

    def test_custom_params(self) -> None:
        config = VolAsymTrendConfig(asym_window=20)
        assert config.asym_window == 20
