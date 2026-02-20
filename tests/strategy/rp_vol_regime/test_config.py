"""Tests for RpVolRegime config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.rp_vol_regime.config import RpVolRegimeConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestRpVolRegimeConfig:
    def test_default_values(self) -> None:
        config = RpVolRegimeConfig()
        assert config.rv_window == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = RpVolRegimeConfig()
        with pytest.raises(ValidationError):
            config.rv_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            RpVolRegimeConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = RpVolRegimeConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = RpVolRegimeConfig(rv_window=30)
        assert config.rv_window == 30
