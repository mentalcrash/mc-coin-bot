"""Tests for cft-2h config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cft_2h.config import Cft2hConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCft2hConfig:
    def test_default_values(self) -> None:
        config = Cft2hConfig()
        assert config.regime_window == 24
        assert config.vol_target == 0.35
        assert config.annualization_factor == 4380.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = Cft2hConfig()
        with pytest.raises(ValidationError):
            config.regime_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            Cft2hConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = Cft2hConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = Cft2hConfig()
        assert config.annualization_factor == 4380.0

    def test_custom_params(self) -> None:
        config = Cft2hConfig(regime_window=20)
        assert config.regime_window == 20
