"""Tests for dvr-mom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.dvr_mom.config import DvrMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestDvrMomConfig:
    def test_default_values(self) -> None:
        config = DvrMomConfig()
        assert config.dvr_window == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = DvrMomConfig()
        with pytest.raises(ValidationError):
            config.dvr_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DvrMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = DvrMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = DvrMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = DvrMomConfig(dvr_window=15)
        assert config.dvr_window == 15
