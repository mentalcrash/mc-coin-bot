"""Tests for Funding Rate Carry Vol config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fr_carry_vol.config import FRCarryVolConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFRCarryVolConfig:
    def test_default_values(self) -> None:
        config = FRCarryVolConfig()
        assert config.fr_lookback == 6
        assert config.fr_extreme_zscore == 1.5
        assert config.vol_condition_pctile == 0.7
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = FRCarryVolConfig()
        with pytest.raises(ValidationError):
            config.fr_lookback = 999  # type: ignore[misc]

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            FRCarryVolConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            FRCarryVolConfig(fr_lookback=50)

    def test_fr_extreme_zscore_range(self) -> None:
        with pytest.raises(ValidationError):
            FRCarryVolConfig(fr_extreme_zscore=0.1)
        with pytest.raises(ValidationError):
            FRCarryVolConfig(fr_extreme_zscore=5.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FRCarryVolConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FRCarryVolConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = FRCarryVolConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = FRCarryVolConfig(fr_lookback=12, fr_extreme_zscore=2.0)
        assert config.fr_lookback == 12
        assert config.fr_extreme_zscore == 2.0
