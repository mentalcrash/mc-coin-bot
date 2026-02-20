"""Tests for FR-Pred config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fr_pred.config import FRPredConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFRPredConfig:
    def test_default_values(self) -> None:
        config = FRPredConfig()
        assert config.fr_ma_window == 7
        assert config.fr_zscore_window == 60
        assert config.fr_mr_threshold == 2.0
        assert config.fr_mom_fast == 7
        assert config.fr_mom_slow == 21
        assert config.mr_weight == 0.5
        assert config.mom_weight == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = FRPredConfig()
        with pytest.raises(ValidationError):
            config.fr_ma_window = 999  # type: ignore[misc]

    def test_fr_ma_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FRPredConfig(fr_ma_window=1)
        with pytest.raises(ValidationError):
            FRPredConfig(fr_ma_window=100)

    def test_fr_mr_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            FRPredConfig(fr_mr_threshold=0.1)
        with pytest.raises(ValidationError):
            FRPredConfig(fr_mr_threshold=5.0)

    def test_fr_mom_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError, match="fr_mom_fast"):
            FRPredConfig(fr_mom_fast=21, fr_mom_slow=21)
        with pytest.raises(ValidationError, match="fr_mom_fast"):
            FRPredConfig(fr_mom_fast=30, fr_mom_slow=21)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FRPredConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FRPredConfig()
        assert config.warmup_periods() == 70  # max(60, 21, 30) + 10

    def test_annualization_factor(self) -> None:
        config = FRPredConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = FRPredConfig(fr_mr_threshold=1.5, mr_weight=0.7, mom_weight=0.3)
        assert config.fr_mr_threshold == 1.5
        assert config.mr_weight == 0.7
