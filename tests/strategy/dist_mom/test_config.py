"""Tests for Return Distribution Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.dist_mom.config import DistMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestDistMomConfig:
    def test_default_values(self) -> None:
        config = DistMomConfig()
        assert config.dist_window == 21
        assert config.long_threshold == 0.60
        assert config.short_threshold == 0.40
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = DistMomConfig()
        with pytest.raises(ValidationError):
            config.dist_window = 999  # type: ignore[misc]

    def test_dist_window_range(self) -> None:
        with pytest.raises(ValidationError):
            DistMomConfig(dist_window=2)
        with pytest.raises(ValidationError):
            DistMomConfig(dist_window=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DistMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = DistMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = DistMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = DistMomConfig(dist_window=30, skew_window=30)
        assert config.dist_window == 30
        assert config.skew_window == 30
