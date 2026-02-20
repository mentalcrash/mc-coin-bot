"""Tests for Volatility Surface Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_surface_mom.config import ShortMode, VolSurfaceMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolSurfaceMomConfig:
    def test_default_values(self) -> None:
        config = VolSurfaceMomConfig()
        assert config.gk_window == 21
        assert config.pk_window == 21
        assert config.yz_window == 21
        assert config.ratio_window == 14
        assert config.momentum_window == 21
        assert config.gk_pk_long_threshold == 1.05
        assert config.gk_pk_short_threshold == 0.95
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VolSurfaceMomConfig()
        with pytest.raises(ValidationError):
            config.gk_window = 999  # type: ignore[misc]

    def test_gk_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(gk_window=3)
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(gk_window=200)

    def test_pk_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(pk_window=3)
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(pk_window=200)

    def test_ratio_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(ratio_window=1)
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(ratio_window=100)

    def test_gk_pk_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(gk_pk_long_threshold=0.5)
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(gk_pk_long_threshold=2.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_long_threshold_gt_short_threshold(self) -> None:
        with pytest.raises(ValidationError):
            VolSurfaceMomConfig(
                gk_pk_long_threshold=0.90,
                gk_pk_short_threshold=0.95,
            )

    def test_warmup_periods(self) -> None:
        config = VolSurfaceMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VolSurfaceMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = VolSurfaceMomConfig(gk_window=30, pk_window=30)
        assert config.gk_window == 30
        assert config.pk_window == 30
