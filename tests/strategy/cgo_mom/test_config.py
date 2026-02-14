"""Tests for Capital Gains Overhang Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cgo_mom.config import CgoMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCgoMomConfig:
    def test_default_values(self) -> None:
        config = CgoMomConfig()
        assert config.turnover_window == 60
        assert config.cgo_zscore_window == 90
        assert config.cgo_threshold == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = CgoMomConfig()
        with pytest.raises(ValidationError):
            config.turnover_window = 999  # type: ignore[misc]

    def test_turnover_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CgoMomConfig(turnover_window=9)
        with pytest.raises(ValidationError):
            CgoMomConfig(turnover_window=366)

    def test_cgo_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CgoMomConfig(cgo_zscore_window=9)
        with pytest.raises(ValidationError):
            CgoMomConfig(cgo_zscore_window=366)

    def test_cgo_threshold_range(self) -> None:
        config = CgoMomConfig(cgo_threshold=0.0)
        assert config.cgo_threshold == 0.0
        with pytest.raises(ValidationError):
            CgoMomConfig(cgo_threshold=-0.1)
        with pytest.raises(ValidationError):
            CgoMomConfig(cgo_threshold=3.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CgoMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CgoMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.cgo_zscore_window

    def test_annualization_factor(self) -> None:
        config = CgoMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = CgoMomConfig(turnover_window=30, cgo_threshold=1.0)
        assert config.turnover_window == 30
        assert config.cgo_threshold == 1.0

    def test_hedge_params(self) -> None:
        config = CgoMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
