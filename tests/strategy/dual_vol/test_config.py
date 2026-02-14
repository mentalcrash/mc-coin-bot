"""Tests for Dual Volatility Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.dual_vol.config import DualVolConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestDualVolConfig:
    def test_default_values(self) -> None:
        config = DualVolConfig()
        assert config.vol_estimator_window == 20
        assert config.ratio_upper == 1.2
        assert config.ratio_lower == 0.8
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = DualVolConfig()
        with pytest.raises(ValidationError):
            config.vol_estimator_window = 999  # type: ignore[misc]

    def test_vol_estimator_window_range(self) -> None:
        with pytest.raises(ValidationError):
            DualVolConfig(vol_estimator_window=4)
        with pytest.raises(ValidationError):
            DualVolConfig(vol_estimator_window=101)

    def test_ratio_upper_must_exceed_lower(self) -> None:
        with pytest.raises(ValidationError):
            DualVolConfig(ratio_upper=0.8, ratio_lower=0.8)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DualVolConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = DualVolConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.vol_estimator_window

    def test_annualization_factor(self) -> None:
        config = DualVolConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = DualVolConfig(vol_estimator_window=30, ratio_upper=1.5)
        assert config.vol_estimator_window == 30
        assert config.ratio_upper == 1.5
