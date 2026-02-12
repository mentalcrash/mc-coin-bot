"""Tests for Asymmetric Semivariance MR config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.asym_semivar_mr.config import AsymSemivarMRConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAsymSemivarMRConfig:
    def test_default_values(self) -> None:
        config = AsymSemivarMRConfig()
        assert config.semivar_window == 60
        assert config.zscore_window == 120
        assert config.entry_zscore == 1.5
        assert config.exit_zscore == 0.5
        assert config.exit_timeout_bars == 30
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = AsymSemivarMRConfig()
        with pytest.raises(ValidationError):
            config.semivar_window = 999  # type: ignore[misc]

    def test_semivar_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(semivar_window=5)
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(semivar_window=500)

    def test_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(zscore_window=10)
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(zscore_window=600)

    def test_entry_zscore_range(self) -> None:
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(entry_zscore=0.1)
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(entry_zscore=5.0)

    def test_exit_zscore_must_be_less_than_entry(self) -> None:
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(entry_zscore=1.5, exit_zscore=1.5)
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(entry_zscore=1.5, exit_zscore=2.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AsymSemivarMRConfig()
        warmup = config.warmup_periods()
        assert warmup >= config.semivar_window
        assert warmup >= config.zscore_window
        assert warmup >= config.mom_lookback

    def test_annualization_factor(self) -> None:
        config = AsymSemivarMRConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = AsymSemivarMRConfig(semivar_window=80, entry_zscore=2.0)
        assert config.semivar_window == 80
        assert config.entry_zscore == 2.0

    def test_hedge_params(self) -> None:
        config = AsymSemivarMRConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_exit_timeout_range(self) -> None:
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(exit_timeout_bars=5)
        with pytest.raises(ValidationError):
            AsymSemivarMRConfig(exit_timeout_bars=100)
