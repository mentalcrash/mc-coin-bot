"""Tests for Kurtosis Carry config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.kurtosis_carry.config import KurtosisCarryConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestKurtosisCarryConfig:
    def test_default_values(self) -> None:
        config = KurtosisCarryConfig()
        assert config.kurtosis_window == 30
        assert config.kurtosis_long_window == 90
        assert config.zscore_window == 60
        assert config.high_kurtosis_zscore == 1.0
        assert config.low_kurtosis_zscore == -0.5
        assert config.momentum_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = KurtosisCarryConfig()
        with pytest.raises(ValidationError):
            config.kurtosis_window = 999  # type: ignore[misc]

    def test_kurtosis_window_range(self) -> None:
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(kurtosis_window=5)
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(kurtosis_window=121, kurtosis_long_window=365)

    def test_kurtosis_long_window_range(self) -> None:
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(kurtosis_long_window=20)
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(kurtosis_long_window=366)

    def test_short_must_be_less_than_long(self) -> None:
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(kurtosis_window=90, kurtosis_long_window=90)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = KurtosisCarryConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.kurtosis_long_window

    def test_annualization_factor(self) -> None:
        config = KurtosisCarryConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = KurtosisCarryConfig(kurtosis_window=20, kurtosis_long_window=60)
        assert config.kurtosis_window == 20
        assert config.kurtosis_long_window == 60

    def test_high_kurtosis_zscore_range(self) -> None:
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(high_kurtosis_zscore=0.2)
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(high_kurtosis_zscore=3.1)

    def test_low_kurtosis_zscore_range(self) -> None:
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(low_kurtosis_zscore=0.0)
        with pytest.raises(ValidationError):
            KurtosisCarryConfig(low_kurtosis_zscore=-3.1)
