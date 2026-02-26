"""Tests for VWR Asymmetric Multi-Scale config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vwr_asym_multi.config import ShortMode, VwrAsymMultiConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)

    def test_all_modes_accessible(self) -> None:
        assert len(ShortMode) == 3


class TestVwrAsymMultiConfig:
    def test_default_values(self) -> None:
        config = VwrAsymMultiConfig()
        assert config.lookback_short == 10
        assert config.lookback_mid == 21
        assert config.lookback_long == 42
        assert config.zscore_window == 60
        assert config.long_threshold == 0.5
        assert config.short_threshold == 0.8
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen(self) -> None:
        config = VwrAsymMultiConfig()
        with pytest.raises(ValidationError):
            config.lookback_short = 999  # type: ignore[misc]

    def test_lookback_short_range(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_short=2)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_short=51)

    def test_lookback_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_mid=4)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_mid=101)

    def test_lookback_long_range(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_long=9)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_long=201)

    def test_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(zscore_window=19)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(zscore_window=201)

    def test_long_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(long_threshold=-0.1)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(long_threshold=3.1)

    def test_short_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(short_threshold=-0.1)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(short_threshold=3.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(vol_target=0.01, min_volatility=0.05)

    def test_lookback_ordering(self) -> None:
        """lookback_short < lookback_mid < lookback_long 필수."""
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_short=30, lookback_mid=21, lookback_long=42)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_short=10, lookback_mid=50, lookback_long=42)
        with pytest.raises(ValidationError):
            VwrAsymMultiConfig(lookback_short=21, lookback_mid=21, lookback_long=42)

    def test_warmup_periods(self) -> None:
        config = VwrAsymMultiConfig()
        warmup = config.warmup_periods()
        # lookback_long(42) + zscore_window(60) = 102 > vol_window(30)
        assert warmup >= config.lookback_long + config.zscore_window
        assert warmup >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VwrAsymMultiConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = VwrAsymMultiConfig(
            lookback_short=5,
            lookback_mid=15,
            lookback_long=30,
            long_threshold=0.3,
            short_threshold=1.0,
        )
        assert config.lookback_short == 5
        assert config.lookback_mid == 15
        assert config.lookback_long == 30
        assert config.long_threshold == 0.3
        assert config.short_threshold == 1.0

    def test_asymmetric_thresholds_allowed(self) -> None:
        """long_threshold != short_threshold 허용 확인."""
        config = VwrAsymMultiConfig(long_threshold=0.3, short_threshold=1.2)
        assert config.long_threshold != config.short_threshold
        assert config.long_threshold < config.short_threshold
