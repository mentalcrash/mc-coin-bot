"""Tests for Capitulation Wick Reversal config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cap_wick_rev.config import CapWickRevConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCapWickRevConfig:
    def test_default_values(self) -> None:
        config = CapWickRevConfig()
        assert config.atr_window == 30
        assert config.atr_spike_threshold == 2.0
        assert config.vol_surge_window == 30
        assert config.vol_surge_threshold == 2.0
        assert config.wick_ratio_threshold == 0.5
        assert config.close_position_threshold == 0.3
        assert config.confirmation_bars == 2
        assert config.exit_timeout_bars == 18
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CapWickRevConfig()
        with pytest.raises(ValidationError):
            config.atr_window = 999  # type: ignore[misc]

    def test_atr_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CapWickRevConfig(atr_window=5)
        with pytest.raises(ValidationError):
            CapWickRevConfig(atr_window=200)

    def test_atr_spike_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CapWickRevConfig(atr_spike_threshold=1.0)
        with pytest.raises(ValidationError):
            CapWickRevConfig(atr_spike_threshold=6.0)

    def test_vol_surge_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CapWickRevConfig(vol_surge_threshold=1.0)
        with pytest.raises(ValidationError):
            CapWickRevConfig(vol_surge_threshold=6.0)

    def test_wick_ratio_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CapWickRevConfig(wick_ratio_threshold=0.1)
        with pytest.raises(ValidationError):
            CapWickRevConfig(wick_ratio_threshold=0.9)

    def test_confirmation_bars_range(self) -> None:
        # 0 is valid (immediate entry)
        config = CapWickRevConfig(confirmation_bars=0)
        assert config.confirmation_bars == 0
        with pytest.raises(ValidationError):
            CapWickRevConfig(confirmation_bars=-1)
        with pytest.raises(ValidationError):
            CapWickRevConfig(confirmation_bars=10)

    def test_exit_timeout_range(self) -> None:
        with pytest.raises(ValidationError):
            CapWickRevConfig(exit_timeout_bars=3)
        with pytest.raises(ValidationError):
            CapWickRevConfig(exit_timeout_bars=60)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CapWickRevConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CapWickRevConfig()
        warmup = config.warmup_periods()
        assert warmup >= config.atr_window
        assert warmup >= config.vol_surge_window
        assert warmup >= config.mom_lookback

    def test_annualization_factor(self) -> None:
        config = CapWickRevConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = CapWickRevConfig(atr_spike_threshold=2.5, confirmation_bars=3)
        assert config.atr_spike_threshold == 2.5
        assert config.confirmation_bars == 3

    def test_hedge_params(self) -> None:
        config = CapWickRevConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
