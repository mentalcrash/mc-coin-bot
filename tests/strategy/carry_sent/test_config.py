"""Tests for Carry-Sentiment Gate config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.carry_sent.config import CarrySentConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCarrySentConfig:
    def test_default_values(self) -> None:
        config = CarrySentConfig()
        assert config.fr_lookback == 3
        assert config.fr_zscore_window == 90
        assert config.fr_entry_threshold == 0.0001
        assert config.fg_fear_threshold == 20
        assert config.fg_greed_threshold == 80
        assert config.fg_gate_low == 30
        assert config.fg_gate_high == 70
        assert config.fg_ma_window == 14
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CarrySentConfig()
        with pytest.raises(ValidationError):
            config.fr_lookback = 999  # type: ignore[misc]

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            CarrySentConfig(fr_lookback=31)

    def test_fg_fear_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_fear_threshold=4)
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_fear_threshold=36)

    def test_fg_greed_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_greed_threshold=64)
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_greed_threshold=96)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(vol_target=0.01, min_volatility=0.05)

    def test_fear_lt_greed(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_fear_threshold=30, fg_greed_threshold=30)

    def test_gate_low_lt_gate_high(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_gate_low=70, fg_gate_high=30)

    def test_fear_lt_gate_low(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_fear_threshold=30, fg_gate_low=30)

    def test_gate_high_lt_greed(self) -> None:
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_gate_high=80, fg_greed_threshold=80)

    def test_zone_overlap_fear_gate(self) -> None:
        """fg_fear_threshold >= fg_gate_low causes zone overlap."""
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_fear_threshold=35, fg_gate_low=30)

    def test_zone_overlap_gate_greed(self) -> None:
        """fg_gate_high >= fg_greed_threshold causes zone overlap."""
        with pytest.raises(ValidationError):
            CarrySentConfig(fg_gate_high=85, fg_greed_threshold=80)

    def test_warmup_periods(self) -> None:
        config = CarrySentConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.fr_zscore_window

    def test_annualization_factor(self) -> None:
        config = CarrySentConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = CarrySentConfig(fr_lookback=5, fg_fear_threshold=15)
        assert config.fr_lookback == 5
        assert config.fg_fear_threshold == 15
