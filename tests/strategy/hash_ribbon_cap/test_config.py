"""Tests for Hash-Ribbon Capitulation config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.hash_ribbon_cap.config import HashRibbonCapConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestHashRibbonCapConfig:
    def test_default_values(self) -> None:
        config = HashRibbonCapConfig()
        assert config.hash_fast_window == 30
        assert config.hash_slow_window == 60
        assert config.recovery_confirm_bars == 3
        assert config.momentum_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen(self) -> None:
        config = HashRibbonCapConfig()
        with pytest.raises(ValidationError):
            config.hash_fast_window = 999  # type: ignore[misc]

    def test_hash_fast_window_range(self) -> None:
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(hash_fast_window=2, hash_slow_window=200)
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(hash_fast_window=61)

    def test_hash_slow_window_range(self) -> None:
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(hash_slow_window=10)
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(hash_slow_window=201)

    def test_fast_must_be_less_than_slow(self) -> None:
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(hash_fast_window=60, hash_slow_window=60)
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(hash_fast_window=50, hash_slow_window=40)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = HashRibbonCapConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.hash_slow_window

    def test_annualization_factor(self) -> None:
        config = HashRibbonCapConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = HashRibbonCapConfig(hash_fast_window=20, hash_slow_window=40)
        assert config.hash_fast_window == 20
        assert config.hash_slow_window == 40

    def test_recovery_confirm_bars_range(self) -> None:
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(recovery_confirm_bars=0)
        with pytest.raises(ValidationError):
            HashRibbonCapConfig(recovery_confirm_bars=11)
