"""Tests for Autocorrelation Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.autocorr_mom.config import AutocorrMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAutocorrMomConfig:
    def test_default_values(self) -> None:
        config = AutocorrMomConfig()
        assert config.autocorr_window == 20
        assert config.momentum_window == 21
        assert config.autocorr_threshold == 0.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = AutocorrMomConfig()
        with pytest.raises(ValidationError):
            config.autocorr_window = 999  # type: ignore[misc]

    def test_autocorr_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AutocorrMomConfig(autocorr_window=2)
        with pytest.raises(ValidationError):
            AutocorrMomConfig(autocorr_window=200)

    def test_momentum_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AutocorrMomConfig(momentum_window=2)
        with pytest.raises(ValidationError):
            AutocorrMomConfig(momentum_window=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AutocorrMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AutocorrMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = AutocorrMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = AutocorrMomConfig(autocorr_window=30)
        assert config.autocorr_window == 30
