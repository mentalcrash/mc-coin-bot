"""Tests for Drawdown-Recovery Phase config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.dd_recovery_phase.config import DDRecoveryPhaseConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestDDRecoveryPhaseConfig:
    def test_default_values(self) -> None:
        config = DDRecoveryPhaseConfig()
        assert config.dd_threshold == -0.15
        assert config.recovery_ratio == 0.50
        assert config.dd_lookback == 60
        assert config.momentum_lookback == 10
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = DDRecoveryPhaseConfig()
        with pytest.raises(ValidationError):
            config.dd_threshold = -0.5  # type: ignore[misc]

    def test_dd_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(dd_threshold=-0.01)
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(dd_threshold=-0.51)

    def test_recovery_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(recovery_ratio=0.1)
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(recovery_ratio=0.95)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = DDRecoveryPhaseConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.dd_lookback

    def test_annualization_factor(self) -> None:
        config = DDRecoveryPhaseConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = DDRecoveryPhaseConfig(dd_threshold=-0.20, recovery_ratio=0.60)
        assert config.dd_threshold == -0.20
        assert config.recovery_ratio == 0.60

    def test_dd_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(dd_lookback=10)
        with pytest.raises(ValidationError):
            DDRecoveryPhaseConfig(dd_lookback=201)
