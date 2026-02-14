"""Tests for Jump Drift Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.jump_drift_mom.config import JumpDriftMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestJumpDriftMomConfig:
    def test_default_values(self) -> None:
        config = JumpDriftMomConfig()
        assert config.rv_window == 20
        assert config.bpv_window == 20
        assert config.jump_zscore_threshold == 1.5
        assert config.drift_lookback == 6
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = JumpDriftMomConfig()
        with pytest.raises(ValidationError):
            config.rv_window = 999  # type: ignore[misc]

    def test_rv_window_range(self) -> None:
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(rv_window=4)
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(rv_window=101)

    def test_bpv_window_range(self) -> None:
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(bpv_window=4)
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(bpv_window=101)

    def test_jump_zscore_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(jump_zscore_threshold=0.4)
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(jump_zscore_threshold=5.1)

    def test_drift_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(drift_lookback=0)
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(drift_lookback=51)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            JumpDriftMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = JumpDriftMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.rv_window

    def test_annualization_factor(self) -> None:
        config = JumpDriftMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = JumpDriftMomConfig(rv_window=30, bpv_window=30)
        assert config.rv_window == 30
        assert config.bpv_window == 30

    def test_hedge_params(self) -> None:
        config = JumpDriftMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
