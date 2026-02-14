"""Tests for Regime-Gated Multi-Factor MR config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.regime_mf_mr.config import RegimeMfMrConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestRegimeMfMrConfig:
    def test_default_values(self) -> None:
        config = RegimeMfMrConfig()
        assert config.bb_period == 20
        assert config.zscore_window == 20
        assert config.mr_score_window == 20
        assert config.rsi_period == 14
        assert config.rsi_oversold == 30.0
        assert config.rsi_overbought == 70.0
        assert config.volume_ma_period == 20
        assert config.volume_threshold == 1.0
        assert config.min_factor_agreement == 2
        assert config.regime_gate_threshold == 0.4
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = RegimeMfMrConfig()
        with pytest.raises(ValidationError):
            config.bb_period = 999  # type: ignore[misc]

    def test_bb_period_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(bb_period=5)
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(bb_period=200)

    def test_rsi_period_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(rsi_period=3)
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(rsi_period=60)

    def test_min_factor_agreement_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(min_factor_agreement=0)
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(min_factor_agreement=5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(vol_target=0.01, min_volatility=0.05)

    def test_rsi_oversold_lt_overbought(self) -> None:
        with pytest.raises(ValidationError):
            RegimeMfMrConfig(rsi_oversold=70.0, rsi_overbought=30.0)

    def test_warmup_periods(self) -> None:
        config = RegimeMfMrConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = RegimeMfMrConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = RegimeMfMrConfig(bb_period=30, rsi_period=21)
        assert config.bb_period == 30
        assert config.rsi_period == 21

    def test_regime_gate_threshold(self) -> None:
        config = RegimeMfMrConfig(regime_gate_threshold=0.5)
        assert config.regime_gate_threshold == 0.5

    def test_regime_adaptive_vol_targets(self) -> None:
        config = RegimeMfMrConfig(
            trending_vol_target=0.0,
            ranging_vol_target=0.40,
            volatile_vol_target=0.0,
        )
        assert config.trending_vol_target == 0.0
        assert config.ranging_vol_target == 0.40
        assert config.volatile_vol_target == 0.0
