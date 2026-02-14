"""Tests for Carry-Conditional Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.carry_cond_mom.config import CarryCondMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCarryCondMomConfig:
    def test_default_values(self) -> None:
        config = CarryCondMomConfig()
        assert config.mom_lookback == 18
        assert config.fr_lookback == 6
        assert config.fr_zscore_window == 90
        assert config.agreement_boost == 1.2
        assert config.disagreement_penalty == 0.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = CarryCondMomConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryCondMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            CarryCondMomConfig(mom_lookback=101)

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryCondMomConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            CarryCondMomConfig(fr_lookback=51)

    def test_agreement_boost_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryCondMomConfig(agreement_boost=0.9)
        with pytest.raises(ValidationError):
            CarryCondMomConfig(agreement_boost=2.1)

    def test_disagreement_penalty_range(self) -> None:
        config = CarryCondMomConfig(disagreement_penalty=0.0)
        assert config.disagreement_penalty == 0.0
        with pytest.raises(ValidationError):
            CarryCondMomConfig(disagreement_penalty=-0.1)
        with pytest.raises(ValidationError):
            CarryCondMomConfig(disagreement_penalty=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CarryCondMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CarryCondMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.fr_zscore_window

    def test_annualization_factor(self) -> None:
        config = CarryCondMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = CarryCondMomConfig(mom_lookback=24, agreement_boost=1.5)
        assert config.mom_lookback == 24
        assert config.agreement_boost == 1.5

    def test_hedge_params(self) -> None:
        config = CarryCondMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
