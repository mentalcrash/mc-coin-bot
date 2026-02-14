"""Tests for Funding Divergence Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fund_div_mom.config import FundDivMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFundDivMomConfig:
    def test_default_values(self) -> None:
        config = FundDivMomConfig()
        assert config.mom_lookback == 18
        assert config.fr_lookback == 6
        assert config.fr_zscore_window == 90
        assert config.divergence_threshold == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FundDivMomConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            FundDivMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            FundDivMomConfig(mom_lookback=101)

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            FundDivMomConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            FundDivMomConfig(fr_lookback=51)

    def test_fr_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FundDivMomConfig(fr_zscore_window=9)
        with pytest.raises(ValidationError):
            FundDivMomConfig(fr_zscore_window=366)

    def test_divergence_threshold_range(self) -> None:
        # threshold=0 is valid (always enter)
        config = FundDivMomConfig(divergence_threshold=0.0)
        assert config.divergence_threshold == 0.0
        with pytest.raises(ValidationError):
            FundDivMomConfig(divergence_threshold=-0.1)
        with pytest.raises(ValidationError):
            FundDivMomConfig(divergence_threshold=3.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FundDivMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FundDivMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.fr_zscore_window

    def test_annualization_factor(self) -> None:
        config = FundDivMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = FundDivMomConfig(mom_lookback=24, fr_lookback=9)
        assert config.mom_lookback == 24
        assert config.fr_lookback == 9

    def test_hedge_params(self) -> None:
        config = FundDivMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
