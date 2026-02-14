"""Tests for up-vol-mom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.up_vol_mom.config import ShortMode, UpVolMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestUpVolMomConfig:
    def test_default_values(self) -> None:
        config = UpVolMomConfig()
        assert config.semivar_window == 20
        assert config.ratio_ma_window == 10
        assert config.ratio_threshold == 0.55
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = UpVolMomConfig()
        with pytest.raises(ValidationError):
            config.semivar_window = 999  # type: ignore[misc]

    def test_semivar_window_range(self) -> None:
        with pytest.raises(ValidationError):
            UpVolMomConfig(semivar_window=4)
        with pytest.raises(ValidationError):
            UpVolMomConfig(semivar_window=121)

    def test_ratio_ma_window_range(self) -> None:
        with pytest.raises(ValidationError):
            UpVolMomConfig(ratio_ma_window=2)
        with pytest.raises(ValidationError):
            UpVolMomConfig(ratio_ma_window=61)

    def test_ratio_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            UpVolMomConfig(ratio_threshold=0.5)  # gt=0.5, not >=
        with pytest.raises(ValidationError):
            UpVolMomConfig(ratio_threshold=0.81)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            UpVolMomConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            UpVolMomConfig(mom_lookback=121)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            UpVolMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = UpVolMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_warmup_periods_covers_semivar_plus_ma(self) -> None:
        config = UpVolMomConfig()
        assert config.warmup_periods() >= config.semivar_window + config.ratio_ma_window

    def test_annualization_factor(self) -> None:
        config = UpVolMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = UpVolMomConfig(semivar_window=30, ratio_threshold=0.6)
        assert config.semivar_window == 30
        assert config.ratio_threshold == 0.6

    def test_hedge_params(self) -> None:
        config = UpVolMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
