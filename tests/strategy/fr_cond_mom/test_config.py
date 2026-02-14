"""Tests for fr-cond-mom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fr_cond_mom.config import FrCondMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFrCondMomConfig:
    def test_default_values(self) -> None:
        config = FrCondMomConfig()
        assert config.mom_lookback == 20
        assert config.mom_ma_window == 5
        assert config.fr_ma_window == 9
        assert config.fr_zscore_window == 60
        assert config.fr_neutral_zone == 0.5
        assert config.fr_extreme_threshold == 2.0
        assert config.fr_dampening == 0.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 1460.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FrCondMomConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            FrCondMomConfig(mom_lookback=121)

    def test_mom_ma_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(mom_ma_window=1)
        with pytest.raises(ValidationError):
            FrCondMomConfig(mom_ma_window=31)

    def test_fr_ma_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_ma_window=2)
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_ma_window=61)

    def test_fr_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_zscore_window=19)
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_zscore_window=201)

    def test_fr_neutral_zone_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_neutral_zone=-0.1)
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_neutral_zone=2.1)

    def test_fr_extreme_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_extreme_threshold=0.4)
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_extreme_threshold=5.1)

    def test_fr_dampening_range(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_dampening=-0.1)
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_dampening=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_neutral_zone_lt_extreme_threshold(self) -> None:
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_neutral_zone=2.0, fr_extreme_threshold=2.0)
        with pytest.raises(ValidationError):
            FrCondMomConfig(fr_neutral_zone=2.0, fr_extreme_threshold=1.5)

    def test_warmup_periods(self) -> None:
        config = FrCondMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.fr_zscore_window

    def test_annualization_factor(self) -> None:
        config = FrCondMomConfig()
        assert config.annualization_factor == 1460.0

    def test_custom_params(self) -> None:
        config = FrCondMomConfig(mom_lookback=30, fr_dampening=0.5)
        assert config.mom_lookback == 30
        assert config.fr_dampening == 0.5

    def test_hedge_params(self) -> None:
        config = FrCondMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
