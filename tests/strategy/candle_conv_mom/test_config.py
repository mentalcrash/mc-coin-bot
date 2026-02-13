"""Tests for candle-conv-mom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.candle_conv_mom.config import CandleConvMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCandleConvMomConfig:
    def test_default_values(self) -> None:
        config = CandleConvMomConfig()
        assert config.conv_window == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = CandleConvMomConfig()
        with pytest.raises(ValidationError):
            config.conv_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CandleConvMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CandleConvMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = CandleConvMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = CandleConvMomConfig(conv_window=10)
        assert config.conv_window == 10
