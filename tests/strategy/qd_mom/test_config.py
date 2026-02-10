"""Tests for Quarter-Day TSMOM config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.qd_mom.config import QdMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestQdMomConfig:
    def test_default_values(self) -> None:
        config = QdMomConfig()
        assert config.vol_filter_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 1460.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = QdMomConfig()
        with pytest.raises(ValidationError):
            config.vol_filter_lookback = 999  # type: ignore[misc]

    def test_vol_filter_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            QdMomConfig(vol_filter_lookback=4)
        with pytest.raises(ValidationError):
            QdMomConfig(vol_filter_lookback=101)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            QdMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = QdMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = QdMomConfig()
        assert config.annualization_factor == 1460.0

    def test_custom_params(self) -> None:
        config = QdMomConfig(vol_filter_lookback=40)
        assert config.vol_filter_lookback == 40
