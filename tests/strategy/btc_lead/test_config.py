"""Tests for BtcLead config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.btc_lead.config import BtcLeadConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestBtcLeadConfig:
    def test_default_values(self) -> None:
        config = BtcLeadConfig()
        assert config.btc_mom_window == 5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = BtcLeadConfig()
        with pytest.raises(ValidationError):
            config.btc_mom_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            BtcLeadConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = BtcLeadConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = BtcLeadConfig(btc_mom_window=10)
        assert config.btc_mom_window == 10
