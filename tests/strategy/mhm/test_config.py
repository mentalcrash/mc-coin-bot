"""Tests for MHM config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.mhm.config import MHMConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMHMConfig:
    def test_default_values(self) -> None:
        config = MHMConfig()
        assert config.lookback_1 == 5
        assert config.lookback_2 == 10
        assert config.lookback_3 == 21
        assert config.lookback_4 == 63
        assert config.lookback_5 == 126
        assert config.agreement_threshold == 3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = MHMConfig()
        with pytest.raises(ValidationError):
            config.lookback_1 = 999  # type: ignore[misc]

    def test_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            MHMConfig(lookback_1=0)
        with pytest.raises(ValidationError):
            MHMConfig(lookback_5=1000)

    def test_lookbacks_strictly_increasing(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            MHMConfig(lookback_1=10, lookback_2=10)
        with pytest.raises(ValidationError, match="strictly increasing"):
            MHMConfig(lookback_3=100, lookback_4=50)

    def test_agreement_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            MHMConfig(agreement_threshold=0)
        with pytest.raises(ValidationError):
            MHMConfig(agreement_threshold=6)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MHMConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = MHMConfig()
        assert config.warmup_periods() == 136  # max(126, 30) + 10

    def test_annualization_factor(self) -> None:
        config = MHMConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = MHMConfig(lookback_1=3, lookback_2=7, agreement_threshold=4)
        assert config.lookback_1 == 3
        assert config.agreement_threshold == 4
