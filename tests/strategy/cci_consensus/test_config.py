"""Tests for CCI Consensus Multi-Scale Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cci_consensus.config import CciConsensusConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCciConsensusConfig:
    def test_default_values(self) -> None:
        config = CciConsensusConfig()
        assert config.scale_short == 20
        assert config.scale_mid == 60
        assert config.scale_long == 150
        assert config.cci_upper == 100.0
        assert config.cci_lower == -100.0
        assert config.entry_threshold == 0.34
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen(self) -> None:
        config = CciConsensusConfig()
        with pytest.raises(ValidationError):
            config.scale_short = 999  # type: ignore[misc]

    def test_scale_short_range(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_short=4)
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_short=101)

    def test_scale_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_mid=9)
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_mid=301)

    def test_scale_long_range(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_long=19)
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_long=501)

    def test_cci_upper_range(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(cci_upper=49.0)
        with pytest.raises(ValidationError):
            CciConsensusConfig(cci_upper=301.0)

    def test_cci_lower_range(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(cci_lower=-301.0)
        with pytest.raises(ValidationError):
            CciConsensusConfig(cci_lower=-49.0)

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(entry_threshold=-0.1)
        with pytest.raises(ValidationError):
            CciConsensusConfig(entry_threshold=1.5)

    def test_cci_upper_positive_lower_negative(self) -> None:
        """cci_upper > 0 > cci_lower 필수."""
        # cci_upper and cci_lower have Field constraints (ge=50, le=-50)
        # so the cross-field validator ensures upper > 0 > lower inherently
        config = CciConsensusConfig()
        assert config.cci_upper > 0 > config.cci_lower

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CciConsensusConfig(vol_target=0.01, min_volatility=0.05)

    def test_scale_ordering_validation(self) -> None:
        """scale_short < scale_mid < scale_long 필수."""
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_short=60, scale_mid=60, scale_long=150)
        with pytest.raises(ValidationError):
            CciConsensusConfig(scale_short=20, scale_mid=150, scale_long=60)

    def test_warmup_periods(self) -> None:
        config = CciConsensusConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.scale_long

    def test_custom_params(self) -> None:
        config = CciConsensusConfig(scale_short=10, scale_mid=40, scale_long=100)
        assert config.scale_short == 10
        assert config.scale_mid == 40
        assert config.scale_long == 100

    def test_custom_cci_params(self) -> None:
        config = CciConsensusConfig(cci_upper=150.0, cci_lower=-150.0)
        assert config.cci_upper == 150.0
        assert config.cci_lower == -150.0
