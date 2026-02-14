"""Tests for Multi-Horizon ROC Ensemble config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.mh_roc.config import MhRocConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMhRocConfig:
    def test_default_values(self) -> None:
        config = MhRocConfig()
        assert config.roc_short == 6
        assert config.roc_medium_short == 18
        assert config.roc_medium_long == 42
        assert config.roc_long == 90
        assert config.vote_threshold == 3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = MhRocConfig()
        with pytest.raises(ValidationError):
            config.roc_short = 999  # type: ignore[misc]

    def test_roc_short_range(self) -> None:
        with pytest.raises(ValidationError):
            MhRocConfig(roc_short=1)
        with pytest.raises(ValidationError):
            MhRocConfig(roc_short=51)

    def test_roc_long_range(self) -> None:
        with pytest.raises(ValidationError):
            MhRocConfig(roc_long=19)
        with pytest.raises(ValidationError):
            MhRocConfig(roc_long=366)

    def test_vote_threshold_range(self) -> None:
        # Valid: 1~4
        config = MhRocConfig(vote_threshold=1)
        assert config.vote_threshold == 1
        config = MhRocConfig(vote_threshold=4)
        assert config.vote_threshold == 4
        with pytest.raises(ValidationError):
            MhRocConfig(vote_threshold=0)
        with pytest.raises(ValidationError):
            MhRocConfig(vote_threshold=5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MhRocConfig(vol_target=0.01, min_volatility=0.05)

    def test_roc_lookbacks_must_be_increasing(self) -> None:
        """ROC lookback은 strictly increasing이어야 함."""
        with pytest.raises(ValidationError, match="strictly increasing"):
            MhRocConfig(roc_short=18, roc_medium_short=18)
        with pytest.raises(ValidationError, match="strictly increasing"):
            MhRocConfig(roc_medium_short=50, roc_medium_long=42)

    def test_warmup_periods(self) -> None:
        config = MhRocConfig()
        assert config.warmup_periods() >= config.roc_long
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = MhRocConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = MhRocConfig(roc_short=8, roc_medium_short=24, vote_threshold=2)
        assert config.roc_short == 8
        assert config.roc_medium_short == 24
        assert config.vote_threshold == 2

    def test_hedge_params(self) -> None:
        config = MhRocConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
