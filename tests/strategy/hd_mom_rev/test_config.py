"""Tests for hd-mom-rev config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.hd_mom_rev.config import HdMomRevConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestHdMomRevConfig:
    def test_default_values(self) -> None:
        config = HdMomRevConfig()
        assert config.jump_threshold == 2.0
        assert config.half_return_ma == 3
        assert config.confidence_cap == 1.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = HdMomRevConfig()
        with pytest.raises(ValidationError):
            config.jump_threshold = 999.0  # type: ignore[misc]

    def test_jump_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            HdMomRevConfig(jump_threshold=0.4)
        with pytest.raises(ValidationError):
            HdMomRevConfig(jump_threshold=5.1)

    def test_half_return_ma_range(self) -> None:
        with pytest.raises(ValidationError):
            HdMomRevConfig(half_return_ma=0)
        with pytest.raises(ValidationError):
            HdMomRevConfig(half_return_ma=21)

    def test_confidence_cap_range(self) -> None:
        with pytest.raises(ValidationError):
            HdMomRevConfig(confidence_cap=0.0)
        with pytest.raises(ValidationError):
            HdMomRevConfig(confidence_cap=2.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            HdMomRevConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = HdMomRevConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_warmup_periods_covers_half_return_ma(self) -> None:
        config = HdMomRevConfig()
        assert config.warmup_periods() >= config.half_return_ma

    def test_annualization_factor(self) -> None:
        config = HdMomRevConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = HdMomRevConfig(jump_threshold=1.5, confidence_cap=0.8)
        assert config.jump_threshold == 1.5
        assert config.confidence_cap == 0.8

    def test_hedge_params(self) -> None:
        config = HdMomRevConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
