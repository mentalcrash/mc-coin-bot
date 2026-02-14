"""Tests for Asymmetric Volume Response config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.asym_vol_resp.config import AsymVolRespConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAsymVolRespConfig:
    def test_default_values(self) -> None:
        config = AsymVolRespConfig()
        assert config.impact_window == 12
        assert config.asym_window == 30
        assert config.asym_threshold == 1.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = AsymVolRespConfig()
        with pytest.raises(ValidationError):
            config.impact_window = 999  # type: ignore[misc]

    def test_impact_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AsymVolRespConfig(impact_window=1)
        with pytest.raises(ValidationError):
            AsymVolRespConfig(impact_window=100)

    def test_asym_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            AsymVolRespConfig(asym_threshold=0.1)
        with pytest.raises(ValidationError):
            AsymVolRespConfig(asym_threshold=5.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AsymVolRespConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AsymVolRespConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = AsymVolRespConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = AsymVolRespConfig(impact_window=24, asym_threshold=1.5)
        assert config.impact_window == 24
        assert config.asym_threshold == 1.5
