"""Tests for ExFlowMom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ex_flow_mom.config import ExFlowMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestExFlowMomConfig:
    def test_default_values(self) -> None:
        config = ExFlowMomConfig()
        assert config.flow_window == 14
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = ExFlowMomConfig()
        with pytest.raises(ValidationError):
            config.flow_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ExFlowMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = ExFlowMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = ExFlowMomConfig(flow_window=20)
        assert config.flow_window == 20
