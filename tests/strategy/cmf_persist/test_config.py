"""Tests for CMF Trend Persistence config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cmf_persist.config import CmfPersistConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCmfPersistConfig:
    def test_default_values(self) -> None:
        config = CmfPersistConfig()
        assert config.cmf_period == 20
        assert config.persist_window == 10
        assert config.persist_threshold == 0.7
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = CmfPersistConfig()
        with pytest.raises(ValidationError):
            config.cmf_period = 999  # type: ignore[misc]

    def test_cmf_period_range(self) -> None:
        with pytest.raises(ValidationError):
            CmfPersistConfig(cmf_period=4)
        with pytest.raises(ValidationError):
            CmfPersistConfig(cmf_period=61)

    def test_persist_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CmfPersistConfig(persist_threshold=0.4)
        with pytest.raises(ValidationError):
            CmfPersistConfig(persist_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CmfPersistConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CmfPersistConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.cmf_period + config.persist_window

    def test_annualization_factor(self) -> None:
        config = CmfPersistConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = CmfPersistConfig(cmf_period=15, persist_window=8)
        assert config.cmf_period == 15
        assert config.persist_window == 8
