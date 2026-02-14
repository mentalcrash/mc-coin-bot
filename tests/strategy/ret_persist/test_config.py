"""Tests for Return Persistence Score config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ret_persist.config import RetPersistConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestRetPersistConfig:
    def test_default_values(self) -> None:
        config = RetPersistConfig()
        assert config.persist_window == 30
        assert config.long_threshold == 0.6
        assert config.short_threshold == 0.4
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = RetPersistConfig()
        with pytest.raises(ValidationError):
            config.persist_window = 999  # type: ignore[misc]

    def test_persist_window_range(self) -> None:
        with pytest.raises(ValidationError):
            RetPersistConfig(persist_window=4)
        with pytest.raises(ValidationError):
            RetPersistConfig(persist_window=121)

    def test_long_must_exceed_short_threshold(self) -> None:
        with pytest.raises(ValidationError):
            RetPersistConfig(long_threshold=0.5, short_threshold=0.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            RetPersistConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = RetPersistConfig()
        assert config.warmup_periods() >= config.persist_window

    def test_custom_params(self) -> None:
        config = RetPersistConfig(persist_window=20, long_threshold=0.65)
        assert config.persist_window == 20
        assert config.long_threshold == 0.65
