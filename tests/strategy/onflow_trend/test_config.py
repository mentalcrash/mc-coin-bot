"""Tests for OnFlow Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.onflow_trend.config import OnflowTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestOnflowTrendConfig:
    def test_default_values(self) -> None:
        config = OnflowTrendConfig()
        assert config.flow_zscore_window == 90
        assert config.flow_long_z == -1.0
        assert config.flow_exit_z == 1.0
        assert config.mvrv_undervalued == 1.0
        assert config.mvrv_overheated == 3.5
        assert config.trend_ema_fast == 12
        assert config.trend_ema_slow == 36
        assert config.vol_target == 0.30
        assert config.annualization_factor == 1095.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = OnflowTrendConfig()
        with pytest.raises(ValidationError):
            config.flow_zscore_window = 999  # type: ignore[misc]

    def test_flow_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            OnflowTrendConfig(flow_zscore_window=10)
        with pytest.raises(ValidationError):
            OnflowTrendConfig(flow_zscore_window=400)

    def test_flow_long_z_range(self) -> None:
        with pytest.raises(ValidationError):
            OnflowTrendConfig(flow_long_z=1.0)
        with pytest.raises(ValidationError):
            OnflowTrendConfig(flow_long_z=-4.0)

    def test_flow_exit_z_range(self) -> None:
        with pytest.raises(ValidationError):
            OnflowTrendConfig(flow_exit_z=-1.0)
        with pytest.raises(ValidationError):
            OnflowTrendConfig(flow_exit_z=4.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            OnflowTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_mvrv_undervalued_lt_overheated(self) -> None:
        with pytest.raises(ValidationError):
            OnflowTrendConfig(mvrv_undervalued=4.0, mvrv_overheated=3.5)
        with pytest.raises(ValidationError):
            OnflowTrendConfig(mvrv_undervalued=3.5, mvrv_overheated=3.5)

    def test_trend_ema_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            OnflowTrendConfig(trend_ema_fast=50, trend_ema_slow=20)
        with pytest.raises(ValidationError):
            OnflowTrendConfig(trend_ema_fast=20, trend_ema_slow=20)

    def test_warmup_periods(self) -> None:
        config = OnflowTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.flow_zscore_window
        assert config.warmup_periods() >= config.trend_ema_slow

    def test_annualization_factor(self) -> None:
        config = OnflowTrendConfig()
        assert config.annualization_factor == 1095.0

    def test_custom_params(self) -> None:
        config = OnflowTrendConfig(flow_zscore_window=60, mvrv_undervalued=0.8)
        assert config.flow_zscore_window == 60
        assert config.mvrv_undervalued == 0.8
