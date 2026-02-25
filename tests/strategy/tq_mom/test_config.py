"""Tests for Trend Quality Momentum (TQ-Mom) config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.tq_mom.config import ShortMode, TqMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestTqMomConfig:
    def test_default_values(self) -> None:
        config = TqMomConfig()
        assert config.hurst_window == 40
        assert config.hurst_threshold == 0.55
        assert config.fd_period == 20
        assert config.fd_threshold == 1.4
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = TqMomConfig()
        with pytest.raises(ValidationError):
            config.hurst_window = 999  # type: ignore[misc]

    def test_hurst_window_range(self) -> None:
        with pytest.raises(ValidationError):
            TqMomConfig(hurst_window=5)
        with pytest.raises(ValidationError):
            TqMomConfig(hurst_window=300)

    def test_hurst_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            TqMomConfig(hurst_threshold=0.4)
        with pytest.raises(ValidationError):
            TqMomConfig(hurst_threshold=1.0)

    def test_fd_period_range(self) -> None:
        with pytest.raises(ValidationError):
            TqMomConfig(fd_period=5)
        with pytest.raises(ValidationError):
            TqMomConfig(fd_period=200)

    def test_fd_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            TqMomConfig(fd_threshold=0.5)
        with pytest.raises(ValidationError):
            TqMomConfig(fd_threshold=2.0)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            TqMomConfig(mom_lookback=1)
        with pytest.raises(ValidationError):
            TqMomConfig(mom_lookback=200)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            TqMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = TqMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.hurst_window
        # fd uses 2*fd_period internally
        assert config.warmup_periods() >= 2 * config.fd_period

    def test_annualization_factor(self) -> None:
        config = TqMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = TqMomConfig(hurst_window=60, fd_period=30)
        assert config.hurst_window == 60
        assert config.fd_period == 30

    def test_hedge_params(self) -> None:
        config = TqMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
