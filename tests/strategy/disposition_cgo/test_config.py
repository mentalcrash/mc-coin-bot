"""Tests for Disposition CGO config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.disposition_cgo.config import DispositionCgoConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestDispositionCgoConfig:
    def test_default_values(self) -> None:
        config = DispositionCgoConfig()
        assert config.turnover_window == 60
        assert config.cgo_smooth_window == 10
        assert config.momentum_window == 20
        assert config.overhang_spread_window == 60
        assert config.cgo_entry_threshold == 0.03
        assert config.spread_confirm_threshold == 0.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = DispositionCgoConfig()
        with pytest.raises(ValidationError):
            config.turnover_window = 999  # type: ignore[misc]

    def test_turnover_window_range(self) -> None:
        with pytest.raises(ValidationError):
            DispositionCgoConfig(turnover_window=5)
        with pytest.raises(ValidationError):
            DispositionCgoConfig(turnover_window=400)

    def test_cgo_smooth_window_range(self) -> None:
        with pytest.raises(ValidationError):
            DispositionCgoConfig(cgo_smooth_window=1)
        with pytest.raises(ValidationError):
            DispositionCgoConfig(cgo_smooth_window=200)

    def test_momentum_window_range(self) -> None:
        with pytest.raises(ValidationError):
            DispositionCgoConfig(momentum_window=2)
        with pytest.raises(ValidationError):
            DispositionCgoConfig(momentum_window=300)

    def test_cgo_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            DispositionCgoConfig(cgo_entry_threshold=-0.1)
        with pytest.raises(ValidationError):
            DispositionCgoConfig(cgo_entry_threshold=1.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DispositionCgoConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = DispositionCgoConfig()
        warmup = config.warmup_periods()
        assert warmup >= config.vol_window
        assert warmup >= config.turnover_window
        assert warmup >= config.overhang_spread_window

    def test_annualization_factor(self) -> None:
        config = DispositionCgoConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = DispositionCgoConfig(turnover_window=90, cgo_entry_threshold=0.05)
        assert config.turnover_window == 90
        assert config.cgo_entry_threshold == 0.05

    def test_hedge_params(self) -> None:
        config = DispositionCgoConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5

    def test_spread_confirm_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            DispositionCgoConfig(spread_confirm_threshold=-2.0)
        with pytest.raises(ValidationError):
            DispositionCgoConfig(spread_confirm_threshold=2.0)
