"""Tests for Basis-Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.basis_momentum.config import BasisMomentumConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestBasisMomentumConfig:
    def test_default_values(self) -> None:
        config = BasisMomentumConfig()
        assert config.fr_change_window == 6
        assert config.fr_std_window == 30
        assert config.entry_zscore == 1.5
        assert config.exit_zscore == 0.5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen(self) -> None:
        config = BasisMomentumConfig()
        with pytest.raises(ValidationError):
            config.fr_change_window = 999  # type: ignore[misc]

    def test_exit_zscore_lt_entry_zscore(self) -> None:
        """exit_zscore < entry_zscore 필수."""
        with pytest.raises(ValidationError):
            BasisMomentumConfig(entry_zscore=1.5, exit_zscore=1.5)
        with pytest.raises(ValidationError):
            BasisMomentumConfig(entry_zscore=1.0, exit_zscore=1.5)

    def test_exit_zscore_valid(self) -> None:
        config = BasisMomentumConfig(entry_zscore=2.0, exit_zscore=0.8)
        assert config.exit_zscore < config.entry_zscore

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            BasisMomentumConfig(vol_target=0.01, min_volatility=0.05)

    def test_vol_target_equal_min_volatility(self) -> None:
        config = BasisMomentumConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility

    def test_fr_change_window_range(self) -> None:
        with pytest.raises(ValidationError):
            BasisMomentumConfig(fr_change_window=1)
        with pytest.raises(ValidationError):
            BasisMomentumConfig(fr_change_window=51)

    def test_fr_std_window_range(self) -> None:
        with pytest.raises(ValidationError):
            BasisMomentumConfig(fr_std_window=4)
        with pytest.raises(ValidationError):
            BasisMomentumConfig(fr_std_window=201)

    def test_entry_zscore_range(self) -> None:
        with pytest.raises(ValidationError):
            BasisMomentumConfig(entry_zscore=0.4, exit_zscore=0.1)
        with pytest.raises(ValidationError):
            BasisMomentumConfig(entry_zscore=5.1)

    def test_warmup_periods(self) -> None:
        config = BasisMomentumConfig()
        expected = max(config.fr_change_window + config.fr_std_window, config.vol_window) + 10
        assert config.warmup_periods() == expected

    def test_warmup_periods_custom(self) -> None:
        config = BasisMomentumConfig(fr_change_window=20, fr_std_window=50)
        assert config.warmup_periods() >= config.fr_change_window + config.fr_std_window

    def test_custom_params(self) -> None:
        config = BasisMomentumConfig(fr_change_window=10, entry_zscore=2.0, exit_zscore=0.8)
        assert config.fr_change_window == 10
        assert config.entry_zscore == 2.0
