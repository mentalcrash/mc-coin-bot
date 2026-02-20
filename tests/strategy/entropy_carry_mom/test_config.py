"""Tests for Entropy-Carry-Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.entropy_carry_mom.config import EntropyCarryMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestEntropyCarryMomConfig:
    def test_default_values(self) -> None:
        config = EntropyCarryMomConfig()
        assert config.entropy_window == 60
        assert config.entropy_bins == 10
        assert config.entropy_low_pct == 30.0
        assert config.entropy_high_pct == 70.0
        assert config.entropy_rank_window == 120
        assert config.mom_lookback == 20
        assert config.mom_strength_window == 10
        assert config.fr_lookback == 3
        assert config.fr_zscore_window == 90
        assert config.fr_entry_threshold == 0.0001
        assert config.mom_weight_low_entropy == 0.8
        assert config.carry_weight_high_entropy == 0.8
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = EntropyCarryMomConfig()
        with pytest.raises(ValidationError):
            config.entropy_window = 999  # type: ignore[misc]

    def test_entropy_window_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_window=19)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_window=251)

    def test_entropy_bins_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_bins=4)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_bins=31)

    def test_entropy_low_pct_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_low_pct=4.0)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_low_pct=51.0)

    def test_entropy_high_pct_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_high_pct=49.0)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_high_pct=96.0)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(mom_lookback=61)

    def test_fr_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(fr_lookback=0)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(fr_lookback=31)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_entropy_low_lt_high(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(entropy_low_pct=50.0, entropy_high_pct=50.0)

    def test_warmup_periods(self) -> None:
        config = EntropyCarryMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.fr_zscore_window
        assert config.warmup_periods() >= config.entropy_window

    def test_annualization_factor(self) -> None:
        config = EntropyCarryMomConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = EntropyCarryMomConfig(entropy_window=40, fr_lookback=5)
        assert config.entropy_window == 40
        assert config.fr_lookback == 5

    def test_mom_weight_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(mom_weight_low_entropy=0.4)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(mom_weight_low_entropy=1.1)

    def test_carry_weight_range(self) -> None:
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(carry_weight_high_entropy=0.4)
        with pytest.raises(ValidationError):
            EntropyCarryMomConfig(carry_weight_high_entropy=1.1)
