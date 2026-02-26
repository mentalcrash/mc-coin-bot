"""Tests for Donchian Filtered config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.donch_filtered.config import DonchFilteredConfig
from src.strategy.donch_multi.config import ShortMode


class TestDonchFilteredConfig:
    def test_default_values(self) -> None:
        config = DonchFilteredConfig()
        assert config.lookback_short == 20
        assert config.lookback_mid == 40
        assert config.lookback_long == 80
        assert config.entry_threshold == 0.34
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_fr_default_values(self) -> None:
        config = DonchFilteredConfig()
        assert config.fr_ma_window == 3
        assert config.fr_zscore_window == 90
        assert config.fr_suppress_threshold == 1.5

    def test_frozen(self) -> None:
        config = DonchFilteredConfig()
        with pytest.raises(ValidationError):
            config.lookback_short = 999  # type: ignore[misc]
        with pytest.raises(ValidationError):
            config.fr_suppress_threshold = 999.0  # type: ignore[misc]

    def test_lookback_ordering(self) -> None:
        with pytest.raises(ValidationError):
            DonchFilteredConfig(lookback_short=40, lookback_mid=40, lookback_long=80)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(lookback_short=80, lookback_mid=40, lookback_long=20)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            DonchFilteredConfig(vol_target=0.01, min_volatility=0.05)

    def test_fr_ma_window_range(self) -> None:
        DonchFilteredConfig(fr_ma_window=1)
        DonchFilteredConfig(fr_ma_window=10)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(fr_ma_window=0)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(fr_ma_window=11)

    def test_fr_zscore_window_range(self) -> None:
        DonchFilteredConfig(fr_zscore_window=30)
        DonchFilteredConfig(fr_zscore_window=180)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(fr_zscore_window=29)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(fr_zscore_window=181)

    def test_fr_suppress_threshold_range(self) -> None:
        DonchFilteredConfig(fr_suppress_threshold=0.5)
        DonchFilteredConfig(fr_suppress_threshold=3.0)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(fr_suppress_threshold=0.4)
        with pytest.raises(ValidationError):
            DonchFilteredConfig(fr_suppress_threshold=3.1)

    def test_warmup_periods(self) -> None:
        config = DonchFilteredConfig()
        wp = config.warmup_periods()
        assert wp >= config.lookback_long
        assert wp >= config.vol_window
        assert wp >= config.fr_zscore_window

    def test_warmup_periods_dominated_by_fr_window(self) -> None:
        """fr_zscore_window(90) > lookback_long(80) > vol_window(30)."""
        config = DonchFilteredConfig()
        assert config.warmup_periods() == config.fr_zscore_window + 10

    def test_custom_params(self) -> None:
        config = DonchFilteredConfig(
            lookback_short=10,
            lookback_mid=30,
            lookback_long=60,
            fr_suppress_threshold=2.0,
        )
        assert config.lookback_short == 10
        assert config.fr_suppress_threshold == 2.0

    def test_hedge_params(self) -> None:
        config = DonchFilteredConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
