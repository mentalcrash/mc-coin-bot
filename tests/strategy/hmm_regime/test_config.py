"""Tests for HMMRegimeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.hmm_regime.config import HMMRegimeConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum tests."""

    def test_values(self) -> None:
        """ShortMode value verification."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self) -> None:
        """Config accepts all ShortMode values."""
        for mode in ShortMode:
            config = HMMRegimeConfig(short_mode=mode)
            assert config.short_mode == mode


class TestHMMRegimeConfig:
    """HMMRegimeConfig tests."""

    def test_default_values(self) -> None:
        """Test creation with all default values."""
        config = HMMRegimeConfig()

        assert config.n_states == 3
        assert config.min_train_window == 252
        assert config.retrain_interval == 21
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.n_iter == 100
        assert config.short_mode == ShortMode.DISABLED
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self) -> None:
        """Frozen model cannot be mutated."""
        config = HMMRegimeConfig()

        with pytest.raises(ValidationError):
            config.n_states = 5  # type: ignore[misc]

    def test_n_states_range(self) -> None:
        """n_states range validation (2-5)."""
        config = HMMRegimeConfig(n_states=2)
        assert config.n_states == 2

        config = HMMRegimeConfig(n_states=5)
        assert config.n_states == 5

        with pytest.raises(ValidationError):
            HMMRegimeConfig(n_states=1)

        with pytest.raises(ValidationError):
            HMMRegimeConfig(n_states=6)

    def test_min_train_window_range(self) -> None:
        """min_train_window range validation (100-500)."""
        config = HMMRegimeConfig(min_train_window=100)
        assert config.min_train_window == 100

        config = HMMRegimeConfig(min_train_window=500)
        assert config.min_train_window == 500

        with pytest.raises(ValidationError):
            HMMRegimeConfig(min_train_window=99)

        with pytest.raises(ValidationError):
            HMMRegimeConfig(min_train_window=501)

    def test_retrain_interval_range(self) -> None:
        """retrain_interval range validation (1-63)."""
        config = HMMRegimeConfig(retrain_interval=1)
        assert config.retrain_interval == 1

        config = HMMRegimeConfig(retrain_interval=63)
        assert config.retrain_interval == 63

        with pytest.raises(ValidationError):
            HMMRegimeConfig(retrain_interval=0)

        with pytest.raises(ValidationError):
            HMMRegimeConfig(retrain_interval=64)

    def test_n_iter_range(self) -> None:
        """n_iter range validation (50-500)."""
        config = HMMRegimeConfig(n_iter=50)
        assert config.n_iter == 50

        config = HMMRegimeConfig(n_iter=500)
        assert config.n_iter == 500

        with pytest.raises(ValidationError):
            HMMRegimeConfig(n_iter=49)

        with pytest.raises(ValidationError):
            HMMRegimeConfig(n_iter=501)

    def test_vol_target_gte_min_volatility(self) -> None:
        """vol_target must be >= min_volatility."""
        # Valid case
        config = HMMRegimeConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # Equal is OK
        config = HMMRegimeConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility is error
        with pytest.raises(ValidationError, match="vol_target"):
            HMMRegimeConfig(vol_target=0.05, min_volatility=0.10)

    def test_min_train_gte_vol_window(self) -> None:
        """min_train_window must be >= vol_window.

        Note: With current field constraints (min_train_window ge=100, vol_window le=60),
        this validator cannot be violated through normal field values alone.
        We verify the valid boundary case instead.
        """
        # Valid case: min_train_window=100, vol_window=60 (boundary)
        config = HMMRegimeConfig(min_train_window=100, vol_window=60)
        assert config.min_train_window >= config.vol_window

        # Equal is OK
        config = HMMRegimeConfig(min_train_window=100, vol_window=5)
        assert config.min_train_window >= config.vol_window

    def test_warmup_periods(self) -> None:
        """warmup_periods() test."""
        config = HMMRegimeConfig(min_train_window=252)
        # min_train_window + 1 = 253
        assert config.warmup_periods() == 253

        config = HMMRegimeConfig(min_train_window=100)
        assert config.warmup_periods() == 101
