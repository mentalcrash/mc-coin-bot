"""Tests for KalmanTrendConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.kalman_trend.config import KalmanTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self) -> None:
        config = KalmanTrendConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestKalmanTrendConfig:
    def test_default_values(self) -> None:
        config = KalmanTrendConfig()

        assert config.base_q == 0.01
        assert config.observation_noise == 1.0
        assert config.vel_threshold == 0.5
        assert config.vol_lookback == 20
        assert config.long_term_vol_lookback == 120
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self) -> None:
        config = KalmanTrendConfig()
        with pytest.raises(ValidationError):
            config.base_q = 0.05  # type: ignore[misc]

    def test_base_q_range(self) -> None:
        config = KalmanTrendConfig(base_q=0.001)
        assert config.base_q == 0.001

        config = KalmanTrendConfig(base_q=1.0)
        assert config.base_q == 1.0

        with pytest.raises(ValidationError):
            KalmanTrendConfig(base_q=0.0005)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(base_q=1.1)

    def test_observation_noise_range(self) -> None:
        config = KalmanTrendConfig(observation_noise=0.01)
        assert config.observation_noise == 0.01

        config = KalmanTrendConfig(observation_noise=100.0)
        assert config.observation_noise == 100.0

        with pytest.raises(ValidationError):
            KalmanTrendConfig(observation_noise=0.005)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(observation_noise=100.1)

    def test_vel_threshold_range(self) -> None:
        config = KalmanTrendConfig(vel_threshold=0.01)
        assert config.vel_threshold == 0.01

        config = KalmanTrendConfig(vel_threshold=5.0)
        assert config.vel_threshold == 5.0

        with pytest.raises(ValidationError):
            KalmanTrendConfig(vel_threshold=0.005)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(vel_threshold=5.1)

    def test_vol_lookback_range(self) -> None:
        config = KalmanTrendConfig(vol_lookback=5)
        assert config.vol_lookback == 5

        config = KalmanTrendConfig(vol_lookback=100, long_term_vol_lookback=200)
        assert config.vol_lookback == 100

        with pytest.raises(ValidationError):
            KalmanTrendConfig(vol_lookback=4)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(vol_lookback=101)

    def test_long_term_vol_lookback_range(self) -> None:
        config = KalmanTrendConfig(long_term_vol_lookback=40, vol_lookback=5)
        assert config.long_term_vol_lookback == 40

        config = KalmanTrendConfig(long_term_vol_lookback=500)
        assert config.long_term_vol_lookback == 500

        with pytest.raises(ValidationError):
            KalmanTrendConfig(long_term_vol_lookback=39)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(long_term_vol_lookback=501)

    def test_vol_lookback_lt_long_term(self) -> None:
        """vol_lookback < long_term_vol_lookback 검증."""
        config = KalmanTrendConfig(vol_lookback=20, long_term_vol_lookback=120)
        assert config.vol_lookback < config.long_term_vol_lookback

        # vol_lookback == long_term_vol_lookback → 실패
        with pytest.raises(ValidationError):
            KalmanTrendConfig(vol_lookback=40, long_term_vol_lookback=40)

        # vol_lookback > long_term_vol_lookback → 실패
        with pytest.raises(ValidationError):
            KalmanTrendConfig(vol_lookback=50, long_term_vol_lookback=40)

    def test_mom_lookback_range(self) -> None:
        config = KalmanTrendConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = KalmanTrendConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            KalmanTrendConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(mom_lookback=61)

    def test_vol_target_gte_min_volatility(self) -> None:
        config = KalmanTrendConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            KalmanTrendConfig(vol_target=0.03, min_volatility=0.05)

    def test_atr_period_range(self) -> None:
        config = KalmanTrendConfig(atr_period=5)
        assert config.atr_period == 5

        config = KalmanTrendConfig(atr_period=50)
        assert config.atr_period == 50

        with pytest.raises(ValidationError):
            KalmanTrendConfig(atr_period=4)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(atr_period=51)

    def test_hedge_threshold_range(self) -> None:
        config = KalmanTrendConfig(hedge_threshold=-0.05)
        assert config.hedge_threshold == -0.05

        config = KalmanTrendConfig(hedge_threshold=-0.30)
        assert config.hedge_threshold == -0.30

        with pytest.raises(ValidationError):
            KalmanTrendConfig(hedge_threshold=-0.31)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(hedge_threshold=-0.04)

    def test_hedge_strength_ratio_range(self) -> None:
        config = KalmanTrendConfig(hedge_strength_ratio=0.1)
        assert config.hedge_strength_ratio == 0.1

        config = KalmanTrendConfig(hedge_strength_ratio=1.0)
        assert config.hedge_strength_ratio == 1.0

        with pytest.raises(ValidationError):
            KalmanTrendConfig(hedge_strength_ratio=0.05)

        with pytest.raises(ValidationError):
            KalmanTrendConfig(hedge_strength_ratio=1.1)

    def test_warmup_periods(self) -> None:
        config = KalmanTrendConfig(
            long_term_vol_lookback=120,
            mom_lookback=20,
            atr_period=14,
        )
        # max(120, 20, 14) + 1 = 121
        assert config.warmup_periods() == 121

    def test_warmup_periods_custom(self) -> None:
        config = KalmanTrendConfig(
            long_term_vol_lookback=200,
            mom_lookback=60,
            atr_period=50,
            vol_lookback=50,
        )
        # max(200, 60, 50) + 1 = 201
        assert config.warmup_periods() == 201

    def test_warmup_periods_mom_dominant(self) -> None:
        """mom_lookback가 가장 클 때 warmup 확인."""
        config = KalmanTrendConfig(
            long_term_vol_lookback=40,
            mom_lookback=60,
            atr_period=14,
            vol_lookback=5,
        )
        # max(40, 60, 14) + 1 = 61
        assert config.warmup_periods() == 61
