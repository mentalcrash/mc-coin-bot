"""Tests for VPINFlowConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vpin_flow.config import ShortMode, VPINFlowConfig


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = VPINFlowConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL


class TestVPINFlowConfig:
    def test_default_values(self):
        config = VPINFlowConfig()

        assert config.n_buckets == 50
        assert config.threshold_high == 0.7
        assert config.threshold_low == 0.3
        assert config.flow_direction_period == 20
        assert config.vol_target == 0.30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = VPINFlowConfig()
        with pytest.raises(ValidationError):
            config.n_buckets = 100  # type: ignore[misc]

    def test_n_buckets_range(self):
        config = VPINFlowConfig(n_buckets=10)
        assert config.n_buckets == 10

        config = VPINFlowConfig(n_buckets=200)
        assert config.n_buckets == 200

        with pytest.raises(ValidationError):
            VPINFlowConfig(n_buckets=9)

        with pytest.raises(ValidationError):
            VPINFlowConfig(n_buckets=201)

    def test_threshold_high_range(self):
        config = VPINFlowConfig(threshold_high=0.3, threshold_low=0.1)
        assert config.threshold_high == 0.3

        config = VPINFlowConfig(threshold_high=0.95)
        assert config.threshold_high == 0.95

        with pytest.raises(ValidationError):
            VPINFlowConfig(threshold_high=0.2)

        with pytest.raises(ValidationError):
            VPINFlowConfig(threshold_high=0.96)

    def test_threshold_low_range(self):
        config = VPINFlowConfig(threshold_low=0.05)
        assert config.threshold_low == 0.05

        config = VPINFlowConfig(threshold_low=0.5, threshold_high=0.8)
        assert config.threshold_low == 0.5

        with pytest.raises(ValidationError):
            VPINFlowConfig(threshold_low=0.04)

        with pytest.raises(ValidationError):
            VPINFlowConfig(threshold_low=0.51)

    def test_threshold_high_gt_low(self):
        """threshold_high > threshold_low 검증."""
        config = VPINFlowConfig(threshold_high=0.8, threshold_low=0.2)
        assert config.threshold_high > config.threshold_low

        with pytest.raises(ValidationError):
            VPINFlowConfig(threshold_high=0.3, threshold_low=0.3)

        with pytest.raises(ValidationError):
            VPINFlowConfig(threshold_high=0.3, threshold_low=0.4)

    def test_flow_direction_period_range(self):
        config = VPINFlowConfig(flow_direction_period=5)
        assert config.flow_direction_period == 5

        config = VPINFlowConfig(flow_direction_period=100)
        assert config.flow_direction_period == 100

        with pytest.raises(ValidationError):
            VPINFlowConfig(flow_direction_period=4)

        with pytest.raises(ValidationError):
            VPINFlowConfig(flow_direction_period=101)

    def test_vol_target_gte_min_volatility(self):
        config = VPINFlowConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            VPINFlowConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = VPINFlowConfig(
            n_buckets=50,
            flow_direction_period=20,
            atr_period=14,
        )
        # max(50, 20, 14) + 1 = 51
        assert config.warmup_periods() == 51

    def test_warmup_periods_custom(self):
        config = VPINFlowConfig(
            n_buckets=100,
            flow_direction_period=80,
            atr_period=50,
        )
        # max(100, 80, 50) + 1 = 101
        assert config.warmup_periods() == 101
