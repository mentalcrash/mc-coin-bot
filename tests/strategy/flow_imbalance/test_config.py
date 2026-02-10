"""Tests for FlowImbalanceConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.flow_imbalance.config import FlowImbalanceConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = FlowImbalanceConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY


class TestFlowImbalanceConfig:
    """FlowImbalanceConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = FlowImbalanceConfig()

        assert config.ofi_window == 6
        assert config.ofi_entry_threshold == 0.6
        assert config.ofi_exit_threshold == 0.2
        assert config.vpin_window == 24
        assert config.vpin_threshold == 0.15
        assert config.timeout_bars == 24
        assert config.vol_target == 0.35
        assert config.annualization_factor == 8760.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = FlowImbalanceConfig()

        with pytest.raises(ValidationError):
            config.ofi_window = 12  # type: ignore[misc]

    def test_ofi_window_range(self):
        """ofi_window 범위 검증."""
        config = FlowImbalanceConfig(ofi_window=2)
        assert config.ofi_window == 2

        with pytest.raises(ValidationError):
            FlowImbalanceConfig(ofi_window=1)

        with pytest.raises(ValidationError):
            FlowImbalanceConfig(ofi_window=49)

    def test_ofi_exit_must_be_less_than_entry(self):
        """ofi_exit_threshold < ofi_entry_threshold 검증."""
        with pytest.raises(ValidationError):
            FlowImbalanceConfig(ofi_exit_threshold=0.6, ofi_entry_threshold=0.6)

        with pytest.raises(ValidationError):
            FlowImbalanceConfig(ofi_exit_threshold=0.7, ofi_entry_threshold=0.6)

    def test_vpin_window_range(self):
        """vpin_window 범위 검증."""
        config = FlowImbalanceConfig(vpin_window=6)
        assert config.vpin_window == 6

        with pytest.raises(ValidationError):
            FlowImbalanceConfig(vpin_window=5)

    def test_timeout_bars_range(self):
        """timeout_bars 범위 검증."""
        config = FlowImbalanceConfig(timeout_bars=1)
        assert config.timeout_bars == 1

        with pytest.raises(ValidationError):
            FlowImbalanceConfig(timeout_bars=0)

        with pytest.raises(ValidationError):
            FlowImbalanceConfig(timeout_bars=169)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        with pytest.raises(ValidationError):
            FlowImbalanceConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = FlowImbalanceConfig()
        # max(24, 6, 14) + 1 = 25
        assert config.warmup_periods() == 25

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = FlowImbalanceConfig(
            vpin_window=48,
            ofi_window=12,
            atr_period=50,
        )
        # max(48, 12, 50) + 1 = 51
        assert config.warmup_periods() == 51
