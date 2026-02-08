"""Tests for ADXRegimeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.adx_regime.config import ADXRegimeConfig
from src.strategy.tsmom.config import ShortMode


class TestADXRegimeConfig:
    """ADXRegimeConfig default values and validation tests."""

    def test_default_values(self) -> None:
        """기본값이 올바르게 설정되는지 확인."""
        config = ADXRegimeConfig()

        assert config.adx_period == 14
        assert config.adx_low == 15.0
        assert config.adx_high == 25.0
        assert config.mom_lookback == 30
        assert config.mr_lookback == 20
        assert config.mr_entry_z == 2.0
        assert config.mr_exit_z == 0.5
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen_model(self) -> None:
        """frozen 모델이므로 속성 수정 불가."""
        config = ADXRegimeConfig()

        with pytest.raises(ValidationError):
            config.adx_period = 20  # type: ignore[misc]

    def test_adx_thresholds_validation(self) -> None:
        """adx_low >= adx_high 이면 ValidationError."""
        with pytest.raises(ValidationError):
            ADXRegimeConfig(adx_low=25.0, adx_high=25.0)

        with pytest.raises(ValidationError):
            ADXRegimeConfig(adx_low=30.0, adx_high=25.0)

    def test_mr_z_validation(self) -> None:
        """mr_exit_z >= mr_entry_z 이면 ValidationError."""
        with pytest.raises(ValidationError):
            ADXRegimeConfig(mr_exit_z=2.0, mr_entry_z=2.0)

        with pytest.raises(ValidationError):
            ADXRegimeConfig(mr_exit_z=2.0, mr_entry_z=1.5)

    def test_vol_target_validation(self) -> None:
        """vol_target < min_volatility 이면 ValidationError."""
        with pytest.raises(ValidationError):
            ADXRegimeConfig(vol_target=0.04, min_volatility=0.05)

        # 같은 값은 OK
        config = ADXRegimeConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == 0.05

    def test_warmup_periods(self) -> None:
        """warmup_periods = max(adx_period*3, mom_lookback, mr_lookback, vol_window) + 1."""
        config = ADXRegimeConfig()
        # default: adx_period*3=42, mom_lookback=30, mr_lookback=20, vol_window=30
        # max = 42, + 1 = 43
        assert config.warmup_periods() == 43

        config2 = ADXRegimeConfig(adx_period=5, mom_lookback=100, mr_lookback=20, vol_window=30)
        # max(15, 100, 20, 30) = 100, + 1 = 101
        assert config2.warmup_periods() == 101

        config3 = ADXRegimeConfig(adx_period=5, mom_lookback=6, mr_lookback=50, vol_window=30)
        # max(15, 6, 50, 30) = 50, + 1 = 51
        assert config3.warmup_periods() == 51

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d')는 annualization_factor=365.0."""
        config = ADXRegimeConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h')는 annualization_factor=2190.0."""
        config = ADXRegimeConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe에 kwargs 오버라이드."""
        config = ADXRegimeConfig.for_timeframe("1h", vol_target=0.40)
        assert config.annualization_factor == 8760.0
        assert config.vol_target == 0.40

    def test_conservative_preset(self) -> None:
        """conservative() 프리셋 검증."""
        config = ADXRegimeConfig.conservative()

        assert config.adx_period == 20
        assert config.adx_low == 20.0
        assert config.adx_high == 35.0
        assert config.mom_lookback == 48
        assert config.mr_lookback == 30
        assert config.mr_entry_z == 2.5
        assert config.mr_exit_z == 0.8
        assert config.vol_target == 0.15
        assert config.min_volatility == 0.08

    def test_aggressive_preset(self) -> None:
        """aggressive() 프리셋 검증."""
        config = ADXRegimeConfig.aggressive()

        assert config.adx_period == 10
        assert config.adx_low == 12.0
        assert config.adx_high == 22.0
        assert config.mom_lookback == 14
        assert config.mr_lookback == 14
        assert config.mr_entry_z == 1.5
        assert config.mr_exit_z == 0.3
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.short_mode == ShortMode.FULL

    def test_adx_period_range(self) -> None:
        """adx_period 범위 검증 (5-50)."""
        ADXRegimeConfig(adx_period=5)
        ADXRegimeConfig(adx_period=50)

        with pytest.raises(ValidationError):
            ADXRegimeConfig(adx_period=4)

        with pytest.raises(ValidationError):
            ADXRegimeConfig(adx_period=51)

    def test_mom_lookback_range(self) -> None:
        """mom_lookback 범위 검증 (6-365)."""
        ADXRegimeConfig(mom_lookback=6)
        ADXRegimeConfig(mom_lookback=365)

        with pytest.raises(ValidationError):
            ADXRegimeConfig(mom_lookback=5)

        with pytest.raises(ValidationError):
            ADXRegimeConfig(mom_lookback=366)
