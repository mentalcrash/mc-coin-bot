"""Unit tests for DonchianEnsembleConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig, ShortMode


class TestConfigDefaults:
    """DonchianEnsembleConfig 기본값 테스트."""

    def test_default_lookbacks(self) -> None:
        """기본 lookbacks 튜플이 올바른지 확인."""
        config = DonchianEnsembleConfig()
        assert config.lookbacks == (5, 10, 20, 30, 60, 90, 150, 250, 360)

    def test_default_atr_period(self) -> None:
        """기본 atr_period가 14인지 확인."""
        config = DonchianEnsembleConfig()
        assert config.atr_period == 14

    def test_default_vol_target(self) -> None:
        """기본 vol_target이 0.40인지 확인."""
        config = DonchianEnsembleConfig()
        assert config.vol_target == 0.40

    def test_default_min_volatility(self) -> None:
        """기본 min_volatility가 0.05인지 확인."""
        config = DonchianEnsembleConfig()
        assert config.min_volatility == 0.05

    def test_default_annualization_factor(self) -> None:
        """기본 annualization_factor가 365.0인지 확인."""
        config = DonchianEnsembleConfig()
        assert config.annualization_factor == 365.0

    def test_default_short_mode(self) -> None:
        """기본 short_mode가 DISABLED인지 확인."""
        config = DonchianEnsembleConfig()
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self) -> None:
        """frozen=True로 인해 속성 변경이 불가한지 확인."""
        config = DonchianEnsembleConfig()
        with pytest.raises(ValidationError):
            config.vol_target = 0.50  # type: ignore[misc]


class TestConfigValidation:
    """DonchianEnsembleConfig 검증 테스트."""

    def test_vol_target_less_than_min_volatility_raises(self) -> None:
        """vol_target < min_volatility일 때 ValidationError 발생."""
        with pytest.raises(ValidationError, match="vol_target"):
            DonchianEnsembleConfig(vol_target=0.03, min_volatility=0.05)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility일 때 통과."""
        config = DonchianEnsembleConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility

    def test_lookback_below_2_raises(self) -> None:
        """lookback < 2일 때 ValidationError 발생."""
        with pytest.raises(ValidationError, match="lookbacks must be >= 2"):
            DonchianEnsembleConfig(lookbacks=(1, 10, 20))

    def test_all_lookbacks_valid(self) -> None:
        """모든 lookback >= 2이면 통과."""
        config = DonchianEnsembleConfig(lookbacks=(2, 5, 10))
        assert config.lookbacks == (2, 5, 10)

    def test_vol_target_range_too_low(self) -> None:
        """vol_target 범위 검증 (< 0.05)."""
        with pytest.raises(ValidationError):
            DonchianEnsembleConfig(vol_target=0.01)

    def test_vol_target_range_too_high(self) -> None:
        """vol_target 범위 검증 (> 1.0)."""
        with pytest.raises(ValidationError):
            DonchianEnsembleConfig(vol_target=1.5)

    def test_atr_period_range_too_low(self) -> None:
        """atr_period < 5일 때 ValidationError."""
        with pytest.raises(ValidationError):
            DonchianEnsembleConfig(atr_period=4)

    def test_atr_period_range_too_high(self) -> None:
        """atr_period > 50일 때 ValidationError."""
        with pytest.raises(ValidationError):
            DonchianEnsembleConfig(atr_period=51)


class TestConfigWarmup:
    """warmup_periods 계산 테스트."""

    def test_warmup_default(self) -> None:
        """warmup = max(lookbacks) + 1 (기본값: 360 + 1 = 361)."""
        config = DonchianEnsembleConfig()
        assert config.warmup_periods() == 361

    def test_warmup_custom_lookbacks(self) -> None:
        """커스텀 lookback에서 warmup 계산."""
        config = DonchianEnsembleConfig(lookbacks=(10, 20, 50))
        assert config.warmup_periods() == 51

    def test_warmup_single_lookback(self) -> None:
        """단일 lookback에서 warmup 계산."""
        config = DonchianEnsembleConfig(lookbacks=(100,))
        assert config.warmup_periods() == 101


class TestConfigTimeframe:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임에서 annualization_factor=365.0."""
        config = DonchianEnsembleConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임에서 annualization_factor=8760.0."""
        config = DonchianEnsembleConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_4h(self) -> None:
        """4h 타임프레임에서 annualization_factor=2190.0."""
        config = DonchianEnsembleConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_with_override(self) -> None:
        """for_timeframe에 추가 파라미터 오버라이드."""
        config = DonchianEnsembleConfig.for_timeframe("1d", vol_target=0.50)
        assert config.annualization_factor == 365.0
        assert config.vol_target == 0.50

    def test_for_timeframe_unknown_defaults_to_365(self) -> None:
        """알 수 없는 타임프레임은 365.0으로 기본값."""
        config = DonchianEnsembleConfig.for_timeframe("15m")
        assert config.annualization_factor == 365.0


class TestConfigPresets:
    """preset 팩토리 메서드 테스트."""

    def test_conservative_preset(self) -> None:
        """보수적 설정: 긴 lookback 위주, 낮은 vol_target."""
        config = DonchianEnsembleConfig.conservative()
        assert config.lookbacks == (20, 30, 60, 90, 150, 250, 360)
        assert config.vol_target == 0.30

    def test_aggressive_preset(self) -> None:
        """공격적 설정: 짧은 lookback 위주, 높은 vol_target."""
        config = DonchianEnsembleConfig.aggressive()
        assert config.lookbacks == (5, 10, 20, 30, 60)
        assert config.vol_target == 0.50

    def test_conservative_is_valid_config(self) -> None:
        """보수적 설정이 유효한 config인지 확인."""
        config = DonchianEnsembleConfig.conservative()
        assert isinstance(config, DonchianEnsembleConfig)

    def test_aggressive_is_valid_config(self) -> None:
        """공격적 설정이 유효한 config인지 확인."""
        config = DonchianEnsembleConfig.aggressive()
        assert isinstance(config, DonchianEnsembleConfig)
