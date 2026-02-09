"""Tests for HurstRegimeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.hurst_regime.config import HurstRegimeConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode 값을 정상적으로 받는지 확인."""
        config_disabled = HurstRegimeConfig(short_mode=ShortMode.DISABLED)
        assert config_disabled.short_mode == ShortMode.DISABLED

        config_full = HurstRegimeConfig(short_mode=ShortMode.FULL)
        assert config_full.short_mode == ShortMode.FULL

        config_hedge = HurstRegimeConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config_hedge.short_mode == ShortMode.HEDGE_ONLY


class TestHurstRegimeConfig:
    """HurstRegimeConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = HurstRegimeConfig()

        assert config.er_lookback == 20
        assert config.hurst_window == 100
        assert config.mom_lookback == 20
        assert config.mr_lookback == 20
        assert config.er_trend_threshold == 0.6
        assert config.er_mr_threshold == 0.3
        assert config.hurst_trend_threshold == 0.55
        assert config.hurst_mr_threshold == 0.45
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = HurstRegimeConfig()

        with pytest.raises(ValidationError):
            config.er_lookback = 30  # type: ignore[misc]

    def test_er_lookback_range(self):
        """er_lookback 범위 검증."""
        config = HurstRegimeConfig(er_lookback=5)
        assert config.er_lookback == 5

        config = HurstRegimeConfig(er_lookback=60)
        assert config.er_lookback == 60

        with pytest.raises(ValidationError):
            HurstRegimeConfig(er_lookback=4)

        with pytest.raises(ValidationError):
            HurstRegimeConfig(er_lookback=61)

    def test_hurst_window_range(self):
        """hurst_window 범위 검증."""
        config = HurstRegimeConfig(hurst_window=50)
        assert config.hurst_window == 50

        config = HurstRegimeConfig(hurst_window=252)
        assert config.hurst_window == 252

        with pytest.raises(ValidationError):
            HurstRegimeConfig(hurst_window=49)

        with pytest.raises(ValidationError):
            HurstRegimeConfig(hurst_window=253)

    def test_mom_lookback_range(self):
        """mom_lookback 범위 검증."""
        config = HurstRegimeConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = HurstRegimeConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            HurstRegimeConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            HurstRegimeConfig(mom_lookback=61)

    def test_mr_lookback_range(self):
        """mr_lookback 범위 검증."""
        config = HurstRegimeConfig(mr_lookback=5)
        assert config.mr_lookback == 5

        config = HurstRegimeConfig(mr_lookback=60)
        assert config.mr_lookback == 60

        with pytest.raises(ValidationError):
            HurstRegimeConfig(mr_lookback=4)

        with pytest.raises(ValidationError):
            HurstRegimeConfig(mr_lookback=61)

    def test_er_trend_gt_mr_validation(self):
        """er_trend_threshold > er_mr_threshold 검증."""
        # 유효한 경우
        config = HurstRegimeConfig(er_trend_threshold=0.7, er_mr_threshold=0.2)
        assert config.er_trend_threshold > config.er_mr_threshold

        # er_trend <= er_mr는 에러
        with pytest.raises(ValidationError):
            HurstRegimeConfig(er_trend_threshold=0.3, er_mr_threshold=0.3)

        with pytest.raises(ValidationError):
            HurstRegimeConfig(er_trend_threshold=0.3, er_mr_threshold=0.4)

    def test_hurst_trend_gt_mr_validation(self):
        """hurst_trend_threshold > hurst_mr_threshold 검증."""
        # 유효한 경우
        config = HurstRegimeConfig(hurst_trend_threshold=0.60, hurst_mr_threshold=0.40)
        assert config.hurst_trend_threshold > config.hurst_mr_threshold

        # hurst_trend <= hurst_mr는 에러
        with pytest.raises(ValidationError):
            HurstRegimeConfig(hurst_trend_threshold=0.50, hurst_mr_threshold=0.50)

        with pytest.raises(ValidationError):
            HurstRegimeConfig(hurst_trend_threshold=0.50, hurst_mr_threshold=0.50)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = HurstRegimeConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # 같은 경우도 유효
        config = HurstRegimeConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            HurstRegimeConfig(vol_target=0.05, min_volatility=0.10)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = HurstRegimeConfig(
            hurst_window=100,
            er_lookback=20,
            mom_lookback=20,
            mr_lookback=20,
            vol_window=20,
            atr_period=14,
        )
        # max(100, 20, 20, 20, 20, 14) + 1 = 101
        assert config.warmup_periods() == 101

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = HurstRegimeConfig(
            hurst_window=50,
            er_lookback=60,
            mom_lookback=60,
            mr_lookback=60,
            vol_window=60,
            atr_period=50,
        )
        # max(50, 60, 60, 60, 60, 50) + 1 = 61
        assert config.warmup_periods() == 61
