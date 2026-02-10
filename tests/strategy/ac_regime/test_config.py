"""Tests for ACRegimeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.ac_regime.config import ACRegimeConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = ACRegimeConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestACRegimeConfig:
    """ACRegimeConfig 테스트."""

    def test_default_values(self):
        config = ACRegimeConfig()

        assert config.ac_window == 60
        assert config.ac_lag == 1
        assert config.significance_z == 1.96
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = ACRegimeConfig()
        with pytest.raises(ValidationError):
            config.ac_window = 100  # type: ignore[misc]

    def test_ac_window_range(self):
        config = ACRegimeConfig(ac_window=20)
        assert config.ac_window == 20

        config = ACRegimeConfig(ac_window=252)
        assert config.ac_window == 252

        with pytest.raises(ValidationError):
            ACRegimeConfig(ac_window=19)

        with pytest.raises(ValidationError):
            ACRegimeConfig(ac_window=253)

    def test_ac_lag_range(self):
        config = ACRegimeConfig(ac_lag=1)
        assert config.ac_lag == 1

        config = ACRegimeConfig(ac_lag=5, ac_window=60)
        assert config.ac_lag == 5

        with pytest.raises(ValidationError):
            ACRegimeConfig(ac_lag=0)

        with pytest.raises(ValidationError):
            ACRegimeConfig(ac_lag=6)

    def test_significance_z_range(self):
        config = ACRegimeConfig(significance_z=1.0)
        assert config.significance_z == 1.0

        config = ACRegimeConfig(significance_z=3.0)
        assert config.significance_z == 3.0

        with pytest.raises(ValidationError):
            ACRegimeConfig(significance_z=0.5)

        with pytest.raises(ValidationError):
            ACRegimeConfig(significance_z=3.5)

    def test_mom_lookback_range(self):
        config = ACRegimeConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = ACRegimeConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            ACRegimeConfig(mom_lookback=4)

        with pytest.raises(ValidationError):
            ACRegimeConfig(mom_lookback=61)

    def test_ac_lag_lt_ac_window_validation(self):
        """ac_lag < ac_window 검증.

        Note: ac_lag max=5, ac_window min=20이므로 Field 범위 제약 상
        validator가 실제로 실패하는 조합은 만들기 어려움.
        정상 케이스만 확인합니다.
        """
        config = ACRegimeConfig(ac_lag=5, ac_window=60)
        assert config.ac_lag < config.ac_window

        config2 = ACRegimeConfig(ac_lag=1, ac_window=20)
        assert config2.ac_lag < config2.ac_window

    def test_vol_target_gte_min_volatility(self):
        config = ACRegimeConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            ACRegimeConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = ACRegimeConfig(
            ac_window=60,
            mom_lookback=20,
            atr_period=14,
        )
        # max(60 + 1, 20, 14) + 1 = 62
        assert config.warmup_periods() == 62

    def test_warmup_periods_custom(self):
        config = ACRegimeConfig(
            ac_window=100,
            mom_lookback=60,
            atr_period=50,
        )
        # max(100 + 1, 60, 50) + 1 = 102
        assert config.warmup_periods() == 102
