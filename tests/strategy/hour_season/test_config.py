"""Tests for HourSeasonConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.hour_season.config import HourSeasonConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = HourSeasonConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL


class TestHourSeasonConfig:
    """HourSeasonConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = HourSeasonConfig()

        assert config.season_window_days == 30
        assert config.t_stat_threshold == 2.0
        assert config.vol_confirm_window == 168
        assert config.vol_confirm_threshold == 1.0
        assert config.vol_target == 0.30
        assert config.annualization_factor == 8760.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = HourSeasonConfig()

        with pytest.raises(ValidationError):
            config.season_window_days = 60  # type: ignore[misc]

    def test_season_window_days_range(self):
        """season_window_days 범위 검증."""
        config = HourSeasonConfig(season_window_days=7)
        assert config.season_window_days == 7

        config = HourSeasonConfig(season_window_days=90)
        assert config.season_window_days == 90

        with pytest.raises(ValidationError):
            HourSeasonConfig(season_window_days=6)

        with pytest.raises(ValidationError):
            HourSeasonConfig(season_window_days=91)

    def test_t_stat_threshold_range(self):
        """t_stat_threshold 범위 검증."""
        config = HourSeasonConfig(t_stat_threshold=1.0)
        assert config.t_stat_threshold == 1.0

        config = HourSeasonConfig(t_stat_threshold=4.0)
        assert config.t_stat_threshold == 4.0

        with pytest.raises(ValidationError):
            HourSeasonConfig(t_stat_threshold=0.9)

        with pytest.raises(ValidationError):
            HourSeasonConfig(t_stat_threshold=4.1)

    def test_vol_confirm_window_range(self):
        """vol_confirm_window 범위 검증."""
        config = HourSeasonConfig(vol_confirm_window=24)
        assert config.vol_confirm_window == 24

        with pytest.raises(ValidationError):
            HourSeasonConfig(vol_confirm_window=23)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        with pytest.raises(ValidationError):
            HourSeasonConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = HourSeasonConfig(season_window_days=30)
        # 30 * 24 + 1 = 721
        assert config.warmup_periods() == 721

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = HourSeasonConfig(season_window_days=60)
        # 60 * 24 + 1 = 1441
        assert config.warmup_periods() == 1441

    def test_vol_confirm_threshold_range(self):
        """vol_confirm_threshold 범위 검증."""
        config = HourSeasonConfig(vol_confirm_threshold=0.3)
        assert config.vol_confirm_threshold == 0.3

        with pytest.raises(ValidationError):
            HourSeasonConfig(vol_confirm_threshold=0.2)

        with pytest.raises(ValidationError):
            HourSeasonConfig(vol_confirm_threshold=3.1)
