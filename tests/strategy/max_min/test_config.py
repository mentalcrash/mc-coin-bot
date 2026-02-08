"""Tests for MaxMinConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.max_min.config import MaxMinConfig
from src.strategy.tsmom.config import ShortMode


class TestMaxMinConfig:
    """MaxMinConfig 테스트."""

    def test_default_values(self) -> None:
        """기본값이 올바르게 설정되는지 확인."""
        config = MaxMinConfig()

        assert config.lookback == 10
        assert config.max_weight == 0.5
        assert config.min_weight == 0.5
        assert config.vol_target == 0.30
        assert config.short_mode == ShortMode.DISABLED
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0

    def test_frozen_model(self) -> None:
        """frozen 모델이므로 속성 수정 불가."""
        config = MaxMinConfig()

        with pytest.raises(ValidationError):
            config.lookback = 20  # type: ignore[misc]

    def test_weight_sum_validation(self) -> None:
        """max_weight + min_weight != 1.0이면 ValidationError."""
        # 합이 1.0이면 정상
        MaxMinConfig(max_weight=0.7, min_weight=0.3)
        MaxMinConfig(max_weight=1.0, min_weight=0.0)
        MaxMinConfig(max_weight=0.0, min_weight=1.0)

        # 합이 1.0이 아니면 에러
        with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
            MaxMinConfig(max_weight=0.5, min_weight=0.4)

        with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
            MaxMinConfig(max_weight=0.8, min_weight=0.8)

    def test_vol_target_validation(self) -> None:
        """vol_target < min_volatility이면 ValidationError."""
        # vol_target >= min_volatility이면 정상
        MaxMinConfig(vol_target=0.30, min_volatility=0.05)
        MaxMinConfig(vol_target=0.05, min_volatility=0.05)

        # vol_target < min_volatility이면 에러
        with pytest.raises(ValidationError, match="vol_target"):
            MaxMinConfig(vol_target=0.04, min_volatility=0.05)

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d') annualization_factor == 365.0."""
        config = MaxMinConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0
        assert config.lookback == 10

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h') annualization_factor == 2190.0."""
        config = MaxMinConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0
        assert config.lookback == 12

    def test_warmup_periods(self) -> None:
        """warmup_periods == lookback + vol_window + 1."""
        config = MaxMinConfig(lookback=10, vol_window=30)
        assert config.warmup_periods() == 10 + 30 + 1

        config2 = MaxMinConfig(lookback=20, vol_window=48)
        assert config2.warmup_periods() == 20 + 48 + 1

    def test_conservative_preset(self) -> None:
        """conservative() 프리셋 검증."""
        config = MaxMinConfig.conservative()

        assert config.lookback == 20
        assert config.max_weight == 0.6
        assert config.min_weight == 0.4
        assert config.vol_window == 48
        assert config.vol_target == 0.15
        assert config.min_volatility == 0.08

    def test_aggressive_preset(self) -> None:
        """aggressive() 프리셋 검증."""
        config = MaxMinConfig.aggressive()

        assert config.lookback == 5
        assert config.max_weight == 0.4
        assert config.min_weight == 0.6
        assert config.vol_window == 14
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05

    def test_lookback_range(self) -> None:
        """lookback 범위 검증 (5-60)."""
        MaxMinConfig(lookback=5)
        MaxMinConfig(lookback=60)

        with pytest.raises(ValidationError):
            MaxMinConfig(lookback=4)

        with pytest.raises(ValidationError):
            MaxMinConfig(lookback=61)

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe에 kwargs 오버라이드."""
        config = MaxMinConfig.for_timeframe("4h", vol_target=0.40)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.40

    def test_short_mode_accepted(self) -> None:
        """Config에서 ShortMode 사용 가능."""
        config = MaxMinConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config2 = MaxMinConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config2.short_mode == ShortMode.HEDGE_ONLY
