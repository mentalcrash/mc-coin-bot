"""Tests for MultiFactorConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.multi_factor.config import MultiFactorConfig
from src.strategy.tsmom.config import ShortMode


class TestMultiFactorConfig:
    """MultiFactorConfig 테스트."""

    def test_default_values(self) -> None:
        """기본값으로 생성 테스트."""
        config = MultiFactorConfig()

        assert config.momentum_lookback == 21
        assert config.volume_shock_window == 5
        assert config.vol_window == 30
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.zscore_window == 60
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """Frozen 모델이므로 변경 불가."""
        config = MultiFactorConfig()

        with pytest.raises(ValidationError):
            config.momentum_lookback = 30  # type: ignore[misc]

    def test_momentum_lookback_range(self) -> None:
        """momentum_lookback 범위 검증."""
        config = MultiFactorConfig(momentum_lookback=5)
        assert config.momentum_lookback == 5

        config = MultiFactorConfig(momentum_lookback=120)
        assert config.momentum_lookback == 120

        with pytest.raises(ValidationError):
            MultiFactorConfig(momentum_lookback=4)

        with pytest.raises(ValidationError):
            MultiFactorConfig(momentum_lookback=121)

    def test_volume_shock_window_range(self) -> None:
        """volume_shock_window 범위 검증."""
        config = MultiFactorConfig(volume_shock_window=2)
        assert config.volume_shock_window == 2

        config = MultiFactorConfig(volume_shock_window=30)
        assert config.volume_shock_window == 30

        with pytest.raises(ValidationError):
            MultiFactorConfig(volume_shock_window=1)

        with pytest.raises(ValidationError):
            MultiFactorConfig(volume_shock_window=31)

    def test_zscore_window_range(self) -> None:
        """zscore_window 범위 검증."""
        config = MultiFactorConfig(zscore_window=20)
        assert config.zscore_window == 20

        config = MultiFactorConfig(zscore_window=252)
        assert config.zscore_window == 252

        with pytest.raises(ValidationError):
            MultiFactorConfig(zscore_window=19)

        with pytest.raises(ValidationError):
            MultiFactorConfig(zscore_window=253)

    def test_vol_target_validation(self) -> None:
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = MultiFactorConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # 경계값: 같은 값도 유효
        config = MultiFactorConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            MultiFactorConfig(vol_target=0.05, min_volatility=0.10)

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        config = MultiFactorConfig(
            momentum_lookback=21,
            vol_window=30,
            zscore_window=60,
        )
        # max(21, 30, 60) + 1 = 61
        assert config.warmup_periods() == 61

    def test_warmup_periods_custom(self) -> None:
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = MultiFactorConfig(
            momentum_lookback=120,
            vol_window=60,
            zscore_window=100,
        )
        # max(120, 60, 100) + 1 = 121
        assert config.warmup_periods() == 121

    def test_short_mode(self) -> None:
        """short_mode 설정 테스트."""
        config = MultiFactorConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = MultiFactorConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

        config = MultiFactorConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_vol_window_range(self) -> None:
        """vol_window 범위 검증."""
        config = MultiFactorConfig(vol_window=5)
        assert config.vol_window == 5

        config = MultiFactorConfig(vol_window=120)
        assert config.vol_window == 120

        with pytest.raises(ValidationError):
            MultiFactorConfig(vol_window=4)

        with pytest.raises(ValidationError):
            MultiFactorConfig(vol_window=121)
