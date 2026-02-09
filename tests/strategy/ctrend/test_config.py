"""Tests for CTRENDConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ctrend.config import CTRENDConfig
from src.strategy.tsmom.config import ShortMode


class TestCTRENDConfigDefaults:
    """CTRENDConfig 기본값 테스트."""

    def test_default_values(self) -> None:
        """기본값으로 생성 테스트."""
        config = CTRENDConfig()

        assert config.training_window == 252
        assert config.prediction_horizon == 5
        assert config.alpha == 0.5
        assert config.vol_window == 30
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """Frozen 모델이므로 변경 불가."""
        config = CTRENDConfig()

        with pytest.raises(ValidationError):
            config.training_window = 100  # type: ignore[misc]


class TestCTRENDConfigValidation:
    """CTRENDConfig 검증 테스트."""

    def test_training_window_range(self) -> None:
        """training_window 범위 검증."""
        config = CTRENDConfig(training_window=60)
        assert config.training_window == 60

        config = CTRENDConfig(training_window=504)
        assert config.training_window == 504

        with pytest.raises(ValidationError):
            CTRENDConfig(training_window=59)

        with pytest.raises(ValidationError):
            CTRENDConfig(training_window=505)

    def test_alpha_range(self) -> None:
        """alpha (L1 ratio) 범위 검증."""
        config = CTRENDConfig(alpha=0.01)
        assert config.alpha == 0.01

        config = CTRENDConfig(alpha=1.0)
        assert config.alpha == 1.0

        with pytest.raises(ValidationError):
            CTRENDConfig(alpha=0.0)

        with pytest.raises(ValidationError):
            CTRENDConfig(alpha=1.01)

    def test_prediction_horizon_range(self) -> None:
        """prediction_horizon 범위 검증."""
        config = CTRENDConfig(prediction_horizon=1)
        assert config.prediction_horizon == 1

        config = CTRENDConfig(prediction_horizon=21)
        assert config.prediction_horizon == 21

        with pytest.raises(ValidationError):
            CTRENDConfig(prediction_horizon=0)

        with pytest.raises(ValidationError):
            CTRENDConfig(prediction_horizon=22)

    def test_vol_target_validation(self) -> None:
        """vol_target >= min_volatility 검증."""
        config = CTRENDConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            CTRENDConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        config = CTRENDConfig(training_window=252)
        assert config.warmup_periods() == 302  # 252 + 50

        config = CTRENDConfig(training_window=60)
        assert config.warmup_periods() == 110  # 60 + 50

    def test_short_mode(self) -> None:
        """short_mode 설정 테스트."""
        config = CTRENDConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

        config = CTRENDConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY

        config = CTRENDConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

    def test_vol_window_range(self) -> None:
        """vol_window 범위 검증."""
        config = CTRENDConfig(vol_window=5)
        assert config.vol_window == 5

        config = CTRENDConfig(vol_window=120)
        assert config.vol_window == 120

        with pytest.raises(ValidationError):
            CTRENDConfig(vol_window=4)

        with pytest.raises(ValidationError):
            CTRENDConfig(vol_window=121)
