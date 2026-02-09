"""Tests for CopulaPairsConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.copula_pairs.config import CopulaPairsConfig
from src.strategy.tsmom.config import ShortMode


class TestDefaults:
    """기본값 테스트."""

    def test_default_values(self) -> None:
        """기본값으로 생성 테스트."""
        config = CopulaPairsConfig()

        assert config.formation_window == 63
        assert config.zscore_entry == 2.0
        assert config.zscore_exit == 0.5
        assert config.zscore_stop == 3.0
        assert config.vol_window == 30
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """Frozen 모델이므로 변경 불가."""
        config = CopulaPairsConfig()

        with pytest.raises(ValidationError):
            config.formation_window = 100  # type: ignore[misc]


class TestFormationWindow:
    """formation_window 범위 검증."""

    def test_formation_window_min(self) -> None:
        """최소값 20 허용."""
        config = CopulaPairsConfig(formation_window=20)
        assert config.formation_window == 20

    def test_formation_window_max(self) -> None:
        """최대값 252 허용."""
        config = CopulaPairsConfig(formation_window=252)
        assert config.formation_window == 252

    def test_formation_window_below_min(self) -> None:
        """19 이하 에러."""
        with pytest.raises(ValidationError):
            CopulaPairsConfig(formation_window=19)

    def test_formation_window_above_max(self) -> None:
        """253 이상 에러."""
        with pytest.raises(ValidationError):
            CopulaPairsConfig(formation_window=253)


class TestZscoreEntry:
    """zscore_entry 범위 검증."""

    def test_zscore_entry_min(self) -> None:
        """최소값 1.0 허용."""
        config = CopulaPairsConfig(zscore_entry=1.0, zscore_exit=0.5, zscore_stop=2.5)
        assert config.zscore_entry == 1.0

    def test_zscore_entry_max(self) -> None:
        """최대값 4.0 허용."""
        config = CopulaPairsConfig(zscore_entry=4.0, zscore_stop=4.5)
        assert config.zscore_entry == 4.0

    def test_zscore_entry_below_min(self) -> None:
        """0.9 이하 에러."""
        with pytest.raises(ValidationError):
            CopulaPairsConfig(zscore_entry=0.9)

    def test_zscore_entry_above_max(self) -> None:
        """4.1 이상 에러."""
        with pytest.raises(ValidationError):
            CopulaPairsConfig(zscore_entry=4.1)


class TestZscoreExitLtEntry:
    """zscore_exit < zscore_entry 검증."""

    def test_exit_lt_entry_valid(self) -> None:
        """exit < entry 유효."""
        config = CopulaPairsConfig(zscore_entry=2.0, zscore_exit=0.5)
        assert config.zscore_exit < config.zscore_entry

    def test_exit_eq_entry_invalid(self) -> None:
        """exit == entry 에러."""
        with pytest.raises(ValidationError, match="zscore_exit"):
            CopulaPairsConfig(zscore_entry=2.0, zscore_exit=2.0)

    def test_exit_gt_entry_invalid(self) -> None:
        """exit > entry는 Field 범위(le=2.0)에 의해 실질적으로 불가."""
        # zscore_exit le=2.0, zscore_entry ge=1.0이므로
        # exit=2.0, entry=1.0이면 validator에 의해 거부
        with pytest.raises(ValidationError, match="zscore_exit"):
            CopulaPairsConfig(zscore_entry=1.5, zscore_exit=1.5, zscore_stop=2.5)


class TestZscoreStopGtEntry:
    """zscore_stop > zscore_entry 검증."""

    def test_stop_gt_entry_valid(self) -> None:
        """stop > entry 유효."""
        config = CopulaPairsConfig(zscore_entry=2.0, zscore_stop=3.0)
        assert config.zscore_stop > config.zscore_entry

    def test_stop_eq_entry_invalid(self) -> None:
        """stop == entry 에러."""
        with pytest.raises(ValidationError, match="zscore_stop"):
            CopulaPairsConfig(zscore_entry=2.5, zscore_stop=2.5)

    def test_stop_lt_entry_invalid(self) -> None:
        """stop < entry 에러."""
        with pytest.raises(ValidationError, match="zscore_stop"):
            CopulaPairsConfig(zscore_entry=3.0, zscore_stop=2.5)


class TestVolTargetValidation:
    """vol_target >= min_volatility 검증."""

    def test_vol_target_gte_min_vol(self) -> None:
        """vol_target >= min_volatility 유효."""
        config = CopulaPairsConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

    def test_vol_target_lt_min_vol(self) -> None:
        """vol_target < min_volatility 에러."""
        with pytest.raises(ValidationError, match="vol_target"):
            CopulaPairsConfig(vol_target=0.05, min_volatility=0.10)


class TestWarmupPeriods:
    """warmup_periods() 테스트."""

    def test_warmup_default(self) -> None:
        """기본값 formation_window=63 -> warmup=64."""
        config = CopulaPairsConfig()
        assert config.warmup_periods() == 64

    def test_warmup_custom(self) -> None:
        """커스텀 formation_window=100 -> warmup=101."""
        config = CopulaPairsConfig(formation_window=100, zscore_stop=3.0)
        assert config.warmup_periods() == 101


class TestShortModeConfig:
    """숏 모드 설정 테스트."""

    def test_accepts_short_mode(self) -> None:
        """ShortMode 수용 테스트."""
        config = CopulaPairsConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

        config = CopulaPairsConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL
