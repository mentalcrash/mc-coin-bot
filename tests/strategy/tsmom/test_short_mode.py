"""Unit tests for TSMOM short mode functionality.

Tests for:
    - ShortMode enum
    - Conditional hedge short logic
    - Drawdown calculation
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.tsmom import ShortMode, TSMOMConfig, TSMOMStrategy
from src.strategy.tsmom.preprocessor import calculate_drawdown, preprocess
from src.strategy.tsmom.signal import generate_signals
from src.strategy.types import Direction


class TestShortModeConfig:
    """ShortMode enum 및 TSMOMConfig 테스트."""

    def test_short_mode_enum_values(self) -> None:
        """ShortMode enum 값 테스트."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_default_config_is_hedge_only(self) -> None:
        """기본 설정이 HEDGE_ONLY인지 확인."""
        config = TSMOMConfig()
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.effective_short_mode() == ShortMode.HEDGE_ONLY

    def test_hedge_mode_config(self) -> None:
        """헤지 모드 설정 테스트."""
        config = TSMOMConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.15,
            hedge_strength_ratio=0.5,
        )
        assert config.effective_short_mode() == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.15
        assert config.hedge_strength_ratio == 0.5

    def test_full_mode_config(self) -> None:
        """Full 모드 설정 테스트."""
        config = TSMOMConfig(short_mode=ShortMode.FULL)
        assert config.effective_short_mode() == ShortMode.FULL

    def test_long_only_backward_compatibility(self) -> None:
        """long_only 필드 하위 호환성 테스트.

        Note: 기본값이 HEDGE_ONLY로 변경되어, long_only만 설정해도
        short_mode가 HEDGE_ONLY이므로 effective는 HEDGE_ONLY가 됩니다.
        DISABLED를 원하면 short_mode=ShortMode.DISABLED를 명시적으로 설정해야 합니다.
        """
        # long_only=True + short_mode=DISABLED → DISABLED
        config = TSMOMConfig(long_only=True, short_mode=ShortMode.DISABLED)
        assert config.effective_short_mode() == ShortMode.DISABLED

        # long_only=False + short_mode=DISABLED → FULL (하위 호환성)
        config = TSMOMConfig(long_only=False, short_mode=ShortMode.DISABLED)
        assert config.effective_short_mode() == ShortMode.FULL

    def test_short_mode_takes_precedence(self) -> None:
        """short_mode가 long_only보다 우선하는지 테스트."""
        config = TSMOMConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            long_only=True,  # 이 값은 무시됨
        )
        assert config.effective_short_mode() == ShortMode.HEDGE_ONLY

    def test_hedge_threshold_validation(self) -> None:
        """헤지 임계값 검증 테스트."""
        # 유효 범위: -0.30 ~ -0.05
        config = TSMOMConfig(hedge_threshold=-0.20)
        assert config.hedge_threshold == -0.20

        with pytest.raises(ValueError):
            TSMOMConfig(hedge_threshold=-0.01)  # 너무 작음

        with pytest.raises(ValueError):
            TSMOMConfig(hedge_threshold=-0.50)  # 너무 큼


class TestCalculateDrawdown:
    """calculate_drawdown 함수 테스트."""

    def test_no_drawdown(self) -> None:
        """상승장에서 드로다운 없음."""
        close = pd.Series([100, 110, 120, 130, 140])
        drawdown = calculate_drawdown(close)

        assert drawdown.iloc[0] == 0.0  # 첫 날은 0
        assert all(drawdown == 0.0)  # 계속 상승하면 드로다운 없음

    def test_simple_drawdown(self) -> None:
        """단순 드로다운 계산 테스트."""
        close = pd.Series([100, 110, 100, 90, 95])
        drawdown = calculate_drawdown(close)

        # 110이 최고점, 이후 드로다운
        assert drawdown.iloc[0] == 0.0
        assert drawdown.iloc[1] == 0.0  # 최고점
        assert drawdown.iloc[2] == pytest.approx(-10 / 110)  # 100 vs 110
        assert drawdown.iloc[3] == pytest.approx(-20 / 110)  # 90 vs 110
        assert drawdown.iloc[4] == pytest.approx(-15 / 110)  # 95 vs 110

    def test_deep_drawdown(self) -> None:
        """깊은 드로다운 테스트 (-20%)."""
        close = pd.Series([100, 120, 96])  # 120 → 96 = -20%
        drawdown = calculate_drawdown(close)

        assert drawdown.iloc[2] == pytest.approx(-0.20)


class TestGenerateSignalsShortMode:
    """generate_signals의 숏 모드 테스트."""

    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """테스트용 OHLCV 데이터 생성."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        # 상승 → 하락 → 상승 패턴
        close = np.concatenate([
            np.linspace(100, 130, 40),  # 상승
            np.linspace(130, 100, 30),  # 하락 (-23%)
            np.linspace(100, 115, 30),  # 회복
        ])

        return pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.random.uniform(1000, 5000, n),
            },
            index=dates,
        )

    def test_disabled_mode_no_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = TSMOMConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # 숏 방향이 없어야 함
        short_count = (signals.direction == Direction.SHORT).sum()
        assert short_count == 0

    def test_full_mode_has_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL 모드에서 숏 시그널 존재."""
        config = TSMOMConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # 하락 구간에서 숏 시그널이 있어야 함
        short_count = (signals.direction == Direction.SHORT).sum()
        assert short_count > 0

    def test_hedge_mode_conditional_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 조건부 숏."""
        config = TSMOMConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.15,  # -15% 드로다운 시 숏 활성화
            hedge_strength_ratio=0.5,
        )
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # 드로다운이 -15% 미만일 때만 숏 가능
        drawdown = processed["drawdown"]
        hedge_active = drawdown < config.hedge_threshold

        # 숏 시그널은 헤지 활성화 기간에만 존재
        short_mask = signals.direction == Direction.SHORT
        if short_mask.any():
            # 숏이 있는 날은 헤지가 활성화된 날이어야 함
            short_indices = short_mask[short_mask].index
            for idx in short_indices:
                assert hedge_active.loc[idx], f"Short at {idx} without hedge active"

    def test_hedge_strength_ratio(self, sample_ohlcv: pd.DataFrame) -> None:
        """헤지 강도 비율 적용 테스트."""
        # Full 모드 강도
        config_full = TSMOMConfig(short_mode=ShortMode.FULL)
        processed_full = preprocess(sample_ohlcv, config_full)
        signals_full = generate_signals(processed_full, config_full)

        # Hedge 모드 강도 (50%)
        config_hedge = TSMOMConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,  # 낮은 임계값으로 더 많은 숏 허용
            hedge_strength_ratio=0.5,
        )
        processed_hedge = preprocess(sample_ohlcv, config_hedge)
        signals_hedge = generate_signals(processed_hedge, config_hedge)

        # 헤지 모드의 숏 강도는 풀 모드의 50% 이하
        short_mask_full = signals_full.direction == Direction.SHORT
        short_mask_hedge = signals_hedge.direction == Direction.SHORT

        if short_mask_full.any() and short_mask_hedge.any():
            # 공통으로 숏인 날 비교
            common_shorts = short_mask_full & short_mask_hedge
            if common_shorts.any():
                full_strength = signals_full.strength[common_shorts].abs()
                hedge_strength = signals_hedge.strength[common_shorts].abs()
                # 헤지 강도가 풀 강도의 50% (약간의 오차 허용)
                ratio = (hedge_strength / full_strength).mean()
                assert ratio <= 0.55  # 50% + 오차


class TestTSMOMStrategyShortMode:
    """TSMOMStrategy의 숏 모드 통합 테스트."""

    def test_strategy_startup_info_default_hedge(self) -> None:
        """기본 설정 (HEDGE_ONLY) 시작 정보."""
        strategy = TSMOMStrategy()
        info = strategy.get_startup_info()
        assert "Hedge-Short" in info["mode"]
        assert info["hedge_strength"] == "50%"

    def test_strategy_startup_info_disabled(self) -> None:
        """DISABLED 모드 시작 정보."""
        config = TSMOMConfig(short_mode=ShortMode.DISABLED)
        strategy = TSMOMStrategy(config)
        info = strategy.get_startup_info()
        assert info["mode"] == "Long-Only"

    def test_strategy_startup_info_hedge(self) -> None:
        """HEDGE_ONLY 모드 시작 정보."""
        config = TSMOMConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.15,
        )
        strategy = TSMOMStrategy(config)
        info = strategy.get_startup_info()
        assert "Hedge-Short" in info["mode"]
        assert "-15%" in info["mode"]
        assert info["hedge_strength"] == "50%"

    def test_strategy_startup_info_full(self) -> None:
        """FULL 모드 시작 정보."""
        config = TSMOMConfig(short_mode=ShortMode.FULL)
        strategy = TSMOMStrategy(config)
        info = strategy.get_startup_info()
        assert info["mode"] == "Long/Short"
