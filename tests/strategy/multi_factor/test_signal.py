"""Tests for Multi-Factor Ensemble signal generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.multi_factor.config import MultiFactorConfig
from src.strategy.multi_factor.preprocessor import preprocess
from src.strategy.multi_factor.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )

    return df


@pytest.fixture
def default_config() -> MultiFactorConfig:
    """기본 MultiFactorConfig."""
    return MultiFactorConfig()


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_signal_output_structure(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """시그널에 entries, exits, direction, strength 필드 존재."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert len(signals.entries) == len(sample_ohlcv_df)
        assert len(signals.exits) == len(sample_ohlcv_df)
        assert len(signals.direction) == len(sample_ohlcv_df)
        assert len(signals.strength) == len(sample_ohlcv_df)

    def test_direction_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """direction은 -1, 0, 1 중 하나의 부분집합."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_entries_exits_bool(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """entries와 exits는 bool 타입."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_strength_float(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """strength는 float 값."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.strength.dtype in [np.float64, np.float32, float]


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_first_row_neutral(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        # 첫 행은 shift(1) 결과 NaN -> 0으로 채워짐
        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0


class TestShortModes:
    """숏 모드 테스트."""

    def test_short_mode_full(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """FULL 모드에서 숏 시그널 존재 가능."""
        config = MultiFactorConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # 시그널이 정상 생성됨
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_short_mode_disabled(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = MultiFactorConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()


class TestCombinedScoreDirection:
    """결합 점수가 시그널 방향을 결정하는지 테스트."""

    def test_combined_score_drives_direction(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """양의 combined_score -> 롱, 음의 combined_score -> 숏."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        # warmup 이후, combined_score가 유효한 행에서만 확인
        warmup = default_config.warmup_periods()
        dir_after = signals.direction.iloc[warmup:]

        # shift(1) 적용된 combined_score
        shifted_score = processed["combined_score"].shift(1).iloc[warmup:]

        # 양의 score에서는 direction >= 0 (LONG 또는 NEUTRAL)
        positive_mask = shifted_score > 0
        if positive_mask.any():
            positive_dirs = dir_after[positive_mask].dropna()
            if len(positive_dirs) > 0:
                # 양의 score -> direction 1 (LONG)
                assert (positive_dirs >= 0).all()

        # 음의 score에서는 direction <= 0 (SHORT 또는 NEUTRAL)
        negative_mask = shifted_score < 0
        if negative_mask.any():
            negative_dirs = dir_after[negative_mask].dropna()
            if len(negative_dirs) > 0:
                # 음의 score -> direction -1 (SHORT)
                assert (negative_dirs <= 0).all()

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)

    def test_default_config_used(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """config=None일 때 기본값 사용."""
        default_config = MultiFactorConfig()
        processed = preprocess(sample_ohlcv_df, default_config)

        # config=None으로 호출 가능
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)
