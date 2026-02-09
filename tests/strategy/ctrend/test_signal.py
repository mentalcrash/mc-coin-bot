"""Tests for CTREND signal generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.ctrend.config import CTRENDConfig
from src.strategy.ctrend.preprocessor import preprocess
from src.strategy.ctrend.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (400일, training_window=252 고려)."""
    np.random.seed(42)
    n = 400

    # 상승 추세 + 노이즈 (시그널이 발생하도록)
    trend = np.linspace(0, 60, n)
    noise = np.cumsum(np.random.randn(n) * 2)
    close = 100 + trend + noise
    high = close + np.abs(np.random.randn(n) * 1.5) + 0.5
    low = close - np.abs(np.random.randn(n) * 1.5) - 0.5
    open_ = close + np.random.randn(n) * 0.5

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def default_config() -> CTRENDConfig:
    """기본 CTRENDConfig (training_window=100 for faster test)."""
    return CTRENDConfig(training_window=100)


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_output_structure(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
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

    def test_entries_exits_bool(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """entries와 exits는 bool 타입."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_strength_float(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """strength는 float 타입."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.strength.dtype in [np.float64, np.float32, float]


class TestDirectionValues:
    """direction 값 테스트."""

    def test_direction_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """direction은 -1, 0, 1 중 하나의 부분집합."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_first_row_neutral(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0


class TestNoLookaheadBias:
    """미래 참조 편향 방지 테스트."""

    def test_no_lookahead_bias(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: CTRENDConfig,
    ) -> None:
        """마지막 행 제거 시 이전 시그널이 동일해야 함 (no lookahead)."""
        processed_full = preprocess(sample_ohlcv_df, default_config)
        signals_full = generate_signals(processed_full, default_config)

        # 마지막 50개 행 제거
        truncated_df = sample_ohlcv_df.iloc[:-50]
        processed_trunc = preprocess(truncated_df, default_config)
        signals_trunc = generate_signals(processed_trunc, default_config)

        # 겹치는 구간에서 시그널 비교 (training_window 이후)
        # Rolling training은 같은 데이터면 같은 모델을 학습
        overlap_idx = signals_trunc.direction.index
        warmup_end = default_config.training_window + 52  # warmup + shift

        if warmup_end < len(overlap_idx):
            full_dir = signals_full.direction.loc[overlap_idx].iloc[warmup_end:]
            trunc_dir = signals_trunc.direction.iloc[warmup_end:]

            # 동일한 training data이므로 direction이 일치해야 함
            np.testing.assert_array_equal(
                full_dir.values,
                trunc_dir.values,
            )


class TestShortModes:
    """숏 모드 테스트."""

    def test_full_short_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """FULL 모드에서 숏 시그널 허용."""
        config = CTRENDConfig(training_window=100, short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # FULL 모드에서는 -1, 0, 1 모두 가능
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_disabled_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = CTRENDConfig(training_window=100, short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()


class TestInsufficientData:
    """데이터 부족 시 동작 테스트."""

    def test_insufficient_data_returns_neutral(self) -> None:
        """training_window보다 짧은 데이터에서 중립 시그널."""
        np.random.seed(42)
        n = 50  # training_window(100)보다 짧음

        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 1.0
        low = close - 1.0

        df = pd.DataFrame(
            {
                "open": close + 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = CTRENDConfig(training_window=100)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # training_window > n이므로 모든 시그널이 중립
        assert (signals.direction == Direction.NEUTRAL).all()
        assert (signals.strength == 0.0).all()


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_feat_columns(self) -> None:
        """feature 컬럼이 없으면 에러."""
        n = 100
        df = pd.DataFrame(
            {
                "close": np.full(n, 100.0),
                "forward_return": np.zeros(n),
                "vol_scalar": np.ones(n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        with pytest.raises(ValueError, match="No feature columns"):
            generate_signals(df)

    def test_missing_required_columns(self) -> None:
        """forward_return, vol_scalar 누락 시 에러."""
        n = 100
        data: dict[str, np.ndarray] = {"feat_test": np.zeros(n)}
        df = pd.DataFrame(
            data,
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
