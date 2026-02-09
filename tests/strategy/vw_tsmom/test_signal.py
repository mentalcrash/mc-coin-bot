"""Tests for VW-TSMOM Pure signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction
from src.strategy.vw_tsmom.config import VWTSMOMConfig
from src.strategy.vw_tsmom.preprocessor import preprocess
from src.strategy.vw_tsmom.signal import generate_signals


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
def default_config() -> VWTSMOMConfig:
    """기본 VW-TSMOM Config."""
    return VWTSMOMConfig()


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_signal_output_structure(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VWTSMOMConfig,
    ):
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

        assert len(signals.entries) == len(processed)
        assert len(signals.exits) == len(processed)
        assert len(signals.direction) == len(processed)
        assert len(signals.strength) == len(processed)

    def test_direction_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """direction은 -1, 0, 1 중 하나."""
        config = VWTSMOMConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_entries_exits_are_bool(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VWTSMOMConfig,
    ):
        """entries와 exits는 bool Series."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_strength_is_float(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VWTSMOMConfig,
    ):
        """strength는 float Series."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert np.issubdtype(signals.strength.dtype, np.floating)


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift_1_rule(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        config = VWTSMOMConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0


class TestShortModeSignals:
    """숏 모드별 시그널 테스트."""

    def test_long_only_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = VWTSMOMConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_short_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """FULL 모드에서 숏 시그널 허용."""
        config = VWTSMOMConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_hedge_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """HEDGE_ONLY 모드에서 드로다운 임계값에 따라 숏 제어."""
        config = VWTSMOMConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.07,
            hedge_strength_ratio=0.8,
        )
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # 시그널이 생성되는지 확인
        assert len(signals.entries) == len(processed)


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_preprocessed_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
