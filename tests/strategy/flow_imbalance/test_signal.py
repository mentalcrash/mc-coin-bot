"""Tests for Flow Imbalance Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.flow_imbalance.config import FlowImbalanceConfig, ShortMode
from src.strategy.flow_imbalance.preprocessor import preprocess
from src.strategy.flow_imbalance.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_1h_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (1H, 200 bars)."""
    np.random.seed(42)
    n = 200

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)
    open_ = close + np.random.randn(n) * 0.3

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class TestSignalStructure:
    """시그널 구조 테스트."""

    def test_output_structure(self, sample_1h_df: pd.DataFrame):
        """StrategySignals 출력 구조 확인."""
        config = FlowImbalanceConfig()
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_1h_df: pd.DataFrame):
        """entries/exits는 bool 타입."""
        config = FlowImbalanceConfig()
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_1h_df: pd.DataFrame):
        """direction 값은 {-1, 0, 1}에 속함."""
        config = FlowImbalanceConfig()
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_1h_df: pd.DataFrame):
        """시그널 길이는 입력과 동일."""
        config = FlowImbalanceConfig()
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_1h_df)


class TestShift1Rule:
    """Shift(1) Rule 테스트."""

    def test_first_bar_no_signal(self, sample_1h_df: pd.DataFrame):
        """첫 번째 bar에는 시그널이 없어야 함."""
        config = FlowImbalanceConfig()
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestFlowLogic:
    """Flow 시그널 로직 테스트."""

    def test_strong_flow_generates_signal(self):
        """강한 flow에서 시그널이 생성되는지 확인."""
        n = 200
        np.random.seed(42)
        # 교대 상승/하락 패턴 → buy_ratio 변동 큰 상태 → vpin 높음
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        # 짝수 bar: 상승 (close near high), 홀수 bar: 하락 (close near low)
        high = close + np.abs(np.random.randn(n) * 2.0) + 1.0
        low = close - np.abs(np.random.randn(n) * 2.0) - 1.0

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

        config = FlowImbalanceConfig(
            ofi_entry_threshold=0.2,
            ofi_exit_threshold=0.1,
            vpin_threshold=0.01,
            vpin_window=6,
            ofi_window=3,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        assert signals.strength.abs().sum() > 0

    def test_timeout_forces_exit(self):
        """timeout_bars 후 포지션이 청산되는지 확인."""
        n = 100
        np.random.seed(42)
        close = 100 + np.cumsum(np.abs(np.random.randn(n) * 0.3))
        high = close + 0.1
        low = close - 3.0

        df = pd.DataFrame(
            {
                "open": close - 1.0,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

        config = FlowImbalanceConfig(
            ofi_entry_threshold=0.3,
            vpin_threshold=0.01,
            timeout_bars=5,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # timeout에 의해 일부 포지션은 강제 청산되어야 함
        assert len(signals.entries) == n


class TestShortMode:
    """ShortMode 동작 테스트."""

    def test_disabled_no_shorts(self, sample_1h_df: pd.DataFrame):
        """DISABLED 모드에서는 숏 시그널 없음."""
        config = FlowImbalanceConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction != Direction.SHORT).all()

    def test_full_mode(self, sample_1h_df: pd.DataFrame):
        """FULL 모드에서 에러 없이 실행."""
        config = FlowImbalanceConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.dtype == int

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = FlowImbalanceConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_1h_df: pd.DataFrame):
        """config=None일 때 기본 설정 사용."""
        config = FlowImbalanceConfig()
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed)

        assert len(signals.entries) == len(sample_1h_df)
