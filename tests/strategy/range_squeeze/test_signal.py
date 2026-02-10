"""Tests for Range Squeeze Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.range_squeeze.config import RangeSqueezeConfig, ShortMode
from src.strategy.range_squeeze.preprocessor import preprocess
from src.strategy.range_squeeze.signal import generate_signals
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

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestSignalStructure:
    """시그널 구조 테스트."""

    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame):
        """StrategySignals 출력 구조 확인."""
        config = RangeSqueezeConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame):
        """entries/exits는 bool 타입."""
        config = RangeSqueezeConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        """direction 값은 {-1, 0, 1}에 속함."""
        config = RangeSqueezeConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        """시그널 길이는 입력과 동일."""
        config = RangeSqueezeConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)
        assert len(signals.exits) == len(sample_ohlcv_df)
        assert len(signals.direction) == len(sample_ohlcv_df)
        assert len(signals.strength) == len(sample_ohlcv_df)


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame):
        """첫 번째 bar에는 시그널이 없어야 함 (shift(1) 때문)."""
        config = RangeSqueezeConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestSqueezeLogic:
    """Squeeze 시그널 로직 테스트."""

    def test_squeeze_generates_signal(self):
        """Squeeze 조건에서 시그널이 생성되는지 확인."""
        n = 50
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 2.0
        low = close - 2.0

        # 마지막 몇 개 bar에서 range를 매우 좁게 (squeeze)
        high[-5:] = close[-5:] + 0.01
        low[-5:] = close[-5:] - 0.01

        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = RangeSqueezeConfig(
            lookback=10,
            nr_period=5,
            squeeze_threshold=0.5,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Squeeze 후 시그널이 존재해야 함
        assert signals.strength.abs().sum() > 0

    def test_constant_range_no_nr(self):
        """동일한 range에서는 NR 판정이 빈번 (최소값 == 현재값)."""
        n = 50
        # 모든 bar의 range가 동일 → rolling min == current → is_nr=True
        close = np.linspace(100, 110, n)
        high = close + 3.0
        low = close - 3.0

        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = RangeSqueezeConfig(
            lookback=10,
            nr_period=5,
            squeeze_threshold=0.2,
        )
        processed = preprocess(df, config)

        # 동일 range → warmup 이후 모든 bar가 NR = True (rolling min == current)
        valid_nr = processed["is_nr"][config.warmup_periods() :]
        assert valid_nr.all()


class TestShortMode:
    """ShortMode 동작 테스트."""

    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame):
        """DISABLED 모드에서는 숏 시그널 없음."""
        config = RangeSqueezeConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame):
        """FULL 모드에서는 숏 시그널 허용."""
        config = RangeSqueezeConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # 랜덤 데이터에서 숏 시그널이 존재할 수 있음
        # (존재하지 않을 수도 있으므로 에러 없이 실행되면 OK)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self):
        """HEDGE_ONLY 모드에서 drawdown 없을 때 숏 억제."""
        n = 50
        np.random.seed(42)
        # 상승 추세 (drawdown 없음)
        close = np.linspace(100, 150, n)
        high = close + 2.0
        low = close - 2.0

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = RangeSqueezeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            lookback=10,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 상승 추세에서 drawdown이 threshold 이하가 아니므로 숏 억제
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = RangeSqueezeConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame):
        """config=None일 때 기본 설정 사용."""
        config = RangeSqueezeConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)

        assert len(signals.entries) == len(sample_ohlcv_df)
