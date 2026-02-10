"""Tests for Hour Seasonality Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hour_season.config import HourSeasonConfig, ShortMode
from src.strategy.hour_season.preprocessor import preprocess
from src.strategy.hour_season.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_1h_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (1H, 1000 bars)."""
    np.random.seed(42)
    n = 1000

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
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_1h_df: pd.DataFrame):
        """entries/exits는 bool 타입."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_1h_df: pd.DataFrame):
        """direction 값은 {-1, 0, 1}에 속함."""
        config = HourSeasonConfig(
            season_window_days=7,
            vol_confirm_window=48,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_1h_df: pd.DataFrame):
        """시그널 길이는 입력과 동일."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_1h_df)


class TestShift1Rule:
    """Shift(1) Rule 테스트."""

    def test_first_bar_no_signal(self, sample_1h_df: pd.DataFrame):
        """첫 번째 bar에는 시그널이 없어야 함."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestSeasonalLogic:
    """Seasonal 시그널 로직 테스트."""

    def test_low_threshold_generates_signals(self, sample_1h_df: pd.DataFrame):
        """낮은 t-stat threshold에서 시그널이 생성되는지 확인."""
        config = HourSeasonConfig(
            season_window_days=7,
            vol_confirm_window=48,
            t_stat_threshold=1.0,
            vol_confirm_threshold=0.3,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert signals.strength.abs().sum() > 0


class TestShortMode:
    """ShortMode 동작 테스트."""

    def test_disabled_no_shorts(self, sample_1h_df: pd.DataFrame):
        """DISABLED 모드에서는 숏 시그널 없음."""
        config = HourSeasonConfig(
            short_mode=ShortMode.DISABLED,
            season_window_days=7,
            vol_confirm_window=48,
        )
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction != Direction.SHORT).all()

    def test_hedge_only_suppresses_without_drawdown(self):
        """HEDGE_ONLY 모드에서 drawdown 없을 때 숏 억제."""
        n = 1000
        np.random.seed(42)
        close = np.linspace(100, 150, n)
        high = close + 0.8
        low = close - 0.8

        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

        config = HourSeasonConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            season_window_days=7,
            vol_confirm_window=48,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = HourSeasonConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_1h_df: pd.DataFrame):
        """config=None일 때 기본 설정 사용."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        processed = preprocess(sample_1h_df, config)
        signals = generate_signals(processed)

        assert len(signals.entries) == len(sample_1h_df)
