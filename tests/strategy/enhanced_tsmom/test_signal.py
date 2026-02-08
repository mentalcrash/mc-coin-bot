"""Tests for Enhanced VW-TSMOM Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.enhanced_tsmom.config import EnhancedTSMOMConfig, ShortMode
from src.strategy.enhanced_tsmom.preprocessor import preprocess
from src.strategy.enhanced_tsmom.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """트렌딩 패턴의 샘플 OHLCV DataFrame."""
    np.random.seed(42)
    n = 200

    base_price = 50000.0
    trend = np.linspace(0, 5000, n)
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + trend + noise

    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = EnhancedTSMOMConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalStructure:
    """시그널 구조 테스트."""

    def test_returns_strategy_signals(self, processed_df: pd.DataFrame):
        """StrategySignals NamedTuple 반환."""
        signals = generate_signals(processed_df)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_bool_dtype(self, processed_df: pd.DataFrame):
        """entries는 bool Series."""
        signals = generate_signals(processed_df)
        assert signals.entries.dtype == bool

    def test_exits_bool_dtype(self, processed_df: pd.DataFrame):
        """exits는 bool Series."""
        signals = generate_signals(processed_df)
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame):
        """direction은 -1, 0, 1 값만."""
        config = EnhancedTSMOMConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)
        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_same_length(self, processed_df: pd.DataFrame):
        """모든 시그널이 동일한 길이."""
        signals = generate_signals(processed_df)
        n = len(processed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_row_is_zero(self, processed_df: pd.DataFrame):
        """첫 행의 strength는 0 (shift로 인한 NaN -> 0)."""
        signals = generate_signals(processed_df)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame):
        """마지막 봉의 데이터 변경이 그 전 봉의 시그널에 영향 없음."""
        config = EnhancedTSMOMConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉 가격 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("close")] = 999999.0
        modified_df.iloc[-1, modified_df.columns.get_loc("evw_momentum")] = 99.0

        signals_after = generate_signals(modified_df, config)
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestShortMode:
    """ShortMode 테스트."""

    def test_disabled_no_shorts(self, processed_df: pd.DataFrame):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = EnhancedTSMOMConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)
        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_has_both_directions(self, processed_df: pd.DataFrame):
        """FULL 모드에서 롱/숏 모두 존재."""
        config = EnhancedTSMOMConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)
        has_long = (signals.direction == Direction.LONG).any()
        has_short = (signals.direction == Direction.SHORT).any()
        assert has_long
        assert has_short

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)

    def test_vol_target_affects_strength(self, sample_ohlcv: pd.DataFrame):
        """vol_target이 strength 크기에 영향."""
        config_low = EnhancedTSMOMConfig(vol_target=0.10, short_mode=ShortMode.FULL)
        config_high = EnhancedTSMOMConfig(vol_target=0.40, short_mode=ShortMode.FULL)

        processed_low = preprocess(sample_ohlcv, config_low)
        processed_high = preprocess(sample_ohlcv, config_high)

        signals_low = generate_signals(processed_low, config_low)
        signals_high = generate_signals(processed_high, config_high)

        avg_abs_low = signals_low.strength.abs().mean()
        avg_abs_high = signals_high.strength.abs().mean()
        assert avg_abs_high > avg_abs_low
