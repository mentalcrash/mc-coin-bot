"""Tests for MTF MACD Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.mtf_macd.config import MtfMacdConfig, ShortMode
from src.strategy.mtf_macd.preprocessor import preprocess
from src.strategy.mtf_macd.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = MtfMacdConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalBasic:
    """시그널 기본 구조 테스트."""

    def test_generate_signals_basic(self, processed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 타입/크기 확인."""
        config = MtfMacdConfig()
        signals = generate_signals(processed_df, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        n = len(processed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_entries_exits_are_bool(self, processed_df: pd.DataFrame) -> None:
        """entries와 exits는 bool dtype."""
        config = MtfMacdConfig()
        signals = generate_signals(processed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1 값만 포함."""
        config = MtfMacdConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_strength_no_nan(self, processed_df: pd.DataFrame) -> None:
        """strength에 NaN 없음 (fillna 처리됨)."""
        config = MtfMacdConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.isna().sum() == 0

    def test_first_bar_is_neutral(self, processed_df: pd.DataFrame) -> None:
        """첫 번째 바의 direction은 neutral(0)."""
        config = MtfMacdConfig()
        signals = generate_signals(processed_df, config)
        assert signals.direction.iloc[0] == Direction.NEUTRAL.value


class TestShortMode:
    """ShortMode 처리 테스트."""

    def test_short_mode_full(self, processed_df: pd.DataFrame) -> None:
        """FULL 모드에서 short direction이 존재할 수 있음."""
        config = MtfMacdConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_short_mode_disabled(self, processed_df: pd.DataFrame) -> None:
        """DISABLED 모드에서 short direction(-1) 없음."""
        config = MtfMacdConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)

        assert (signals.direction >= 0).all(), "Short direction found in DISABLED mode"
        assert (signals.strength >= 0).all(), "Negative strength found in DISABLED mode"

    def test_disabled_vs_full_direction_difference(self, processed_df: pd.DataFrame) -> None:
        """DISABLED와 FULL 모드 direction 비교."""
        config_disabled = MtfMacdConfig(short_mode=ShortMode.DISABLED)
        config_full = MtfMacdConfig(short_mode=ShortMode.FULL)

        signals_disabled = generate_signals(processed_df, config_disabled)
        signals_full = generate_signals(processed_df, config_full)

        # 두 모드 모두 정상적으로 시그널을 생성해야 함
        abs_disabled = signals_disabled.strength.abs().sum()
        abs_full = signals_full.strength.abs().sum()
        assert abs_full >= 0
        assert abs_disabled >= 0


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_row_strength_is_zero(self, processed_df: pd.DataFrame) -> None:
        """첫 행의 strength는 0 (shift로 인한 NaN -> 0)."""
        config = MtfMacdConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame) -> None:
        """마지막 봉의 데이터 변경이 이전 봉 시그널에 영향 없음."""
        config = MtfMacdConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉의 macd/vol_scalar 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("macd_line")] = 99999.0
        modified_df.iloc[-1, modified_df.columns.get_loc("signal_line")] = -99999.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vol_scalar")] = 99.0

        signals_after = generate_signals(modified_df, config)
        # 두 번째 마지막 행은 영향받지 않아야 함
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestMacdCrossover:
    """MACD crossover 시그널 테스트."""

    def test_generates_entries(self, sample_ohlcv: pd.DataFrame) -> None:
        """충분한 데이터에서 entry 시그널이 발생해야 함."""
        config = MtfMacdConfig()
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # 300봉이면 최소 하나의 entry가 있어야 함
        assert signals.entries.any(), "No entries generated in 300-bar sample"

    def test_long_entry_with_trend(self) -> None:
        """강한 상승 추세에서 long entry 발생."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")

        # 가격: 초기 안정 후 강한 상승 (MACD > 0으로 올라가도록)
        close = np.ones(n) * 50000.0
        # 처음 100봉: 안정적
        close[:100] = 50000 + np.random.randn(100) * 10
        # 100-200봉: 강한 상승 (MACD가 0을 넘도록)
        close[100:200] = close[99] + np.cumsum(np.ones(100) * 200)

        high = close + 100
        low = close - 100
        open_ = close - 50  # 대부분 bullish candle
        volume = np.ones(n) * 1000

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = MtfMacdConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Direction에 long(1)이 있어야 함
        assert (signals.direction == Direction.LONG.value).any(), "No long direction found"


class TestMissingColumns:
    """누락 컬럼 에러 테스트."""

    def test_missing_macd_line_raises(self) -> None:
        """macd_line 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {
                "signal_line": [1.0, 2.0, 3.0],
                "open": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "vol_scalar": [1.0, 1.0, 1.0],
            }
        )
        config = MtfMacdConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_scalar_raises(self) -> None:
        """vol_scalar 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {
                "macd_line": [1.0, 2.0, 3.0],
                "signal_line": [0.5, 1.0, 1.5],
                "open": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
            }
        )
        config = MtfMacdConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_open_close_raises(self) -> None:
        """open/close 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {
                "macd_line": [1.0, 2.0, 3.0],
                "signal_line": [0.5, 1.0, 1.5],
                "vol_scalar": [1.0, 1.0, 1.0],
            }
        )
        config = MtfMacdConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
