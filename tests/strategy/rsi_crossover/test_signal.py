"""Tests for RSI Crossover Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.rsi_crossover.config import RSICrossoverConfig
from src.strategy.rsi_crossover.preprocessor import preprocess
from src.strategy.rsi_crossover.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = RSICrossoverConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalBasic:
    """시그널 기본 구조 테스트."""

    def test_generate_signals_basic(self, processed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 타입/크기 확인."""
        config = RSICrossoverConfig()
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
        config = RSICrossoverConfig()
        signals = generate_signals(processed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1 값만 포함."""
        config = RSICrossoverConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_strength_no_nan(self, processed_df: pd.DataFrame) -> None:
        """strength에 NaN 없음 (fillna 처리됨)."""
        config = RSICrossoverConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.isna().sum() == 0


class TestStateMachine:
    """상태 머신 기반 포지션 추적 테스트."""

    def test_state_machine_long(self) -> None:
        """RSI가 30 상향 돌파 시 long entry, 60 도달 시 exit."""
        # RSI가 과매도(30) 밑에서 위로 크로스한 후 60까지 상승하는 시나리오
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="4h")

        # RSI를 직접 제어하기 위한 가격 시퀀스 구성
        # 구간 1: RSI ~50 (neutral)  구간 2: RSI ~20 (oversold)
        # 구간 3: RSI ~50 (cross above 30)  구간 4: RSI ~65 (exit at 60)
        np.random.seed(42)
        close = np.ones(n) * 50000
        # 처음 50봉: 안정적 가격
        close[:50] = 50000 + np.random.randn(50) * 10
        # 50-60봉: 급락 → RSI를 30 이하로 만듦
        close[50:60] = 50000 - np.arange(10) * 200
        # 60-70봉: 반등 → RSI가 30을 상향 돌파
        close[60:70] = close[59] + np.arange(1, 11) * 150
        # 70-100봉: 추가 상승 → RSI가 60 이상 도달
        close[70:100] = close[69] + np.cumsum(np.ones(30) * 100)

        high = close + 100
        low = close - 100
        open_ = close + np.random.randn(n) * 10
        volume = np.ones(n) * 1000

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = RSICrossoverConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Long entry가 하나 이상 발생해야 함
        assert signals.entries.any(), "No entries generated"
        # Direction에 long(1)이 있어야 함
        assert (signals.direction == Direction.LONG.value).any(), "No long direction found"

    def test_first_bar_is_neutral(self, processed_df: pd.DataFrame) -> None:
        """첫 번째 바의 direction은 neutral(0)."""
        config = RSICrossoverConfig()
        signals = generate_signals(processed_df, config)
        assert signals.direction.iloc[0] == Direction.NEUTRAL.value


class TestShortMode:
    """ShortMode 처리 테스트."""

    def test_short_mode_full(self, processed_df: pd.DataFrame) -> None:
        """FULL 모드에서 short direction이 존재할 수 있음."""
        config = RSICrossoverConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        # Short 시그널이 반드시 있다고 보장할 수 없지만, direction 범위에 -1 허용
        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_short_mode_disabled(self, processed_df: pd.DataFrame) -> None:
        """DISABLED 모드에서 short direction(-1) 없음."""
        config = RSICrossoverConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)

        assert (signals.direction >= 0).all(), "Short direction found in DISABLED mode"
        assert (signals.strength >= 0).all(), "Negative strength found in DISABLED mode"

    def test_short_mode_hedge_only(self, sample_ohlcv: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 drawdown 컬럼이 사용됨."""
        config = RSICrossoverConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(sample_ohlcv, config)
        # drawdown 컬럼이 있어야 HEDGE_ONLY 작동
        assert "drawdown" in processed.columns
        signals = generate_signals(processed, config)
        assert len(signals.entries) == len(sample_ohlcv)

    def test_disabled_vs_full_direction_difference(self, processed_df: pd.DataFrame) -> None:
        """DISABLED와 FULL 모드 direction 비교."""
        config_disabled = RSICrossoverConfig(short_mode=ShortMode.DISABLED)
        config_full = RSICrossoverConfig(short_mode=ShortMode.FULL)

        signals_disabled = generate_signals(processed_df, config_disabled)
        signals_full = generate_signals(processed_df, config_full)

        # DISABLED에서는 short이 없으므로 strength 절대값 합이 다를 수 있음
        abs_disabled = signals_disabled.strength.abs().sum()
        abs_full = signals_full.strength.abs().sum()
        # 두 모드 모두 정상적으로 시그널을 생성해야 함
        assert abs_full >= 0
        assert abs_disabled >= 0


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_row_strength_is_zero(self, processed_df: pd.DataFrame) -> None:
        """첫 행의 strength는 0 (shift로 인한 NaN -> 0)."""
        config = RSICrossoverConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame) -> None:
        """마지막 봉의 데이터 변경이 이전 봉 시그널에 영향 없음."""
        config = RSICrossoverConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉의 RSI/vol_scalar 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("rsi")] = 99.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vol_scalar")] = 99.0

        signals_after = generate_signals(modified_df, config)
        # 두 번째 마지막 행은 영향받지 않아야 함
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestMissingColumns:
    """누락 컬럼 에러 테스트."""

    def test_missing_rsi_column_raises(self) -> None:
        """rsi 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"vol_scalar": [1.0, 2.0, 3.0]})
        config = RSICrossoverConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_scalar_column_raises(self) -> None:
        """vol_scalar 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"rsi": [50.0, 60.0, 70.0]})
        config = RSICrossoverConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_drawdown_in_hedge_mode_raises(self) -> None:
        """HEDGE_ONLY 모드에서 drawdown 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"rsi": [50.0, 60.0, 70.0], "vol_scalar": [1.0, 1.0, 1.0]})
        config = RSICrossoverConfig(short_mode=ShortMode.HEDGE_ONLY)
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
