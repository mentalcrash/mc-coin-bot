"""Tests for MAX/MIN Signal Generator."""

import pandas as pd
import pytest

from src.strategy.max_min.config import MaxMinConfig
from src.strategy.max_min.preprocessor import preprocess
from src.strategy.max_min.signal import generate_signals
from src.strategy.tsmom.config import ShortMode


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame (preprocess 적용)."""
    config = MaxMinConfig()
    return preprocess(sample_ohlcv, config)


class TestGenerateSignals:
    """generate_signals() 테스트."""

    def test_generate_signals_basic(self, preprocessed_df: pd.DataFrame) -> None:
        """기본 실행: 출력 타입 확인."""
        config = MaxMinConfig()
        signals = generate_signals(preprocessed_df, config)

        assert isinstance(signals.entries, pd.Series)
        assert isinstance(signals.exits, pd.Series)
        assert isinstance(signals.direction, pd.Series)
        assert isinstance(signals.strength, pd.Series)

        # 길이 일치
        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_shift1_rule(self, preprocessed_df: pd.DataFrame) -> None:
        """Shift(1) Rule: 첫 번째 시그널은 0이어야 함.

        vol_scalar.shift(1)로 인해 첫 번째 행의 strength는 0.
        """
        config = MaxMinConfig()
        signals = generate_signals(preprocessed_df, config)

        # 첫 번째 행: vol_scalar.shift(1) == NaN → fillna(0) → strength == 0
        assert signals.strength.iloc[0] == 0.0

    def test_long_only_mode(self, preprocessed_df: pd.DataFrame) -> None:
        """DISABLED short_mode: direction에 음수(-1) 값 없음."""
        config = MaxMinConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        assert (signals.direction >= 0).all()

    def test_entries_exits_are_bool(self, preprocessed_df: pd.DataFrame) -> None:
        """entries/exits는 bool dtype."""
        config = MaxMinConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame) -> None:
        """direction 값은 -1, 0, 1만 허용."""
        config = MaxMinConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        unique_vals = set(signals.direction.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        config = MaxMinConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_strength_nonnegative_in_disabled_mode(self, preprocessed_df: pd.DataFrame) -> None:
        """DISABLED 모드에서 strength는 0 이상."""
        config = MaxMinConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        assert (signals.strength >= 0).all()

    def test_default_config_used_when_none(self, preprocessed_df: pd.DataFrame) -> None:
        """config=None이면 기본 MaxMinConfig 사용."""
        signals = generate_signals(preprocessed_df, config=None)

        assert isinstance(signals.entries, pd.Series)
        assert len(signals.entries) == len(preprocessed_df)

    def test_entries_at_direction_change(self, preprocessed_df: pd.DataFrame) -> None:
        """entries는 direction이 변하는 시점에서 발생."""
        config = MaxMinConfig()
        signals = generate_signals(preprocessed_df, config)

        # entries가 True인 위치에서 direction이 이전과 다른지 확인
        prev_dir = signals.direction.shift(1).fillna(0)
        for idx in signals.entries[signals.entries].index:
            curr = signals.direction[idx]
            prev = prev_dir[idx]
            # entry 시점에서는 direction이 LONG 또는 SHORT이고 이전과 달라야 함
            assert curr != 0
            assert curr != prev
