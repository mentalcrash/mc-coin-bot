"""Tests for Copula Pairs Trading signal generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.copula_pairs.config import CopulaPairsConfig
from src.strategy.copula_pairs.preprocessor import preprocess
from src.strategy.copula_pairs.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def sample_pairs_data() -> pd.DataFrame:
    """합성 cointegrated pair 데이터 생성."""
    np.random.seed(42)
    n = 200
    common_factor = np.cumsum(np.random.randn(n) * 200)
    close = 50000.0 + common_factor + np.random.randn(n) * 100
    pair_close = 3000.0 + common_factor * 0.06 + np.random.randn(n) * 50

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
            "pair_close": pair_close,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def default_config() -> CopulaPairsConfig:
    """기본 CopulaPairs Config."""
    return CopulaPairsConfig()


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_output_structure(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """시그널에 entries, exits, direction, strength 필드 존재."""
        processed = preprocess(sample_pairs_data, default_config)
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
        sample_pairs_data: pd.DataFrame,
    ) -> None:
        """direction은 -1, 0, 1 중 하나."""
        config = CopulaPairsConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_pairs_data, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_entries_exits_bool(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """entries/exits는 bool 타입."""
        processed = preprocess(sample_pairs_data, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_first_row_neutral(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        processed = preprocess(sample_pairs_data, default_config)
        signals = generate_signals(processed, default_config)

        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0


class TestLongEntry:
    """Long entry 테스트 (spread_zscore <= -zscore_entry)."""

    def test_long_entry_on_negative_zscore(self) -> None:
        """큰 음수 z-score에서 long entry."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        config = CopulaPairsConfig(zscore_entry=2.0, zscore_exit=0.5, zscore_stop=3.0)

        df = pd.DataFrame(
            {
                "spread_zscore": np.full(n, -2.5),  # <= -2.0 -> long
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # shift(1) 후 두 번째 행부터 direction == 1 (long)
        valid_direction = signals.direction.iloc[1:]
        assert (valid_direction == Direction.LONG).all()


class TestShortEntry:
    """Short entry 테스트 (spread_zscore >= zscore_entry)."""

    def test_short_entry_on_positive_zscore(self) -> None:
        """큰 양수 z-score에서 short entry."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        config = CopulaPairsConfig(zscore_entry=2.0, zscore_exit=0.5, zscore_stop=3.0)

        df = pd.DataFrame(
            {
                "spread_zscore": np.full(n, 2.5),  # >= 2.0 -> short
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # shift(1) 후 두 번째 행부터 direction == -1 (short)
        valid_direction = signals.direction.iloc[1:]
        assert (valid_direction == Direction.SHORT).all()


class TestExit:
    """Exit 시그널 테스트."""

    def test_exit_on_small_zscore(self) -> None:
        """z-score가 exit 임계값 이내이면 포지션 청산."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        config = CopulaPairsConfig(zscore_entry=2.0, zscore_exit=0.5, zscore_stop=3.0)

        df = pd.DataFrame(
            {
                "spread_zscore": np.full(n, 0.3),  # abs <= 0.5 -> exit
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # 모든 direction == 0 (exit)
        valid_direction = signals.direction.iloc[1:]
        assert (valid_direction == Direction.NEUTRAL).all()


class TestStop:
    """Stop 시그널 테스트."""

    def test_stop_on_extreme_zscore(self) -> None:
        """극단적 z-score에서 emergency stop."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        config = CopulaPairsConfig(zscore_entry=2.0, zscore_exit=0.5, zscore_stop=3.0)

        # z-score가 stop 임계값 이상이면 stop 우선 (entry보다 나중에 설정되어 덮어씀)
        df = pd.DataFrame(
            {
                "spread_zscore": np.full(n, 3.5),  # abs >= 3.0 -> stop
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # stop은 exit와 동일하게 direction == 0
        # 3.5는 >= entry(2.0) AND >= stop(3.0)이므로 stop이 우선 (나중에 설정)
        valid_direction = signals.direction.iloc[1:]
        assert (valid_direction == Direction.NEUTRAL).all()


class TestShortModeDisabled:
    """ShortMode.DISABLED 테스트."""

    def test_full_short_mode(
        self,
        sample_pairs_data: pd.DataFrame,
    ) -> None:
        """FULL 모드에서 숏 시그널 허용."""
        config = CopulaPairsConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_pairs_data, config)
        signals = generate_signals(processed, config)

        # FULL 모드에서는 -1, 0, 1 모두 가능
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_disabled_no_shorts(
        self,
        sample_pairs_data: pd.DataFrame,
    ) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = CopulaPairsConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_pairs_data, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
