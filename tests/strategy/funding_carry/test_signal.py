"""Tests for Funding Rate Carry signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.funding_carry.config import FundingCarryConfig
from src.strategy.funding_carry.preprocessor import preprocess
from src.strategy.funding_carry.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_with_funding() -> pd.DataFrame:
    """샘플 OHLCV + funding_rate DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200
    close = 50000.0 + np.cumsum(np.random.randn(n) * 300)
    funding_rate = np.random.randn(n) * 0.0003

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_with_funding: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = FundingCarryConfig()
    return preprocess(sample_ohlcv_with_funding, config)


class TestSignalOutput:
    """시그널 출력 구조 테스트."""

    def test_output_structure(self, preprocessed_df: pd.DataFrame) -> None:
        """StrategySignals 구조 확인."""
        signals = generate_signals(preprocessed_df)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, preprocessed_df: pd.DataFrame) -> None:
        """entries/exits가 bool Series인지 확인."""
        signals = generate_signals(preprocessed_df)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame) -> None:
        """direction이 -1, 0, 1 값만 가지는지 확인."""
        config = FundingCarryConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_length_matches_input(self, preprocessed_df: pd.DataFrame) -> None:
        """출력 길이가 입력과 동일한지 확인."""
        signals = generate_signals(preprocessed_df)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)


class TestShift1Rule:
    """Shift(1) 규칙 테스트."""

    def test_first_value_neutral(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 시그널은 중립 (shift 때문)."""
        signals = generate_signals(preprocessed_df)

        # shift(1) 때문에 첫 번째 값은 0
        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0

    def test_no_entry_on_first_bar(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 bar에서는 진입 시그널 없음."""
        signals = generate_signals(preprocessed_df)
        assert signals.entries.iloc[0] is np.bool_(False)


class TestCarryDirection:
    """Carry 방향 테스트."""

    def test_positive_funding_rate_short(self) -> None:
        """Positive funding rate -> short direction."""
        n = 200
        np.random.seed(42)
        close = 50000.0 + np.cumsum(np.random.randn(n) * 100)

        # 강한 positive funding rate
        df = pd.DataFrame(
            {
                "close": close,
                "high": close + 200,
                "low": close - 200,
                "open": close,
                "volume": np.full(n, 5000000.0),
                "funding_rate": np.full(n, 0.001),  # consistently positive
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = FundingCarryConfig(
            short_mode=ShortMode.FULL,
            entry_threshold=0.0,  # no threshold
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 유효한 시그널 (NaN 아닌 것) 중 direction 확인
        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            # Positive FR -> 대부분 short
            short_ratio = (valid == Direction.SHORT).sum() / len(valid)
            assert short_ratio > 0.8, f"Expected mostly short, got {short_ratio:.1%}"

    def test_negative_funding_rate_long(self) -> None:
        """Negative funding rate -> long direction."""
        n = 200
        np.random.seed(42)
        close = 50000.0 + np.cumsum(np.random.randn(n) * 100)

        # 강한 negative funding rate
        df = pd.DataFrame(
            {
                "close": close,
                "high": close + 200,
                "low": close - 200,
                "open": close,
                "volume": np.full(n, 5000000.0),
                "funding_rate": np.full(n, -0.001),  # consistently negative
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = FundingCarryConfig(
            short_mode=ShortMode.FULL,
            entry_threshold=0.0,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            long_ratio = (valid == Direction.LONG).sum() / len(valid)
            assert long_ratio > 0.8, f"Expected mostly long, got {long_ratio:.1%}"


class TestEntryThreshold:
    """entry_threshold 필터 테스트."""

    def test_threshold_filters_small_fr(self) -> None:
        """작은 FR은 entry_threshold로 필터링."""
        n = 200
        np.random.seed(42)
        close = 50000.0 + np.cumsum(np.random.randn(n) * 100)

        # 매우 작은 funding rate (threshold 미만)
        df = pd.DataFrame(
            {
                "close": close,
                "high": close + 200,
                "low": close - 200,
                "open": close,
                "volume": np.full(n, 5000000.0),
                "funding_rate": np.full(n, 0.00001),  # very small
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = FundingCarryConfig(entry_threshold=0.001)  # high threshold
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 모든 시그널이 중립이어야 함
        assert (signals.direction == Direction.NEUTRAL).all()

    def test_zero_threshold_always_enters(self) -> None:
        """entry_threshold=0일 때 항상 진입."""
        n = 200
        np.random.seed(42)
        close = 50000.0 + np.cumsum(np.random.randn(n) * 100)

        df = pd.DataFrame(
            {
                "close": close,
                "high": close + 200,
                "low": close - 200,
                "open": close,
                "volume": np.full(n, 5000000.0),
                "funding_rate": np.full(n, 0.0002),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = FundingCarryConfig(entry_threshold=0.0)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 유효 데이터 구간에서 시그널이 있어야 함
        valid = signals.direction[signals.direction != 0]
        assert len(valid) > 0


class TestShortMode:
    """Short mode 테스트."""

    def test_full_mode_has_shorts(self, preprocessed_df: pd.DataFrame) -> None:
        """FULL 모드에서 숏 시그널 존재."""
        config = FundingCarryConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        short_count = (signals.direction == Direction.SHORT).sum()
        assert short_count > 0

    def test_disabled_mode_no_shorts(self, preprocessed_df: pd.DataFrame) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = FundingCarryConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        short_count = (signals.direction == Direction.SHORT).sum()
        assert short_count == 0


class TestMissingColumns:
    """누락 컬럼 테스트."""

    def test_missing_avg_funding_rate(self) -> None:
        """avg_funding_rate 누락 시 에러."""
        df = pd.DataFrame(
            {
                "vol_scalar": [1.0, 1.1, 0.9],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)

    def test_missing_vol_scalar(self) -> None:
        """vol_scalar 누락 시 에러."""
        df = pd.DataFrame(
            {
                "avg_funding_rate": [0.001, -0.001, 0.002],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
