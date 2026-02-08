"""Tests for Vol-Regime Adaptive signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.types import Direction
from src.strategy.vol_regime.config import ShortMode, VolRegimeConfig
from src.strategy.vol_regime.preprocessor import preprocess
from src.strategy.vol_regime.signal import generate_signals


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (350일)."""
    np.random.seed(42)
    n = 350

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
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = VolRegimeConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalStructure:
    """시그널 구조 테스트."""

    def test_signal_fields(self, preprocessed_df: pd.DataFrame):
        """시그널에 entries, exits, direction, strength 필드 존재."""
        signals = generate_signals(preprocessed_df)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_signal_types(self, preprocessed_df: pd.DataFrame):
        """시그널 타입 확인."""
        signals = generate_signals(preprocessed_df)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

    def test_direction_values(self, preprocessed_df: pd.DataFrame):
        """direction은 -1, 0, 1 중 하나."""
        signals = generate_signals(preprocessed_df)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_signal_length(self, preprocessed_df: pd.DataFrame):
        """시그널 길이가 입력 DataFrame과 동일."""
        signals = generate_signals(preprocessed_df)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_first_row_is_neutral(self, preprocessed_df: pd.DataFrame):
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        signals = generate_signals(preprocessed_df)

        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0

    def test_signal_uses_previous_bar(self, preprocessed_df: pd.DataFrame):
        """시그널이 전봉 데이터 기반인지 확인.

        regime_strength를 직접 확인하여 shift(1)이 적용되었는지 검증.
        FULL 모드를 사용하여 숏 모드 처리가 방향을 변경하지 않도록 함.
        """
        config = VolRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        # regime_strength의 shift(1)과 strength가 부호 일치하는지 확인
        # (NaN 제외, strength != 0인 구간)
        regime_strength: pd.Series = preprocessed_df["regime_strength"]  # type: ignore[assignment]
        shifted = regime_strength.shift(1)

        # shift(1) 적용 후 NaN이 아닌 곳에서 부호가 일치해야 함
        valid_mask = shifted.notna() & (shifted != 0)
        if valid_mask.any():
            expected_dir = np.sign(shifted[valid_mask])
            actual_dir = signals.direction[valid_mask]
            np.testing.assert_array_equal(actual_dir.values, expected_dir.astype(int).values)


class TestShortMode:
    """숏 모드 테스트."""

    def test_disabled_mode_no_shorts(self, preprocessed_df: pd.DataFrame):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = VolRegimeConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_allows_shorts(self, preprocessed_df: pd.DataFrame):
        """FULL 모드에서 숏 시그널 허용."""
        config = VolRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        # FULL 모드에서는 숏 시그널이 존재할 수 있음
        # (데이터에 따라 없을 수도 있지만 direction -1이 허용됨)
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_hedge_only_mode(self, preprocessed_df: pd.DataFrame):
        """HEDGE_ONLY 모드에서 드로다운 임계값에 따라 숏 제어."""
        config = VolRegimeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.07,
            hedge_strength_ratio=0.8,
        )
        signals = generate_signals(preprocessed_df, config)

        # 시그널이 생성되는지 확인
        assert len(signals.entries) == len(preprocessed_df)

    def test_hedge_only_needs_drawdown_column(self):
        """HEDGE_ONLY 모드에서 drawdown 컬럼 없으면 에러."""
        config = VolRegimeConfig(short_mode=ShortMode.HEDGE_ONLY)

        df = pd.DataFrame(
            {
                "regime_strength": [0.5, -0.3, 0.1],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
