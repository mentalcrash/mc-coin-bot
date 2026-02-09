"""Tests for Vol Structure Regime signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.types import Direction
from src.strategy.vol_structure.config import ShortMode, VolStructureConfig
from src.strategy.vol_structure.preprocessor import preprocess
from src.strategy.vol_structure.signal import generate_signals


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200

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
def default_config() -> VolStructureConfig:
    """기본 Vol Structure Config."""
    return VolStructureConfig()


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_signal_output_structure(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VolStructureConfig,
    ):
        """시그널에 entries, exits, direction, strength 필드 존재."""
        processed = preprocess(sample_ohlcv_df, default_config)
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
        sample_ohlcv_df: pd.DataFrame,
        default_config: VolStructureConfig,
    ):
        """direction은 -1, 0, 1 중 하나."""
        config = VolStructureConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )


class TestRegimeClassification:
    """Regime 분류 테스트."""

    def test_expansion_regime(self):
        """Expansion regime: 높은 vol_ratio + 높은 abs(norm_mom) → non-zero direction."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        # 합성 데이터: expansion 조건 충족
        config = VolStructureConfig(
            expansion_vol_ratio=1.2,
            expansion_mom_threshold=1.5,
            short_mode=ShortMode.FULL,
        )

        df = pd.DataFrame(
            {
                "vol_ratio": np.full(n, 1.8),  # > 1.2
                "norm_momentum": np.full(n, 2.5),  # abs > 1.5
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # shift(1)로 인해 첫 번째 행은 0이지만, 나머지는 non-zero
        valid_direction = signals.direction.iloc[2:]
        non_zero = (valid_direction != 0).sum()
        assert non_zero > 0, "Expansion regime should produce non-zero direction"

    def test_contraction_regime(self):
        """Contraction regime: 낮은 vol_ratio + 낮은 abs(norm_mom) → zero direction."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        config = VolStructureConfig(
            contraction_vol_ratio=0.8,
            contraction_mom_threshold=0.5,
            short_mode=ShortMode.FULL,
        )

        df = pd.DataFrame(
            {
                "vol_ratio": np.full(n, 0.5),  # < 0.8
                "norm_momentum": np.full(n, 0.2),  # abs < 0.5
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # Contraction: 모든 direction == 0
        valid_direction = signals.direction.iloc[2:]
        zero_count = (valid_direction == 0).sum()
        assert zero_count == len(valid_direction), (
            "Contraction regime should produce zero direction"
        )


class TestShortModeSignals:
    """숏 모드별 시그널 테스트."""

    def test_short_mode_disabled(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = VolStructureConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_short_mode_hedge_only(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """HEDGE_ONLY 모드에서 드로다운 임계값에 따라 숏 제어."""
        config = VolStructureConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.07,
            hedge_strength_ratio=0.8,
        )
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # 시그널이 생성되는지 확인
        assert len(signals.entries) == len(processed)

    def test_full_mode_allows_shorts(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """FULL 모드에서 숏 시그널 허용."""
        config = VolStructureConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_no_lookahead(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        config = VolStructureConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0

    def test_signal_uses_previous_bar(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """시그널이 전봉 데이터 기반인지 확인."""
        config = VolStructureConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # vol_ratio, norm_momentum의 shift(1)과 시그널이 일관성 있는지 확인
        vol_ratio: pd.Series = processed["vol_ratio"]  # type: ignore[assignment]
        norm_mom: pd.Series = processed["norm_momentum"]  # type: ignore[assignment]

        # shift(1) 값이 NaN이 아닌 곳에서 검증
        shifted_vr = vol_ratio.shift(1)
        shifted_nm = norm_mom.shift(1)

        valid_mask = shifted_vr.notna() & shifted_nm.notna()
        if valid_mask.any():
            # expansion 조건인 곳에서 direction이 norm_mom 부호와 일치
            expansion_mask = (
                (shifted_vr > config.expansion_vol_ratio)
                & (shifted_nm.abs() > config.expansion_mom_threshold)
                & valid_mask
            )
            if expansion_mask.any():
                expected_dir = np.sign(shifted_nm[expansion_mask])
                actual_dir = signals.direction[expansion_mask]
                np.testing.assert_array_equal(
                    actual_dir.values,
                    expected_dir.astype(int).values,
                )


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
