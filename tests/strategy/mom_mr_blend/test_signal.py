"""Unit tests for Mom-MR Blend signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.mom_mr_blend.config import MomMrBlendConfig, ShortMode
from src.strategy.mom_mr_blend.preprocessor import preprocess
from src.strategy.mom_mr_blend.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame fixture."""
    config = MomMrBlendConfig()
    return preprocess(sample_ohlcv, config)


class TestGenerateSignalsBasic:
    """시그널 생성 기본 테스트."""

    def test_generate_signals_basic(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 타입/크기 확인."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_generate_signals_returns_named_tuple(self, preprocessed_df: pd.DataFrame) -> None:
        """StrategySignals NamedTuple 반환 확인."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_generate_signals_default_config(self, preprocessed_df: pd.DataFrame) -> None:
        """config=None일 때 기본 설정으로 동작."""
        signals = generate_signals(preprocessed_df, config=None)
        assert len(signals.entries) == len(preprocessed_df)


class TestShift1Rule:
    """Shift(1) Rule 검증 (미래 참조 편향 방지)."""

    def test_shift1_rule(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 시그널은 0이어야 한다 (shift(1) 적용)."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead_bias(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널이 shift(1) 적용되어 미래 데이터 참조 없음 확인."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.strength.iloc[0] == 0.0
        assert signals.direction.iloc[0] == Direction.NEUTRAL


class TestShortMode:
    """ShortMode 처리 테스트."""

    def test_disabled_mode_no_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서는 숏 시그널이 없어야 한다."""
        config = MomMrBlendConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_allows_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL 모드에서는 숏 시그널이 자유롭게 생성된다."""
        config = MomMrBlendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # FULL 모드에서는 direction에 -1이 포함될 수 있음
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_hedge_only_mode_suppresses_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서도 숏이 억제된다 (drawdown 미확인 시)."""
        config = MomMrBlendConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # HEDGE_ONLY에서 drawdown 없으면 숏 억제
        assert (signals.direction >= 0).all()


class TestBlendLogic:
    """블렌딩 로직 검증."""

    def test_blend_produces_bounded_direction(self, preprocessed_df: pd.DataFrame) -> None:
        """블렌드 결과의 direction이 {-1, 0, 1} 범위."""
        config = MomMrBlendConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_mom_only_weight(self, sample_ohlcv: pd.DataFrame) -> None:
        """mom_weight=1.0, mr_weight=0.0일 때 순수 모멘텀 시그널."""
        config = MomMrBlendConfig(
            mom_weight=1.0,
            mr_weight=0.0,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # 순수 모멘텀이므로 시그널이 존재해야 함
        assert len(signals.direction) == len(sample_ohlcv)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_mr_only_weight(self, sample_ohlcv: pd.DataFrame) -> None:
        """mom_weight=0.0, mr_weight=1.0일 때 순수 평균회귀 시그널."""
        config = MomMrBlendConfig(
            mom_weight=0.0,
            mr_weight=1.0,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert len(signals.direction) == len(sample_ohlcv)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})


class TestEntriesExits:
    """entries/exits 시그널 검증."""

    def test_entries_exits_are_bool(self, preprocessed_df: pd.DataFrame) -> None:
        """entries와 exits는 bool 타입이어야 한다."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_entries_exist(self, preprocessed_df: pd.DataFrame) -> None:
        """충분한 데이터에서 진입 시그널이 존재해야 한다."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.any(), "No entry signals generated"

    def test_no_simultaneous_entry_exit(self, preprocessed_df: pd.DataFrame) -> None:
        """동시에 entry와 exit이 True인 경우 반드시 reversal이어야 한다."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        both_true = signals.entries & signals.exits
        if both_true.any():
            prev_dir = signals.direction.shift(1).fillna(0)
            reversal = signals.direction * prev_dir < 0
            non_reversal_both = both_true & ~reversal
            assert not non_reversal_both.any()


class TestDirectionValues:
    """direction 값 검증."""

    def test_direction_values(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1만 포함해야 한다."""
        config = MomMrBlendConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        unique_vals = set(signals.direction.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_direction_is_integer(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 정수 타입이어야 한다."""
        config = MomMrBlendConfig()
        signals = generate_signals(preprocessed_df, config)

        assert np.issubdtype(signals.direction.dtype, np.integer)


class TestSignalMissingColumns:
    """시그널 생성 시 필수 컬럼 누락 테스트."""

    def test_missing_mom_zscore_raises(self) -> None:
        """mom_zscore 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"mr_zscore": [1.0, 2.0, 3.0], "vol_scalar": [0.5, 0.5, 0.5]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = MomMrBlendConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_mr_zscore_raises(self) -> None:
        """mr_zscore 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"mom_zscore": [0.01, 0.02, -0.01], "vol_scalar": [0.5, 0.5, 0.5]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = MomMrBlendConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_scalar_raises(self) -> None:
        """vol_scalar 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"mom_zscore": [0.01, 0.02, -0.01], "mr_zscore": [1.0, -1.0, 0.5]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = MomMrBlendConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
