"""Unit tests for Donchian Ensemble signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig, ShortMode
from src.strategy.donchian_ensemble.preprocessor import preprocess
from src.strategy.donchian_ensemble.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame fixture."""
    config = DonchianEnsembleConfig()
    return preprocess(sample_ohlcv, config)


@pytest.fixture
def short_preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """짧은 lookback으로 전처리된 DataFrame fixture (더 많은 시그널 생성)."""
    config = DonchianEnsembleConfig(lookbacks=(5, 10, 20))
    return preprocess(sample_ohlcv, config)


class TestGenerateSignalsBasic:
    """시그널 생성 기본 테스트."""

    def test_generate_signals_basic(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 크기 확인."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_generate_signals_returns_named_tuple(self, preprocessed_df: pd.DataFrame) -> None:
        """StrategySignals NamedTuple 반환 확인."""
        config = DonchianEnsembleConfig()
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

    def test_first_direction_is_zero(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 direction은 0이어야 한다 (shift(1) 적용)."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.direction.iloc[0] == 0

    def test_first_strength_is_zero(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 strength는 0이어야 한다 (shift(1) 적용)."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead_bias(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널이 shift(1) 적용되어 미래 데이터 참조 없음 확인."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        # 첫 번째 값은 NaN -> 0 변환된 것이므로 0
        assert signals.strength.iloc[0] == 0.0
        assert signals.direction.iloc[0] == Direction.NEUTRAL


class TestShortMode:
    """숏 모드 처리 테스트."""

    def test_disabled_mode_no_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서는 숏 시그널이 없어야 한다."""
        config = DonchianEnsembleConfig(
            short_mode=ShortMode.DISABLED,
            lookbacks=(5, 10, 20),
        )
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_allows_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL 모드에서는 숏 시그널이 자유롭게 생성된다."""
        config = DonchianEnsembleConfig(
            short_mode=ShortMode.FULL,
            lookbacks=(5, 10, 20),
        )
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # FULL 모드에서는 direction에 -1이 포함될 수 있음
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_disabled_mode_zeroes_out_short_strength(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서 숏 방향의 strength가 0인지 확인."""
        config = DonchianEnsembleConfig(
            short_mode=ShortMode.DISABLED,
            lookbacks=(5, 10, 20),
        )
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # direction이 0인 곳의 strength도 0이어야 하거나 양수여야 함
        short_mask = signals.direction < 0
        assert not short_mask.any(), "DISABLED mode should have no short directions"


class TestEntriesExits:
    """entries/exits 시그널 검증."""

    def test_entries_exits_are_bool(self, preprocessed_df: pd.DataFrame) -> None:
        """entries와 exits는 bool 타입이어야 한다."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_entries_exist_with_short_lookbacks(self, short_preprocessed_df: pd.DataFrame) -> None:
        """짧은 lookback 데이터에서 진입 시그널이 존재해야 한다."""
        config = DonchianEnsembleConfig(lookbacks=(5, 10, 20))
        signals = generate_signals(short_preprocessed_df, config)

        assert signals.entries.any(), "No entry signals generated"

    def test_no_simultaneous_entry_exit_without_reversal(
        self, short_preprocessed_df: pd.DataFrame
    ) -> None:
        """방향 전환이 아닌 경우 동시에 entry와 exit이 True이면 안 된다."""
        config = DonchianEnsembleConfig(lookbacks=(5, 10, 20))
        signals = generate_signals(short_preprocessed_df, config)

        both_true = signals.entries & signals.exits
        if both_true.any():
            # 방향 전환 인덱스 확인
            prev_dir = signals.direction.shift(1).fillna(0)
            reversal = signals.direction * prev_dir < 0
            non_reversal_both = both_true & ~reversal
            assert not non_reversal_both.any()


class TestDirectionValues:
    """direction 값 검증."""

    def test_direction_values_in_valid_set(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1만 포함해야 한다."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        unique_vals = set(signals.direction.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_direction_is_integer(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 정수 타입이어야 한다."""
        config = DonchianEnsembleConfig()
        signals = generate_signals(preprocessed_df, config)

        assert np.issubdtype(signals.direction.dtype, np.integer)


class TestSignalMissingColumns:
    """시그널 생성 시 필수 컬럼 누락 테스트."""

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"vol_scalar": [1.0, 2.0, 3.0], "dc_upper_5": [10, 11, 12], "dc_lower_5": [8, 9, 10]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig(lookbacks=(5,))
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_scalar_raises(self) -> None:
        """vol_scalar 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 200, 300], "dc_upper_5": [10, 11, 12], "dc_lower_5": [8, 9, 10]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig(lookbacks=(5,))
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_dc_channel_raises(self) -> None:
        """dc_upper/dc_lower 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 200, 300], "vol_scalar": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig(lookbacks=(5, 10))
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
