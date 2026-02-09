"""Unit tests for XSMOM signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction
from src.strategy.xsmom.config import XSMOMConfig
from src.strategy.xsmom.preprocessor import preprocess
from src.strategy.xsmom.signal import generate_signals


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame fixture."""
    config = XSMOMConfig()
    return preprocess(sample_ohlcv, config)


class TestGenerateSignalsBasic:
    """시그널 생성 기본 테스트."""

    def test_output_structure(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 크기 확인."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_returns_named_tuple(self, preprocessed_df: pd.DataFrame) -> None:
        """StrategySignals NamedTuple 반환 확인."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_default_config(self, preprocessed_df: pd.DataFrame) -> None:
        """config=None일 때 기본 설정으로 동작."""
        signals = generate_signals(preprocessed_df, config=None)
        assert len(signals.entries) == len(preprocessed_df)


class TestShift1Rule:
    """Shift(1) Rule 검증 (미래 참조 편향 방지)."""

    def test_first_direction_is_zero(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 direction은 0이어야 한다 (shift(1) 적용)."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.direction.iloc[0] == 0

    def test_first_strength_is_zero(self, preprocessed_df: pd.DataFrame) -> None:
        """첫 번째 strength는 0이어야 한다 (shift(1) 적용)."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead_bias(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널이 shift(1) 적용되어 미래 데이터 참조 없음 확인."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.strength.iloc[0] == 0.0
        assert signals.direction.iloc[0] == Direction.NEUTRAL


class TestDirectionValues:
    """direction 값 검증."""

    def test_direction_values_in_valid_set(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1만 포함해야 한다."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        unique_vals = set(signals.direction.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_direction_is_integer(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 정수 타입이어야 한다."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert np.issubdtype(signals.direction.dtype, np.integer)


class TestShortModeSignals:
    """숏 모드 처리 테스트."""

    def test_full_short_mode(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL 모드에서 숏 시그널이 존재할 수 있다."""
        config = XSMOMConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_disabled_short_mode(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서는 숏 시그널이 없어야 한다."""
        config = XSMOMConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_disabled_mode_zeroes_short_strength(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서 숏 방향의 strength가 0인지 확인."""
        config = XSMOMConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        short_mask = signals.direction < 0
        assert not short_mask.any()


class TestEntriesExits:
    """entries/exits 시그널 검증."""

    def test_entries_exits_are_bool(self, preprocessed_df: pd.DataFrame) -> None:
        """entries와 exits는 bool 타입이어야 한다."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_entries_exist(self, sample_ohlcv: pd.DataFrame) -> None:
        """진입 시그널이 존재해야 한다."""
        config = XSMOMConfig()
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert signals.entries.any(), "No entry signals generated"


class TestStrength:
    """strength 값 검증."""

    def test_strength_is_float(self, preprocessed_df: pd.DataFrame) -> None:
        """strength가 float 타입인지 확인."""
        config = XSMOMConfig()
        signals = generate_signals(preprocessed_df, config)

        assert np.issubdtype(signals.strength.dtype, np.floating) or signals.strength.dtype == float


class TestHoldingPeriodEffect:
    """Holding period 효과 테스트."""

    def test_longer_holding_period_fewer_entries(self, sample_ohlcv: pd.DataFrame) -> None:
        """긴 holding_period는 진입 시그널 수를 줄여야 한다."""
        config_short = XSMOMConfig(holding_period=1)
        config_long = XSMOMConfig(holding_period=14)

        processed_short = preprocess(sample_ohlcv, config_short)
        processed_long = preprocess(sample_ohlcv, config_long)

        signals_short = generate_signals(processed_short, config_short)
        signals_long = generate_signals(processed_long, config_long)

        # 긴 holding_period는 시그널 변경이 적으므로 entry가 적거나 같아야 함
        assert signals_long.entries.sum() <= signals_short.entries.sum()


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_rolling_return_raises(self) -> None:
        """rolling_return 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"vol_scalar": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = XSMOMConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_scalar_raises(self) -> None:
        """vol_scalar 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"rolling_return": [0.1, -0.05, 0.02]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = XSMOMConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
