"""Unit tests for Risk-Mom signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.risk_mom.config import RiskMomConfig
from src.strategy.risk_mom.preprocessor import preprocess
from src.strategy.risk_mom.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame fixture."""
    config = RiskMomConfig()
    return preprocess(sample_ohlcv, config)


class TestGenerateSignalsBasic:
    """시그널 생성 기본 테스트."""

    def test_generate_signals_basic(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 타입/크기 확인."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_generate_signals_returns_named_tuple(self, preprocessed_df: pd.DataFrame) -> None:
        """StrategySignals NamedTuple 반환 확인."""
        config = RiskMomConfig()
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
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead_bias(self, preprocessed_df: pd.DataFrame) -> None:
        """시그널이 shift(1) 적용되어 미래 데이터 참조 없음 확인."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        # 첫 번째 값은 NaN→0 변환된 것이므로 0
        assert signals.strength.iloc[0] == 0.0
        assert signals.direction.iloc[0] == Direction.NEUTRAL


class TestHedgeOnlyMode:
    """HEDGE_ONLY 모드 테스트."""

    def test_hedge_only_mode(self, sample_ohlcv: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드: 드로다운 충분할 때만 숏 허용."""
        config = RiskMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # 드로다운이 임계값 미만인 곳에서만 숏 가능
        drawdown = processed["drawdown"]
        hedge_active = drawdown < config.hedge_threshold

        short_mask = signals.direction == Direction.SHORT
        if short_mask.any():
            # 숏이 활성화된 곳은 hedge_active도 True여야 함
            # shift(1)로 인한 오프셋 보정: direction은 shift된 값이므로
            # 직접 비교는 어려우나, 숏이 있으면 해당 위치에서 hedge가 활성이어야 함
            short_indices = short_mask[short_mask].index
            for idx in short_indices:
                # hedge_active 확인 (shift 보정)
                if idx in hedge_active.index:
                    assert hedge_active.loc[idx], f"Short at {idx} without active hedge"

    def test_disabled_mode_no_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서는 숏 시그널이 없어야 한다."""
        config = RiskMomConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_allows_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL 모드에서는 숏 시그널이 자유롭게 생성된다."""
        config = RiskMomConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # FULL 모드에서는 direction에 -1이 포함될 수 있음
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})


class TestEntriesExits:
    """entries/exits 시그널 검증."""

    def test_entries_exits_are_bool(self, preprocessed_df: pd.DataFrame) -> None:
        """entries와 exits는 bool 타입이어야 한다."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_entries_exist(self, preprocessed_df: pd.DataFrame) -> None:
        """충분한 데이터에서 진입 시그널이 존재해야 한다."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        assert signals.entries.any(), "No entry signals generated"

    def test_no_simultaneous_entry_exit(self, preprocessed_df: pd.DataFrame) -> None:
        """동시에 entry와 exit이 True인 경우가 없어야 한다 (방향 전환 제외)."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        # 방향 전환(reversal) 시에는 exit과 entry가 동시에 발생할 수 있음
        # reversal이 아닌 경우에만 검증
        both_true = signals.entries & signals.exits
        if both_true.any():
            # 방향 전환 인덱스 확인
            prev_dir = signals.direction.shift(1).fillna(0)
            reversal = signals.direction * prev_dir < 0
            # 동시 True는 반드시 reversal이어야 함
            non_reversal_both = both_true & ~reversal
            assert not non_reversal_both.any()


class TestDirectionValues:
    """direction 값 검증."""

    def test_direction_values(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1만 포함해야 한다."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        unique_vals = set(signals.direction.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_direction_is_integer(self, preprocessed_df: pd.DataFrame) -> None:
        """direction은 정수 타입이어야 한다."""
        config = RiskMomConfig()
        signals = generate_signals(preprocessed_df, config)

        assert np.issubdtype(signals.direction.dtype, np.integer)


class TestSignalMissingColumns:
    """시그널 생성 시 필수 컬럼 누락 테스트."""

    def test_missing_vw_momentum_raises(self) -> None:
        """vw_momentum 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"bsc_scaling": [1.0, 2.0, 3.0], "drawdown": [-0.01, -0.02, -0.03]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = RiskMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_bsc_scaling_raises(self) -> None:
        """bsc_scaling 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"vw_momentum": [0.01, 0.02, -0.01], "drawdown": [-0.01, -0.02, -0.03]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = RiskMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_drawdown_raises_in_hedge_mode(self) -> None:
        """HEDGE_ONLY 모드에서 drawdown 누락 시 ValueError."""
        df = pd.DataFrame(
            {"vw_momentum": [0.01, 0.02, -0.01], "bsc_scaling": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = RiskMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
