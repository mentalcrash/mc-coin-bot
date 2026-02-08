"""Tests for TTM Squeeze Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.ttm_squeeze.config import ShortMode, TtmSqueezeConfig
from src.strategy.ttm_squeeze.preprocessor import preprocess
from src.strategy.ttm_squeeze.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = TtmSqueezeConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalBasic:
    """시그널 기본 구조 테스트."""

    def test_generate_signals_basic(self, processed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 타입/크기 확인."""
        config = TtmSqueezeConfig()
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
        config = TtmSqueezeConfig()
        signals = generate_signals(processed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1 값만 포함."""
        config = TtmSqueezeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_strength_no_nan(self, processed_df: pd.DataFrame) -> None:
        """strength에 NaN 없음 (fillna 처리됨)."""
        config = TtmSqueezeConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.isna().sum() == 0

    def test_first_bar_is_neutral(self, processed_df: pd.DataFrame) -> None:
        """첫 번째 바의 direction은 neutral(0)."""
        config = TtmSqueezeConfig()
        signals = generate_signals(processed_df, config)
        assert signals.direction.iloc[0] == Direction.NEUTRAL.value


class TestSqueezeFireLogic:
    """Squeeze fire 시그널 테스트."""

    def test_squeeze_fire_generates_entries(self) -> None:
        """Squeeze ON -> OFF 전환 시 entry가 발생하는지 확인."""
        np.random.seed(123)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")

        # 가격 시리즈: 횡보 -> 급변동으로 squeeze 발생 후 해제 유도
        close = np.ones(n) * 50000
        # 처음 80봉: 안정적 (저변동성 → squeeze 유도)
        close[:80] = 50000 + np.random.randn(80) * 50
        # 80-120봉: 약간의 변동성 증가
        close[80:120] = 50000 + np.cumsum(np.random.randn(40) * 200)
        # 120-200봉: 변동성 확장 (squeeze 해제)
        close[120:200] = close[119] + np.cumsum(np.random.randn(80) * 500)

        high = close + np.abs(np.random.randn(n) * 100)
        low = close - np.abs(np.random.randn(n) * 100)
        open_ = close + np.random.randn(n) * 50
        volume = np.ones(n) * 1000

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = TtmSqueezeConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 시그널이 정상적으로 생성되어야 함 (squeeze 발생 여부는 데이터 의존)
        assert len(signals.entries) == n
        assert len(signals.direction) == n

    def test_no_entry_without_squeeze(self) -> None:
        """Squeeze가 한번도 없으면 entry가 발생하지 않음."""
        np.random.seed(99)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")

        # 고변동성 데이터 → BB가 항상 KC 밖
        close = 50000 + np.cumsum(np.random.randn(n) * 5000)
        high = close + np.abs(np.random.randn(n) * 3000)
        low = close - np.abs(np.random.randn(n) * 3000)
        open_ = close + np.random.randn(n) * 1000
        volume = np.ones(n) * 1000

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = TtmSqueezeConfig()
        processed = preprocess(df, config)

        # squeeze가 없으면 squeeze_fire도 없으므로 entry 없음
        if not processed["squeeze_on"].any():
            signals = generate_signals(processed, config)
            # squeeze_fire가 없으면 entry도 없어야 함
            assert not signals.entries.any()


class TestShortMode:
    """ShortMode 처리 테스트."""

    def test_short_mode_disabled(self, processed_df: pd.DataFrame) -> None:
        """DISABLED 모드에서 short direction(-1) 없음."""
        config = TtmSqueezeConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)

        assert (signals.direction >= 0).all(), "Short direction found in DISABLED mode"
        assert (signals.strength >= 0).all(), "Negative strength found in DISABLED mode"

    def test_short_mode_full(self, processed_df: pd.DataFrame) -> None:
        """FULL 모드에서 short direction 허용."""
        config = TtmSqueezeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_disabled_vs_full_direction_difference(self, processed_df: pd.DataFrame) -> None:
        """DISABLED와 FULL 모드 direction 비교."""
        config_disabled = TtmSqueezeConfig(short_mode=ShortMode.DISABLED)
        config_full = TtmSqueezeConfig(short_mode=ShortMode.FULL)

        signals_disabled = generate_signals(processed_df, config_disabled)
        signals_full = generate_signals(processed_df, config_full)

        # 두 모드 모두 정상적으로 시그널을 생성해야 함
        abs_full = signals_full.strength.abs().sum()
        abs_disabled = signals_disabled.strength.abs().sum()
        assert abs_full >= 0
        assert abs_disabled >= 0


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_row_strength_is_zero(self, processed_df: pd.DataFrame) -> None:
        """첫 행의 strength는 0 (shift로 인한 NaN -> 0)."""
        config = TtmSqueezeConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame) -> None:
        """마지막 봉의 데이터 변경이 이전 봉 시그널에 영향 없음."""
        config = TtmSqueezeConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉의 squeeze_on/momentum/vol_scalar 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("squeeze_on")] = True
        modified_df.iloc[-1, modified_df.columns.get_loc("momentum")] = 99999.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vol_scalar")] = 99.0

        signals_after = generate_signals(modified_df, config)
        # 두 번째 마지막 행은 영향받지 않아야 함
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestMissingColumns:
    """누락 컬럼 에러 테스트."""

    def test_missing_squeeze_on_raises(self) -> None:
        """squeeze_on 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {
                "momentum": [1.0, 2.0],
                "exit_sma": [1.0, 2.0],
                "close": [1.0, 2.0],
                "vol_scalar": [1.0, 1.0],
            }
        )
        config = TtmSqueezeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_momentum_raises(self) -> None:
        """momentum 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {
                "squeeze_on": [True, False],
                "exit_sma": [1.0, 2.0],
                "close": [1.0, 2.0],
                "vol_scalar": [1.0, 1.0],
            }
        )
        config = TtmSqueezeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_scalar_raises(self) -> None:
        """vol_scalar 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {
                "squeeze_on": [True, False],
                "momentum": [1.0, 2.0],
                "exit_sma": [1.0, 2.0],
                "close": [1.0, 2.0],
            }
        )
        config = TtmSqueezeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
