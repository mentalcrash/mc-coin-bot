"""Tests for Z-Score MR Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.types import Direction
from src.strategy.zscore_mr.config import ShortMode, ZScoreMRConfig
from src.strategy.zscore_mr.preprocessor import preprocess
from src.strategy.zscore_mr.signal import generate_signals


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """평균회귀 패턴의 샘플 OHLCV DataFrame."""
    np.random.seed(42)
    n = 200

    base_price = 50000.0
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + noise - noise.mean()
    close = np.maximum(close, base_price * 0.8)

    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = ZScoreMRConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalStructure:
    """시그널 구조 테스트."""

    def test_returns_strategy_signals(self, processed_df: pd.DataFrame):
        """StrategySignals NamedTuple 반환."""
        signals = generate_signals(processed_df)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_bool_dtype(self, processed_df: pd.DataFrame):
        """entries는 bool Series."""
        signals = generate_signals(processed_df)
        assert signals.entries.dtype == bool

    def test_exits_bool_dtype(self, processed_df: pd.DataFrame):
        """exits는 bool Series."""
        signals = generate_signals(processed_df)
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame):
        """direction은 -1, 0, 1 값만."""
        config = ZScoreMRConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)
        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_same_length(self, processed_df: pd.DataFrame):
        """모든 시그널이 동일한 길이."""
        signals = generate_signals(processed_df)
        n = len(processed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_row_is_zero(self, processed_df: pd.DataFrame):
        """첫 행의 strength는 0 (shift로 인한 NaN -> 0)."""
        signals = generate_signals(processed_df)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame):
        """마지막 봉의 데이터 변경이 이전 봉의 시그널에 영향 없음."""
        config = ZScoreMRConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉 가격 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("zscore")] = 99.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vol_scalar")] = 99.0

        signals_after = generate_signals(modified_df, config)
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestShortMode:
    """ShortMode 테스트."""

    def test_disabled_no_shorts(self, processed_df: pd.DataFrame):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = ZScoreMRConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)
        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_has_both_directions(self, processed_df: pd.DataFrame):
        """FULL 모드에서 롱/숏 모두 존재."""
        config = ZScoreMRConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)
        has_long = (signals.direction == Direction.LONG).any()
        has_short = (signals.direction == Direction.SHORT).any()
        assert has_long
        assert has_short

    def test_hedge_only_suppresses_without_drawdown(self, processed_df: pd.DataFrame):
        """HEDGE_ONLY 모드에서 드로다운 없을 때 숏 억제."""
        # drawdown을 모두 0으로 설정 (드로다운 없음)
        modified_df = processed_df.copy()
        modified_df["drawdown"] = 0.0

        config = ZScoreMRConfig(short_mode=ShortMode.HEDGE_ONLY)
        signals = generate_signals(modified_df, config)
        assert (signals.direction >= 0).all()


class TestMeanReversionLogic:
    """평균회귀 로직 검증."""

    def test_extreme_positive_z_gives_short(self):
        """극단적 양의 z-score -> 숏 시그널."""
        n = 100
        index = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "zscore": [3.0] * n,
                "vol_scalar": [1.0] * n,
                "drawdown": [0.0] * n,
            },
            index=index,
        )

        config = ZScoreMRConfig(entry_z=2.0, exit_z=0.5, short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)

        # shift(1) 때문에 2번째 행부터 숏
        assert (signals.direction.iloc[1:] == Direction.SHORT).all()

    def test_extreme_negative_z_gives_long(self):
        """극단적 음의 z-score -> 롱 시그널."""
        n = 100
        index = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "zscore": [-3.0] * n,
                "vol_scalar": [1.0] * n,
                "drawdown": [0.0] * n,
            },
            index=index,
        )

        config = ZScoreMRConfig(entry_z=2.0, exit_z=0.5, short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)

        # shift(1) 때문에 2번째 행부터 롱
        assert (signals.direction.iloc[1:] == Direction.LONG).all()

    def test_z_near_zero_is_neutral(self):
        """z-score가 exit_z 미만이면 중립."""
        n = 100
        index = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "zscore": [0.1] * n,
                "vol_scalar": [1.0] * n,
                "drawdown": [0.0] * n,
            },
            index=index,
        )

        config = ZScoreMRConfig(entry_z=2.0, exit_z=0.5, short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)

        # shift(1) 때문에 2번째 행부터 중립
        assert (signals.direction.iloc[1:] == Direction.NEUTRAL).all()

    def test_ffill_holds_position(self):
        """entry~exit 사이에서 포지션이 유지됨 (ffill)."""
        n = 10
        index = pd.date_range("2024-01-01", periods=n, freq="D")
        # z가 -3(entry) -> -1.5(hold) -> -1.5(hold) -> 0.2(exit)
        zscore_values = [-3.0, -3.0, -1.5, -1.5, -1.5, 0.2, 0.2, 0.2, -3.0, -3.0]
        df = pd.DataFrame(
            {
                "zscore": zscore_values,
                "vol_scalar": [1.0] * n,
                "drawdown": [0.0] * n,
            },
            index=index,
        )

        config = ZScoreMRConfig(entry_z=2.0, exit_z=0.5, short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)

        # index 1: z_shifted=-3.0 -> long entry
        # index 2: z_shifted=-3.0 -> long hold
        # index 3: z_shifted=-1.5 -> between entry/exit -> ffill (long hold)
        # index 4: z_shifted=-1.5 -> ffill (long hold)
        # index 5: z_shifted=-1.5 -> ffill (long hold)
        # index 6: z_shifted=0.2 -> exit (neutral)
        assert signals.direction.iloc[1] == Direction.LONG
        assert signals.direction.iloc[2] == Direction.LONG
        assert signals.direction.iloc[3] == Direction.LONG
        assert signals.direction.iloc[4] == Direction.LONG
        assert signals.direction.iloc[5] == Direction.LONG
        assert signals.direction.iloc[6] == Direction.NEUTRAL

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)

    def test_vol_scalar_affects_strength(self, sample_ohlcv: pd.DataFrame):
        """vol_target이 strength 크기에 영향."""
        config_low = ZScoreMRConfig(vol_target=0.10)
        config_high = ZScoreMRConfig(vol_target=0.40)

        processed_low = preprocess(sample_ohlcv, config_low)
        processed_high = preprocess(sample_ohlcv, config_high)

        signals_low = generate_signals(processed_low, config_low)
        signals_high = generate_signals(processed_high, config_high)

        avg_abs_low = signals_low.strength.abs().mean()
        avg_abs_high = signals_high.strength.abs().mean()
        assert avg_abs_high > avg_abs_low
