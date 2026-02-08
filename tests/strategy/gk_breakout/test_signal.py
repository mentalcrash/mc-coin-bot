"""Tests for GK Breakout Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.gk_breakout.config import GKBreakoutConfig, ShortMode
from src.strategy.gk_breakout.preprocessor import preprocess
from src.strategy.gk_breakout.signal import generate_signals


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """추세 + 횡보 패턴의 샘플 OHLCV DataFrame."""
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
    config = GKBreakoutConfig()
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
        config = GKBreakoutConfig(short_mode=ShortMode.FULL)
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

    def test_no_nan_in_output(self, processed_df: pd.DataFrame):
        """출력에 NaN 없음."""
        signals = generate_signals(processed_df)
        assert signals.entries.notna().all()
        assert signals.exits.notna().all()
        assert signals.direction.notna().all()
        assert signals.strength.notna().all()


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_row_is_zero(self, processed_df: pd.DataFrame):
        """첫 행의 strength는 0 (shift로 인한 NaN -> 0)."""
        signals = generate_signals(processed_df)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame):
        """마지막 봉의 데이터 변경이 그 전 봉의 시그널에 영향 없음."""
        config = GKBreakoutConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉 가격 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("close")] = 999999.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vol_ratio")] = 99.0

        signals_after = generate_signals(modified_df, config)
        # 두 번째 마지막 행은 영향받지 않아야 함
        assert signals_after.strength.iloc[-2] == strength_second_last

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)


class TestShortMode:
    """ShortMode 테스트."""

    def test_disabled_no_shorts(self, processed_df: pd.DataFrame):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = GKBreakoutConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)
        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_allows_shorts(self, processed_df: pd.DataFrame):
        """FULL 모드에서 direction에 -1, 0, 1 모두 가능."""
        config = GKBreakoutConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)
        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_hedge_only_requires_drawdown(self, processed_df: pd.DataFrame):
        """HEDGE_ONLY 모드에서 drawdown 컬럼 필요."""
        config = GKBreakoutConfig(short_mode=ShortMode.HEDGE_ONLY)
        # drawdown 컬럼이 있는 processed_df에서는 정상 작동
        signals = generate_signals(processed_df, config)
        assert len(signals.entries) == len(processed_df)

    def test_hedge_only_missing_drawdown_raises(self):
        """HEDGE_ONLY 모드에서 drawdown 컬럼 없으면 ValueError."""
        config = GKBreakoutConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = pd.DataFrame(
            {
                "vol_ratio": [0.5, 0.6, 0.7],
                "close": [100, 101, 102],
                "dc_upper": [105, 106, 107],
                "dc_lower": [95, 94, 93],
                "vol_scalar": [1.0, 1.0, 1.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)


class TestBreakoutLogic:
    """돌파 로직 검증."""

    def test_vol_scalar_affects_strength(self):
        """vol_target이 strength 크기에 영향.

        횡보(좁은 range) 후 강한 추세 돌파가 발생하는 데이터를 생성하여
        compression -> breakout 패턴을 확실하게 만듭니다.
        """
        np.random.seed(42)
        n = 200

        # Phase 1 (0-79): 매우 좁은 횡보 → 변동성 압축 유발
        # Phase 2 (80-120): 강한 상승 추세 → Donchian 상단 돌파
        # Phase 3 (121-200): 이후 구간
        base_price = 50000.0
        close = np.full(n, base_price)

        # 횡보: 아주 작은 변동
        for i in range(1, 80):
            close[i] = close[i - 1] + np.random.randn() * 10

        # 급등: 매일 +800씩 (Donchian 채널 돌파 보장)
        for i in range(80, 120):
            close[i] = close[i - 1] + 800 + np.random.randn() * 50

        # 이후 소폭 횡보
        for i in range(120, n):
            close[i] = close[i - 1] + np.random.randn() * 50

        # OHLC 생성 — 횡보 구간은 range 극소화, 돌파 구간은 큰 range
        high = close.copy()
        low = close.copy()
        open_ = close.copy()
        for i in range(n):
            if i < 80:
                # 횡보: 극히 좁은 range → GK variance 최소화
                high[i] = close[i] + 5
                low[i] = close[i] - 5
                open_[i] = close[i] + np.random.randn() * 2
            else:
                # 추세: 넓은 range
                high[i] = close[i] + 400
                low[i] = close[i] - 200
                open_[i] = close[i] - 300

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n) * 1000,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config_low = GKBreakoutConfig(
            vol_target=0.10, short_mode=ShortMode.FULL, compression_threshold=0.99
        )
        config_high = GKBreakoutConfig(
            vol_target=0.50, short_mode=ShortMode.FULL, compression_threshold=0.99
        )

        processed_low = preprocess(df, config_low)
        processed_high = preprocess(df, config_high)

        signals_low = generate_signals(processed_low, config_low)
        signals_high = generate_signals(processed_high, config_high)

        # 적어도 하나의 진입이 발생해야 함
        assert signals_low.entries.any() or signals_high.entries.any(), (
            "No breakout entries generated — test data needs stronger pattern"
        )

        avg_abs_low = signals_low.strength.abs().mean()
        avg_abs_high = signals_high.strength.abs().mean()
        assert avg_abs_high > avg_abs_low

    def test_compression_sensitivity(self, sample_ohlcv: pd.DataFrame):
        """compression_threshold가 낮으면 압축 조건이 더 엄격해짐."""
        config_loose = GKBreakoutConfig(compression_threshold=0.95, short_mode=ShortMode.FULL)
        config_strict = GKBreakoutConfig(compression_threshold=0.40, short_mode=ShortMode.FULL)

        processed_loose = preprocess(sample_ohlcv, config_loose)
        processed_strict = preprocess(sample_ohlcv, config_strict)

        signals_loose = generate_signals(processed_loose, config_loose)
        signals_strict = generate_signals(processed_strict, config_strict)

        # 더 느슨한 threshold는 더 많은 진입을 허용
        entries_loose = int(signals_loose.entries.sum())
        entries_strict = int(signals_strict.entries.sum())
        assert entries_loose >= entries_strict
