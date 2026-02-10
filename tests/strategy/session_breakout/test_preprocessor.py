"""Tests for Session Breakout Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.session_breakout.config import SessionBreakoutConfig
from src.strategy.session_breakout.preprocessor import preprocess


@pytest.fixture
def sample_1h_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (1H, 1000 bars = ~42일)."""
    np.random.seed(42)
    n = 1000

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)
    open_ = close + np.random.randn(n) * 0.3

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class TestPreprocess:
    """preprocess() 테스트."""

    def test_output_columns(self, sample_1h_df: pd.DataFrame):
        """출력 DataFrame에 필요한 컬럼이 존재하는지 확인."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        expected_cols = [
            "returns",
            "is_asian",
            "is_trade_window",
            "is_exit_hour",
            "asian_high",
            "asian_low",
            "asian_range",
            "range_pctl",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_asian_session_flags(self, sample_1h_df: pd.DataFrame):
        """Asian session 플래그가 올바르게 설정되는지 확인."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        # 00-07 UTC (end=8, exclusive) 에서 is_asian=True
        asian_hours = result.index.hour  # type: ignore[union-attr]
        is_asian: pd.Series = result["is_asian"]  # type: ignore[assignment]

        asian_bars = is_asian[asian_hours < 8]
        assert asian_bars.all()

        non_asian_bars = is_asian[asian_hours >= 8]
        assert not non_asian_bars.any()

    def test_trade_window_flags(self, sample_1h_df: pd.DataFrame):
        """Trade window 플래그가 올바르게 설정되는지 확인."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        hours = result.index.hour  # type: ignore[union-attr]
        is_trade: pd.Series = result["is_trade_window"]  # type: ignore[assignment]

        trade_bars = is_trade[(hours >= 8) & (hours < 20)]
        assert trade_bars.all()

    def test_exit_hour_flag(self, sample_1h_df: pd.DataFrame):
        """Exit hour 플래그가 올바르게 설정되는지 확인."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        hours = result.index.hour  # type: ignore[union-attr]
        is_exit: pd.Series = result["is_exit_hour"]  # type: ignore[assignment]

        exit_bars = is_exit[hours == 22]
        assert exit_bars.all()

        non_exit_bars = is_exit[hours != 22]
        assert not non_exit_bars.any()

    def test_asian_high_low(self, sample_1h_df: pd.DataFrame):
        """Asian high/low가 올바르게 계산되는지 확인."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        # ffill이 적용되어 Asian session 이후에도 값이 존재
        valid_high = result["asian_high"].dropna()
        valid_low = result["asian_low"].dropna()
        assert len(valid_high) > 0
        assert len(valid_low) > 0
        assert (valid_high >= valid_low).all()

    def test_asian_range_positive(self, sample_1h_df: pd.DataFrame):
        """asian_range는 항상 >= 0."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        valid_range = result["asian_range"].dropna()
        assert (valid_range >= 0).all()

    def test_vol_scalar_positive(self, sample_1h_df: pd.DataFrame):
        """vol_scalar는 양수."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_immutability(self, sample_1h_df: pd.DataFrame):
        """원본 DataFrame은 수정되지 않아야 함."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        original = sample_1h_df.copy()
        preprocess(sample_1h_df, config)

        pd.testing.assert_frame_equal(sample_1h_df, original)

    def test_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = SessionBreakoutConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_1h_df: pd.DataFrame):
        """출력 길이는 입력과 동일."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        result = preprocess(sample_1h_df, config)

        assert len(result) == len(sample_1h_df)
