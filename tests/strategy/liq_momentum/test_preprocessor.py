"""Tests for Liquidity-Adjusted Momentum Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.liq_momentum.config import LiqMomentumConfig
from src.strategy.liq_momentum.preprocessor import preprocess


@pytest.fixture
def sample_1h_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (1H, 1000 bars, 주말 포함)."""
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
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        expected_cols = [
            "returns",
            "rel_vol",
            "amihud",
            "amihud_pctl",
            "liq_state",
            "is_weekend",
            "mom_signal",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_liq_state_values(self, sample_1h_df: pd.DataFrame):
        """liq_state는 {-1, 0, 1} 중 하나."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        valid = result["liq_state"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_is_weekend_flag(self, sample_1h_df: pd.DataFrame):
        """is_weekend이 주말에 True."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        # 주말 확인
        weekend_mask = result.index.dayofweek >= 5  # type: ignore[union-attr]
        is_weekend: pd.Series = result["is_weekend"]  # type: ignore[assignment]

        pd.testing.assert_series_equal(
            is_weekend,
            pd.Series(weekend_mask, index=result.index, name="is_weekend"),
        )

    def test_mom_signal_values(self, sample_1h_df: pd.DataFrame):
        """mom_signal은 {-1, 0, 1} 중 하나."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        valid = result["mom_signal"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_rel_vol_positive(self, sample_1h_df: pd.DataFrame):
        """rel_vol은 양수."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        valid = result["rel_vol"].dropna()
        assert (valid > 0).all()

    def test_amihud_pctl_range(self, sample_1h_df: pd.DataFrame):
        """amihud_pctl은 0~1 범위."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        valid = result["amihud_pctl"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_immutability(self, sample_1h_df: pd.DataFrame):
        """원본 DataFrame은 수정되지 않아야 함."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        original = sample_1h_df.copy()
        preprocess(sample_1h_df, config)

        pd.testing.assert_frame_equal(sample_1h_df, original)

    def test_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = LiqMomentumConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_1h_df: pd.DataFrame):
        """출력 길이는 입력과 동일."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        assert len(result) == len(sample_1h_df)

    def test_vol_scalar_positive(self, sample_1h_df: pd.DataFrame):
        """vol_scalar는 양수."""
        config = LiqMomentumConfig(
            rel_vol_window=48,
            amihud_pctl_window=96,
            amihud_window=12,
        )
        result = preprocess(sample_1h_df, config)

        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
