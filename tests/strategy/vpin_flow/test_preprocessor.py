"""Tests for VPIN Flow Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vpin_flow.config import VPINFlowConfig
from src.strategy.vpin_flow.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "v_buy",
            "v_sell",
            "vpin",
            "flow_direction",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_vpin_bounded(self, sample_ohlcv_df: pd.DataFrame):
        """VPIN은 [0, 1] 범위."""
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_vpin = result["vpin"].dropna()
        assert len(valid_vpin) > 0
        assert (valid_vpin >= 0).all()
        assert (valid_vpin <= 1.0 + 1e-10).all()

    def test_v_buy_v_sell_sum_to_volume(self, sample_ohlcv_df: pd.DataFrame):
        """v_buy + v_sell == volume."""
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)

        total = result["v_buy"] + result["v_sell"]
        pd.testing.assert_series_equal(total, result["volume"], check_names=False, atol=1e-6)

    def test_v_buy_positive(self, sample_ohlcv_df: pd.DataFrame):
        """v_buy는 항상 >= 0."""
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert (result["v_buy"] >= 0).all()

    def test_v_sell_positive(self, sample_ohlcv_df: pd.DataFrame):
        """v_sell은 항상 >= 0."""
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert (result["v_sell"] >= 0).all()

    def test_flow_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        """flow_direction은 {-1, 0, 1}."""
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["flow_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_bullish_bar_high_buy_pct(self):
        """강한 양봉에서 v_buy > v_sell."""
        n = 50
        close = np.full(n, 110.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 112.0)
        low = np.full(n, 99.0)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 10000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = VPINFlowConfig()
        result = preprocess(df, config)

        # 강한 양봉이므로 v_buy > v_sell
        assert (result["v_buy"] > result["v_sell"]).all()

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = VPINFlowConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = VPINFlowConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = VPINFlowConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
