"""Tests for Flow Imbalance Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.flow_imbalance.config import FlowImbalanceConfig
from src.strategy.flow_imbalance.preprocessor import preprocess


@pytest.fixture
def sample_1h_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (1H, 200 bars)."""
    np.random.seed(42)
    n = 200

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
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        expected_cols = [
            "returns",
            "buy_ratio",
            "buy_vol",
            "sell_vol",
            "ofi",
            "vpin_proxy",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_buy_ratio_range(self, sample_1h_df: pd.DataFrame):
        """buy_ratio는 [0, 1] 범위."""
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        buy_ratio = result["buy_ratio"].dropna()
        assert (buy_ratio >= 0).all()
        assert (buy_ratio <= 1).all()

    def test_buy_sell_vol_sum(self, sample_1h_df: pd.DataFrame):
        """buy_vol + sell_vol ≈ volume."""
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        total = result["buy_vol"] + result["sell_vol"]
        np.testing.assert_allclose(
            total.values,
            result["volume"].values,
            rtol=1e-9,
        )

    def test_ofi_range(self, sample_1h_df: pd.DataFrame):
        """ofi는 [-1, 1] 범위 (normalized)."""
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        valid_ofi = result["ofi"].dropna()
        assert (valid_ofi >= -1.0 - 1e-9).all()
        assert (valid_ofi <= 1.0 + 1e-9).all()

    def test_vpin_proxy_positive(self, sample_1h_df: pd.DataFrame):
        """vpin_proxy는 비음수."""
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        valid = result["vpin_proxy"].dropna()
        assert (valid >= 0).all()

    def test_vol_scalar_positive(self, sample_1h_df: pd.DataFrame):
        """vol_scalar는 양수."""
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_immutability(self, sample_1h_df: pd.DataFrame):
        """원본 DataFrame은 수정되지 않아야 함."""
        config = FlowImbalanceConfig()
        original = sample_1h_df.copy()
        preprocess(sample_1h_df, config)

        pd.testing.assert_frame_equal(sample_1h_df, original)

    def test_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = FlowImbalanceConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_1h_df: pd.DataFrame):
        """출력 길이는 입력과 동일."""
        config = FlowImbalanceConfig()
        result = preprocess(sample_1h_df, config)

        assert len(result) == len(sample_1h_df)

    def test_bullish_bar_buy_ratio_high(self):
        """상승 bar에서 buy_ratio가 높아야 함."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [110.0],
                "low": [95.0],
                "close": [108.0],  # close가 high에 가까움
                "volume": [1000.0],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="1h"),
        )

        config = FlowImbalanceConfig()
        result = preprocess(df, config)

        # buy_ratio = (108 - 95) / (110 - 95) = 13/15 ≈ 0.867
        assert result["buy_ratio"].iloc[0] > 0.8
