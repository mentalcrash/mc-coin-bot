"""Tests for VWAP Disposition Momentum Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwap_disposition.config import VWAPDispositionConfig
from src.strategy.vwap_disposition.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 800
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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "vwap",
            "cgo",
            "volume_ratio",
            "mom_direction",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_vwap_calculation_correctness(self, sample_ohlcv_df: pd.DataFrame):
        """VWAP 계산 정확성: sum(close*vol)/sum(vol) 검증."""
        config = VWAPDispositionConfig(vwap_window=100)
        result = preprocess(sample_ohlcv_df, config)

        # 수동 계산과 비교 (특정 인덱스)
        idx = 150
        close = sample_ohlcv_df["close"].iloc[idx - 100 + 1 : idx + 1].values
        volume = sample_ohlcv_df["volume"].iloc[idx - 100 + 1 : idx + 1].values
        expected_vwap = np.sum(close * volume) / np.sum(volume)

        actual_vwap = result["vwap"].iloc[idx]
        np.testing.assert_almost_equal(actual_vwap, expected_vwap, decimal=6)

    def test_cgo_sign(self, sample_ohlcv_df: pd.DataFrame):
        """CGO 부호: close > vwap이면 양수, close < vwap이면 음수."""
        config = VWAPDispositionConfig(vwap_window=100)
        result = preprocess(sample_ohlcv_df, config)

        valid = result.dropna(subset=["cgo", "vwap"])
        if len(valid) > 0:
            above = valid[valid["close"] > valid["vwap"]]
            if len(above) > 0:
                assert (above["cgo"] > 0).all()

            below = valid[valid["close"] < valid["vwap"]]
            if len(below) > 0:
                assert (below["cgo"] < 0).all()

    def test_volume_ratio_around_one(self, sample_ohlcv_df: pd.DataFrame):
        """Volume ratio 평균은 약 1.0 근처."""
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_vr = result["volume_ratio"].dropna()
        if len(valid_vr) > 0:
            mean_vr = valid_vr.mean()
            assert 0.5 < mean_vr < 2.0

    def test_mom_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = VWAPDispositionConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_custom_vwap_window(self, sample_ohlcv_df: pd.DataFrame):
        """다른 vwap_window로도 정상 동작."""
        config = VWAPDispositionConfig(vwap_window=200)
        result = preprocess(sample_ohlcv_df, config)

        valid_vwap = result["vwap"].dropna()
        assert len(valid_vwap) > 0

    def test_volume_ratio_with_custom_window(self, sample_ohlcv_df: pd.DataFrame):
        """vol_ratio_window 변경 시 정상 동작."""
        config = VWAPDispositionConfig(vol_ratio_window=30)
        result = preprocess(sample_ohlcv_df, config)

        valid = result["volume_ratio"].dropna()
        assert len(valid) > 0
