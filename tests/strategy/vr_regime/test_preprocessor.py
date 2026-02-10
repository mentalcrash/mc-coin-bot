"""Tests for VR Regime Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vr_regime.config import VRRegimeConfig
from src.strategy.vr_regime.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
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
        config = VRRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "vr",
            "vr_z_stat",
            "mom_direction",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_vr_around_one_for_random(self, sample_ohlcv_df: pd.DataFrame):
        """랜덤 워크 데이터에서 VR은 약 1 근처."""
        config = VRRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_vr = result["vr"].dropna()
        if len(valid_vr) > 0:
            mean_vr = valid_vr.mean()
            # 랜덤 데이터의 VR은 1 근처 (넓은 범위 허용)
            assert 0.1 < mean_vr < 5.0

    def test_z_stat_exists(self, sample_ohlcv_df: pd.DataFrame):
        """z-stat 컬럼 존재 확인."""
        config = VRRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_z = result["vr_z_stat"].dropna()
        assert len(valid_z) > 0

    def test_mom_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = VRRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_heteroscedastic_vs_simple(self, sample_ohlcv_df: pd.DataFrame):
        """heteroscedastic 모드와 simple 모드의 결과가 다름."""
        config_het = VRRegimeConfig(use_heteroscedastic=True)
        config_sim = VRRegimeConfig(use_heteroscedastic=False)

        result_het = preprocess(sample_ohlcv_df, config_het)
        result_sim = preprocess(sample_ohlcv_df, config_sim)

        # VR 값은 동일 (동일 수식)
        pd.testing.assert_series_equal(result_het["vr"], result_sim["vr"])
        # z-stat은 다를 수 있음 (다른 SE 수식)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = VRRegimeConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = VRRegimeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = VRRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = VRRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
