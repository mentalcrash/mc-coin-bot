"""Tests for Entropy Switch Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.entropy_switch.config import EntropySwitchConfig
from src.strategy.entropy_switch.preprocessor import preprocess


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "entropy",
            "mom_direction",
            "adx",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_entropy_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Entropy 값은 양수이고 합리적 범위에 있어야 함."""
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_entropy = result["entropy"].dropna()
        if len(valid_entropy) > 0:
            # Shannon entropy는 항상 >= 0
            assert (valid_entropy >= 0).all()
            # 10 bins의 최대 entropy: log(10) ≈ 2.302
            # 실제 값은 이보다 작거나 같아야 함
            assert valid_entropy.max() <= np.log(config.entropy_bins) + 0.1

    def test_adx_exists(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ADX 컬럼 존재 확인."""
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_adx = result["adx"].dropna()
        assert len(valid_adx) > 0
        # ADX는 0~100 범위
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_mom_direction_values(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = EntropySwitchConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_entropy_different_bins(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """빈 수가 다르면 entropy 값도 다름."""
        config_10 = EntropySwitchConfig(entropy_bins=10)
        config_20 = EntropySwitchConfig(entropy_bins=20)

        result_10 = preprocess(sample_ohlcv_df, config_10)
        result_20 = preprocess(sample_ohlcv_df, config_20)

        valid_10 = result_10["entropy"].dropna()
        valid_20 = result_20["entropy"].dropna()

        if len(valid_10) > 0 and len(valid_20) > 0:
            # 다른 빈 수에서 entropy 평균이 달라야 함
            assert valid_10.mean() != valid_20.mean()

    def test_drawdown_negative_or_zero(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Drawdown은 항상 0 이하."""
        config = EntropySwitchConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()
