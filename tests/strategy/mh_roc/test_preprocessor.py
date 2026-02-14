"""Tests for Multi-Horizon ROC Ensemble preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mh_roc.config import MhRocConfig
from src.strategy.mh_roc.preprocessor import preprocess


@pytest.fixture
def config() -> MhRocConfig:
    return MhRocConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "roc_short",
            "roc_medium_short",
            "roc_medium_long",
            "roc_long",
            "vote_sum",
            "vote_ratio",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: MhRocConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestMultiHorizonROC:
    def test_roc_values_finite(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for col in ["roc_short", "roc_medium_short", "roc_medium_long", "roc_long"]:
            valid = result[col].dropna()
            assert np.isfinite(valid).all()

    def test_vote_sum_range(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        """vote_sum은 -4 ~ +4 범위."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vote_sum"].dropna()
        assert (valid >= -4).all()
        assert (valid <= 4).all()

    def test_vote_ratio_range(self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> None:
        """vote_ratio는 -1.0 ~ +1.0 범위."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vote_ratio"].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_trending_up_unanimous(self) -> None:
        """순수 상승 추세에서 vote_sum=+4 (만장일치)."""
        n = 200
        close = np.linspace(100, 200, n)
        high = close + 2.0
        low = close - 2.0
        open_ = close - 0.5
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        config = MhRocConfig()
        result = preprocess(df, config)
        # 충분한 warmup 이후 vote_sum = +4 (모든 horizon 상승)
        late = result.iloc[config.roc_long + 5 :]
        valid_votes = late["vote_sum"].dropna()
        assert (valid_votes == 4).all()

    def test_roc_shorter_has_more_valid(
        self, sample_ohlcv_df: pd.DataFrame, config: MhRocConfig
    ) -> None:
        """짧은 horizon은 NaN이 적음."""
        result = preprocess(sample_ohlcv_df, config)
        short_valid = result["roc_short"].dropna().shape[0]
        long_valid = result["roc_long"].dropna().shape[0]
        assert short_valid >= long_valid
