"""Tests for Entropy-Carry-Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.entropy_carry_mom.config import EntropyCarryMomConfig
from src.strategy.entropy_carry_mom.preprocessor import preprocess


@pytest.fixture
def config() -> EntropyCarryMomConfig:
    return EntropyCarryMomConfig()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig) -> None:
        result = preprocess(sample_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "entropy",
            "entropy_rank",
            "mom_direction",
            "mom_strength",
            "avg_funding_rate",
            "fr_zscore",
            "carry_direction",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig) -> None:
        result = preprocess(sample_df, config)
        assert len(result) == len(sample_df)

    def test_immutability(self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig) -> None:
        original = sample_df.copy()
        preprocess(sample_df, config)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_missing_columns(self, config: EntropyCarryMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: EntropyCarryMomConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        result = preprocess(sample_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_entropy_nonnegative(
        self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        result = preprocess(sample_df, config)
        valid_entropy = result["entropy"].dropna()
        assert (valid_entropy >= 0).all()

    def test_entropy_rank_range(
        self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        result = preprocess(sample_df, config)
        valid_rank = result["entropy_rank"].dropna()
        assert (valid_rank >= 0).all()
        assert (valid_rank <= 1).all()

    def test_carry_direction_values(
        self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        result = preprocess(sample_df, config)
        valid_cd = result["carry_direction"].dropna()
        assert set(valid_cd.unique()).issubset({-1.0, 0.0, 1.0})

    def test_drawdown_nonpositive(
        self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        result = preprocess(sample_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()

    def test_mom_direction_values(
        self, sample_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        result = preprocess(sample_df, config)
        valid_md = result["mom_direction"].dropna()
        assert set(valid_md.unique()).issubset({-1.0, 0.0, 1.0})

    def test_funding_rate_ffill(self, config: EntropyCarryMomConfig) -> None:
        """Funding rate NaN should be forward-filled."""
        np.random.seed(42)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 1
        low = close - 1
        funding_rate = np.full(n, 0.001)
        funding_rate[:10] = np.nan  # First 10 NaN
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": funding_rate,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        result = preprocess(df, config)
        # avg_funding_rate should have values after warmup
        assert result["avg_funding_rate"].iloc[-1] > 0
