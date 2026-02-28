"""Tests for Participation Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.participation_mom_12h.config import ParticipationMomConfig
from src.strategy.participation_mom_12h.preprocessor import preprocess


@pytest.fixture
def config() -> ParticipationMomConfig:
    return ParticipationMomConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


@pytest.fixture
def sample_ohlcv_with_tflow(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + tflow_intensity 포함 테스트 데이터."""
    df = sample_ohlcv_df.copy()
    np.random.seed(42)
    df["tflow_intensity"] = np.random.uniform(10, 200, len(df))
    return df


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "intensity_zscore",
            "mom_direction",
            "mom_strength",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: ParticipationMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_mom_direction_values(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert set(result["mom_direction"].unique()).issubset({-1, 1})

    def test_drawdown_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestGracefulDegradation:
    """tflow_intensity 부재 시 전략이 중립(0)으로 동작하는지 검증."""

    def test_without_tflow_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        """tflow_intensity 컬럼 없이도 에러 없이 실행."""
        result = preprocess(sample_ohlcv_df, config)
        assert "intensity_zscore" in result.columns
        assert (result["intensity_zscore"] == 0.0).all()

    def test_with_tflow_columns(
        self,
        sample_ohlcv_with_tflow: pd.DataFrame,
        config: ParticipationMomConfig,
    ) -> None:
        """tflow_intensity 있으면 Z-score가 계산됨."""
        result = preprocess(sample_ohlcv_with_tflow, config)
        assert "intensity_zscore" in result.columns
        # Z-score는 0이 아닌 값이 존재해야 함
        valid = result["intensity_zscore"].dropna()
        assert not (valid == 0.0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
