"""Tests for Vol Structure Regime preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_structure.config import VolStructureConfig
from src.strategy.vol_structure.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일).

    vol_long_window=60 기본값에서 non-NaN 값을 보장하기 위해
    최소 61일 이상의 데이터가 필요합니다. 200일로 설정.
    """
    np.random.seed(42)
    n = 200

    # 상승 추세 + 노이즈
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )

    return df


@pytest.fixture
def sample_config() -> VolStructureConfig:
    """기본 Vol Structure Config."""
    return VolStructureConfig()


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_returns_all_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, sample_config)

        expected_cols = [
            "returns",
            "vol_short",
            "vol_long",
            "vol_ratio",
            "norm_momentum",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_vol_ratio_positive(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """vol_ratio는 양수 (양쪽 vol > 0이므로)."""
        result = preprocess(sample_ohlcv_df, sample_config)

        vol_ratio = result["vol_ratio"].dropna()
        assert len(vol_ratio) > 0
        assert (vol_ratio > 0).all()

    def test_vol_ratio_calculation(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """vol_ratio = vol_short / vol_long."""
        result = preprocess(sample_ohlcv_df, sample_config)

        vol_short = result["vol_short"].dropna()
        vol_long = result["vol_long"].dropna()
        vol_ratio = result["vol_ratio"].dropna()

        # 공통 인덱스에서 비교
        common_idx = vol_short.index.intersection(vol_long.index).intersection(vol_ratio.index)
        assert len(common_idx) > 0

        expected_ratio = vol_short.loc[common_idx] / vol_long.loc[common_idx]
        np.testing.assert_allclose(
            vol_ratio.loc[common_idx].values,
            expected_ratio.values,
            rtol=1e-10,
        )

    def test_norm_momentum_is_z_like(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """norm_momentum은 z-score 유사 값 (returns sum / returns std)."""
        result = preprocess(sample_ohlcv_df, sample_config)

        norm_mom = result["norm_momentum"].dropna()
        assert len(norm_mom) > 0

        # z-score와 유사하므로 대부분 합리적 범위 내 (극단값 허용)
        assert norm_mom.abs().max() < 20.0

    def test_missing_columns_raises(self, sample_config: VolStructureConfig):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, sample_config)

    def test_original_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, sample_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)

    def test_drawdown_is_non_positive(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """drawdown은 항상 0 이하."""
        result = preprocess(sample_ohlcv_df, sample_config)

        drawdown = result["drawdown"].dropna()
        assert (drawdown <= 0).all()

    def test_vol_scalar_positive(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolStructureConfig,
    ):
        """vol_scalar는 항상 양수."""
        result = preprocess(sample_ohlcv_df, sample_config)

        vol_scalar = result["vol_scalar"].dropna()
        assert len(vol_scalar) > 0
        assert (vol_scalar > 0).all()
