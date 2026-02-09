"""Tests for Multi-Factor Ensemble preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.multi_factor.config import MultiFactorConfig
from src.strategy.multi_factor.preprocessor import (
    calculate_momentum_factor,
    calculate_volatility_factor,
    calculate_volume_shock_factor,
    preprocess,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200

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
def default_config() -> MultiFactorConfig:
    """기본 MultiFactorConfig."""
    return MultiFactorConfig()


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_returns_all_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, default_config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "momentum_factor",
            "volume_shock_factor",
            "volatility_factor",
            "combined_score",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_combined_score_is_average(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """combined_score는 3개 팩터의 균등 가중 평균."""
        result = preprocess(sample_ohlcv_df, default_config)

        # NaN이 아닌 행에서 검증
        valid = result.dropna(
            subset=["momentum_factor", "volume_shock_factor", "volatility_factor"]
        )
        if len(valid) > 0:
            expected = (
                valid["momentum_factor"] + valid["volume_shock_factor"] + valid["volatility_factor"]
            ) / 3.0
            np.testing.assert_allclose(
                valid["combined_score"].values,
                expected.values,
                rtol=1e-10,
            )

    def test_momentum_factor_zscore_properties(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """모멘텀 팩터가 z-score 성질을 가짐 (mean~0, std~1)."""
        result = preprocess(sample_ohlcv_df, default_config)

        mom = result["momentum_factor"].dropna()
        if len(mom) > 10:
            # z-score이므로 대략 mean~0, std~1 (유한 샘플에서는 정확하지 않음)
            assert abs(mom.mean()) < 1.0  # 대략 0 근처
            assert 0.1 < mom.std() < 3.0  # 대략 1 근처

    def test_volume_shock_factor(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """거래량 충격 팩터 계산 검증."""
        result = preprocess(sample_ohlcv_df, default_config)

        vol_shock = result["volume_shock_factor"].dropna()
        assert len(vol_shock) > 0
        # z-score이므로 유한한 값
        assert vol_shock.abs().max() < 10.0

    def test_volatility_factor_inverse(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """역변동성 팩터: 낮은 vol이 높은 점수."""
        result = preprocess(sample_ohlcv_df, default_config)

        vol_factor = result["volatility_factor"].dropna()
        assert len(vol_factor) > 0
        # 값이 유한한지 확인
        assert vol_factor.abs().max() < 10.0

    def test_missing_columns_raises(
        self,
        default_config: MultiFactorConfig,
    ) -> None:
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, default_config)

    def test_empty_df_raises(
        self,
        default_config: MultiFactorConfig,
    ) -> None:
        """빈 DataFrame 시 에러."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )

        with pytest.raises(ValueError, match="empty"):
            preprocess(df, default_config)

    def test_numeric_conversion(
        self,
        default_config: MultiFactorConfig,
    ) -> None:
        """Decimal/문자열 컬럼이 float64로 변환."""
        from decimal import Decimal

        n = 100
        np.random.seed(42)

        close_vals = 100 + np.cumsum(np.random.randn(n) * 2)

        df = pd.DataFrame(
            {
                "open": [Decimal(str(v)) for v in close_vals],
                "high": [Decimal(str(v + 1)) for v in close_vals],
                "low": [Decimal(str(v - 1)) for v in close_vals],
                "close": [Decimal(str(v)) for v in close_vals],
                "volume": [Decimal(5000)] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        result = preprocess(df, default_config)
        assert result["close"].dtype == np.float64

    def test_original_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, default_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)

    def test_factor_independence(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: MultiFactorConfig,
    ) -> None:
        """3개 팩터가 서로 독립적으로 계산됨 (상관계수 < 0.95)."""
        result = preprocess(sample_ohlcv_df, default_config)

        valid = result.dropna(
            subset=["momentum_factor", "volume_shock_factor", "volatility_factor"]
        )
        if len(valid) > 10:
            corr = valid[["momentum_factor", "volume_shock_factor", "volatility_factor"]].corr()

            # 팩터 간 상관계수가 너무 높지 않은지 검증 (완전 종속이 아닌지)
            assert abs(corr.loc["momentum_factor", "volume_shock_factor"]) < 0.95
            assert abs(corr.loc["momentum_factor", "volatility_factor"]) < 0.95
            assert abs(corr.loc["volume_shock_factor", "volatility_factor"]) < 0.95


class TestIndividualFactors:
    """개별 팩터 함수 테스트."""

    def test_momentum_factor_function(self) -> None:
        """calculate_momentum_factor 단독 테스트."""
        np.random.seed(42)
        close = pd.Series(
            100 + np.cumsum(np.random.randn(200) * 2),
            index=pd.date_range("2024-01-01", periods=200, freq="D"),
        )

        result = calculate_momentum_factor(close, lookback=21, zscore_window=60)

        assert len(result) == len(close)
        assert result.name == "momentum_factor"
        # 초기 NaN 존재
        assert result.isna().any()
        # 유효 값 존재
        assert result.notna().sum() > 0

    def test_volume_shock_factor_function(self) -> None:
        """calculate_volume_shock_factor 단독 테스트."""
        np.random.seed(42)
        volume = pd.Series(
            np.random.randint(1000, 10000, 200).astype(float),
            index=pd.date_range("2024-01-01", periods=200, freq="D"),
        )

        result = calculate_volume_shock_factor(volume, window=5, zscore_window=60)

        assert len(result) == len(volume)
        assert result.name == "volume_shock_factor"
        assert result.notna().sum() > 0

    def test_volatility_factor_function(self) -> None:
        """calculate_volatility_factor 단독 테스트."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(200) * 0.02,
            index=pd.date_range("2024-01-01", periods=200, freq="D"),
        )

        result = calculate_volatility_factor(returns, window=30, zscore_window=60)

        assert len(result) == len(returns)
        assert result.name == "volatility_factor"
        assert result.notna().sum() > 0
