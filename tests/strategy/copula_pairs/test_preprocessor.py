"""Tests for Copula Pairs Trading preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.copula_pairs.config import CopulaPairsConfig
from src.strategy.copula_pairs.preprocessor import (
    calculate_hedge_ratio,
    calculate_spread,
    calculate_spread_zscore,
    preprocess,
)


@pytest.fixture
def sample_pairs_data() -> pd.DataFrame:
    """합성 cointegrated pair 데이터 생성."""
    np.random.seed(42)
    n = 200
    # Generate cointegrated pair
    common_factor = np.cumsum(np.random.randn(n) * 200)
    close = 50000.0 + common_factor + np.random.randn(n) * 100
    pair_close = 3000.0 + common_factor * 0.06 + np.random.randn(n) * 50  # ~beta=0.06

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
            "pair_close": pair_close,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def default_config() -> CopulaPairsConfig:
    """기본 CopulaPairs Config."""
    return CopulaPairsConfig()


class TestPreprocessColumns:
    """preprocess 출력 컬럼 테스트."""

    def test_preprocess_returns_all_columns(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_pairs_data, default_config)

        expected_cols = [
            "returns",
            "realized_vol",
            "hedge_ratio",
            "spread",
            "spread_zscore",
            "vol_scalar",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_not_modified(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_pairs_data.columns)
        original_values = sample_pairs_data["close"].values.copy()

        _ = preprocess(sample_pairs_data, default_config)

        assert list(sample_pairs_data.columns) == original_cols
        np.testing.assert_array_equal(sample_pairs_data["close"].values, original_values)


class TestHedgeRatio:
    """hedge ratio 계산 테스트."""

    def test_hedge_ratio_calculation(
        self,
        sample_pairs_data: pd.DataFrame,
    ) -> None:
        """hedge ratio가 NaN이 아닌 유효값을 포함."""
        close: pd.Series = sample_pairs_data["close"]  # type: ignore[assignment]
        pair_close: pd.Series = sample_pairs_data["pair_close"]  # type: ignore[assignment]

        hr = calculate_hedge_ratio(close, pair_close, window=63)

        # formation_window(63) 이후에 값이 있어야 함
        valid = hr.dropna()
        assert len(valid) > 0
        assert len(valid) == len(hr) - 63  # 첫 63개는 NaN

    def test_hedge_ratio_reasonable_range(
        self,
        sample_pairs_data: pd.DataFrame,
    ) -> None:
        """hedge ratio가 합리적 범위 내."""
        close: pd.Series = sample_pairs_data["close"]  # type: ignore[assignment]
        pair_close: pd.Series = sample_pairs_data["pair_close"]  # type: ignore[assignment]

        hr = calculate_hedge_ratio(close, pair_close, window=63)
        valid = hr.dropna()

        # close ~50000, pair_close ~3000, common_factor 비율 1:0.06
        # beta는 대략 1/0.06 = ~16.7 근처에 있을 수 있음
        # 데이터 특성에 따라 달라질 수 있으므로 넓은 범위로 검증
        assert valid.abs().max() < 1000.0, "Hedge ratio out of reasonable range"

    def test_hedge_ratio_length_mismatch(self) -> None:
        """입력 시리즈 길이 불일치 시 에러."""
        close = pd.Series([100, 101, 102])
        pair_close = pd.Series([50, 51])

        with pytest.raises(ValueError, match="same length"):
            calculate_hedge_ratio(close, pair_close, window=2)


class TestSpread:
    """spread 계산 테스트."""

    def test_spread_calculation(self) -> None:
        """spread = close - beta * pair_close."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        close = pd.Series([100, 102, 104, 106, 108], index=idx, dtype=float)
        pair_close = pd.Series([50, 51, 52, 53, 54], index=idx, dtype=float)
        hedge_ratio = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0], index=idx, dtype=float)

        spread = calculate_spread(close, pair_close, hedge_ratio)

        expected = close - 2.0 * pair_close
        np.testing.assert_allclose(spread.values, expected.values)


class TestSpreadZscore:
    """spread z-score 계산 테스트."""

    def test_spread_zscore_properties(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """z-score가 합리적 범위 내."""
        result = preprocess(sample_pairs_data, default_config)
        zscore = result["spread_zscore"].dropna()

        assert len(zscore) > 0
        # z-score는 대부분 -5 ~ 5 범위 내
        assert zscore.abs().max() < 20.0

    def test_zscore_zero_std_handling(self) -> None:
        """표준편차 0일 때 NaN 반환."""
        # 상수 spread (std=0)
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        spread = pd.Series(np.ones(30) * 100, index=idx)

        zscore = calculate_spread_zscore(spread, window=10)

        # std=0이므로 z-score는 NaN
        valid_window = zscore.iloc[10:]
        assert valid_window.isna().all()


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_pair_close_raises(self, default_config: CopulaPairsConfig) -> None:
        """pair_close 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [98, 99],
                "close": [100, 101],
                "volume": [1000, 2000],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, default_config)

    def test_missing_close_raises(self, default_config: CopulaPairsConfig) -> None:
        """close 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {
                "pair_close": [100, 101],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, default_config)


class TestNumericConversion:
    """숫자 변환 테스트."""

    def test_string_columns_converted(
        self,
        sample_pairs_data: pd.DataFrame,
        default_config: CopulaPairsConfig,
    ) -> None:
        """문자열 컬럼이 float로 변환."""
        # close를 문자열로 변환
        df = sample_pairs_data.copy()
        df["close"] = df["close"].astype(str)

        result = preprocess(df, default_config)

        # 변환 후 숫자 타입이어야 함
        assert result["close"].dtype in [np.float64, float]
