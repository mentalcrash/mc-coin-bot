"""Tests for VW-TSMOM Pure preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vw_tsmom.config import VWTSMOMConfig
from src.strategy.vw_tsmom.preprocessor import calculate_vw_returns, preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일).

    vol_window=30 기본값에서 non-NaN 값을 보장하기 위해
    최소 31일 이상의 데이터가 필요합니다. 200일로 설정.
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
def sample_config() -> VWTSMOMConfig:
    """기본 VW-TSMOM Config."""
    return VWTSMOMConfig()


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_adds_expected_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VWTSMOMConfig,
    ):
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, sample_config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vw_returns",
            "vol_scalar",
            "drawdown",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_returns_calculation(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VWTSMOMConfig,
    ):
        """수익률 계산 검증 (로그 수익률)."""
        result = preprocess(sample_ohlcv_df, sample_config)

        returns = result["returns"].dropna()
        assert len(returns) > 0

        # 로그 수익률: ln(P_t / P_{t-1})
        close = sample_ohlcv_df["close"].astype(float)
        expected = np.log(close / close.shift(1)).dropna()

        np.testing.assert_allclose(
            returns.values,
            expected.values,
            rtol=1e-10,
        )

    def test_vol_scalar_bounds(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VWTSMOMConfig,
    ):
        """vol_scalar는 항상 양수."""
        result = preprocess(sample_ohlcv_df, sample_config)

        vol_scalar = result["vol_scalar"].dropna()
        assert len(vol_scalar) > 0
        assert (vol_scalar > 0).all()

    def test_missing_columns_raises(self, sample_config: VWTSMOMConfig):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, sample_config)

    def test_numeric_conversion(
        self,
        sample_config: VWTSMOMConfig,
    ):
        """Decimal/문자열 컬럼이 float64로 변환."""
        n = 100
        df = pd.DataFrame(
            {
                "open": [str(100 + i * 0.1) for i in range(n)],
                "high": [str(101 + i * 0.1) for i in range(n)],
                "low": [str(99 + i * 0.1) for i in range(n)],
                "close": [str(100 + i * 0.1) for i in range(n)],
                "volume": [str(1000.5 + i) for i in range(n)],
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        result = preprocess(df, sample_config)

        assert result["close"].dtype == np.float64
        assert result["volume"].dtype == np.float64

    def test_drawdown_calculation(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VWTSMOMConfig,
    ):
        """drawdown은 항상 0 이하."""
        result = preprocess(sample_ohlcv_df, sample_config)

        drawdown = result["drawdown"].dropna()
        assert (drawdown <= 0).all()

    def test_original_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VWTSMOMConfig,
    ):
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, sample_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)


class TestCalculateVWReturns:
    """calculate_vw_returns 함수 테스트."""

    def test_vw_returns_calculation(self):
        """VW returns 기본 계산 검증."""
        np.random.seed(42)
        n = 50

        returns = pd.Series(np.random.randn(n) * 0.01)
        volume = pd.Series(np.random.randint(1000, 10000, n).astype(float))

        vw_ret = calculate_vw_returns(returns, volume, window=10)

        # 처음 9개는 NaN (window=10, min_periods=10)
        assert vw_ret.iloc[:9].isna().all()

        # 10번째부터 값 존재
        valid = vw_ret.dropna()
        assert len(valid) > 0

        # 유한 값인지 확인
        assert np.isfinite(valid.values).all()

    def test_vw_returns_with_uniform_volume_equals_simple_mean(self):
        """균등 거래량에서 VW returns = 단순 평균 수익률."""
        n = 50
        np.random.seed(42)

        returns = pd.Series(np.random.randn(n) * 0.01)
        # 모든 거래량이 동일
        uniform_volume = pd.Series(np.full(n, 5000.0))

        window = 10
        vw_ret = calculate_vw_returns(returns, uniform_volume, window=window)
        simple_mean: pd.Series = returns.rolling(  # type: ignore[assignment]
            window=window, min_periods=window
        ).mean()

        # 균등 거래량이면 VW returns == simple rolling mean
        valid_mask = vw_ret.notna() & simple_mean.notna()
        if valid_mask.any():
            np.testing.assert_allclose(
                vw_ret[valid_mask].values,
                simple_mean[valid_mask].values,
                rtol=1e-10,
            )

    def test_vw_returns_high_volume_bar_dominates(self):
        """높은 거래량의 bar가 VW returns에 더 큰 영향."""
        n = 20

        # 모든 returns 동일하지만 마지막 bar만 높은 거래량
        returns = pd.Series([0.01] * 10 + [-0.01] * 10)
        volume_uniform = pd.Series(np.full(n, 1000.0))
        volume_skewed = pd.Series(
            [1000.0] * 10 + [100000.0] * 10,
        )

        window = 20
        vw_uniform = calculate_vw_returns(returns, volume_uniform, window=window)
        vw_skewed = calculate_vw_returns(returns, volume_skewed, window=window)

        # 균등 거래량: 평균 0
        # 편향 거래량: 음수 방향 (큰 거래량이 음수 return 구간)
        last_uniform = vw_uniform.iloc[-1]
        last_skewed = vw_skewed.iloc[-1]
        assert last_skewed < last_uniform
