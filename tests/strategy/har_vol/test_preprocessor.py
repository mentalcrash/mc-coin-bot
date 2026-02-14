"""Tests for HAR Volatility Overlay preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import parkinson_volatility
from src.strategy.har_vol.config import HARVolConfig
from src.strategy.har_vol.preprocessor import (
    calculate_har_features,
    calculate_har_forecast,
    calculate_vol_surprise,
    preprocess,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (400일).

    training_window=252 + monthly_window=22 + buffer를 위해
    400일 이상의 데이터가 필요합니다.
    """
    np.random.seed(42)
    n = 400

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5) + 0.01  # 항상 close보다 높음
    low = close - np.abs(np.random.randn(n) * 1.5) - 0.01  # 항상 close보다 낮음
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )

    return df


@pytest.fixture
def small_config() -> HARVolConfig:
    """작은 윈도우의 HAR Vol Config (빠른 테스트용)."""
    return HARVolConfig(
        daily_window=1,
        weekly_window=3,
        monthly_window=15,
        training_window=60,
    )


class TestParkinsonVol:
    """Parkinson volatility 테스트."""

    def test_parkinson_vol_calculation(self):
        """Parkinson volatility 공식 검증."""
        high = pd.Series([110.0, 120.0, 115.0])
        low = pd.Series([90.0, 100.0, 105.0])

        result = parkinson_volatility(high, low)

        # 수동 계산: sqrt(1/(4*ln2) * ln(H/L)^2)
        expected_0 = np.sqrt(1.0 / (4.0 * np.log(2.0)) * np.log(110.0 / 90.0) ** 2)
        np.testing.assert_almost_equal(result.iloc[0], expected_0, decimal=10)

    def test_parkinson_vol_positive(self, sample_ohlcv_df: pd.DataFrame):
        """Parkinson vol은 항상 양수 (high > low일 때)."""
        high: pd.Series = sample_ohlcv_df["high"]  # type: ignore[assignment]
        low: pd.Series = sample_ohlcv_df["low"]  # type: ignore[assignment]
        result = parkinson_volatility(high, low)

        assert (result.dropna() >= 0).all()


class TestHARFeatures:
    """HAR features 테스트."""

    def test_har_features_shapes(self):
        """HAR features의 shape와 컬럼 확인."""
        np.random.seed(42)
        n = 100
        parkinson_vol = pd.Series(
            np.abs(np.random.randn(n) * 0.02) + 0.01,
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        features = calculate_har_features(parkinson_vol, daily_w=1, weekly_w=5, monthly_w=22)

        assert isinstance(features, pd.DataFrame)
        assert "rv_daily" in features.columns
        assert "rv_weekly" in features.columns
        assert "rv_monthly" in features.columns
        assert len(features) == n

    def test_har_features_rolling_means(self):
        """HAR features는 rolling mean으로 계산."""
        np.random.seed(42)
        n = 50
        parkinson_vol = pd.Series(
            np.abs(np.random.randn(n) * 0.02) + 0.01,
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        features = calculate_har_features(parkinson_vol, daily_w=1, weekly_w=5, monthly_w=22)

        # rv_daily (window=1) == parkinson_vol 자체
        np.testing.assert_array_equal(
            features["rv_daily"].values,
            parkinson_vol.values,
        )

        # rv_weekly는 window=5 rolling mean
        expected_weekly = parkinson_vol.rolling(5, min_periods=5).mean()
        np.testing.assert_allclose(
            features["rv_weekly"].dropna().values,
            expected_weekly.dropna().values,
            rtol=1e-10,
        )


class TestHARForecast:
    """HAR forecast 테스트."""

    def test_har_forecast_no_lookahead(self, sample_ohlcv_df: pd.DataFrame):
        """HAR forecast는 미래 데이터를 사용하지 않음 (no lookahead)."""
        config = HARVolConfig(
            daily_window=1,
            weekly_window=3,
            monthly_window=15,
            training_window=60,
        )
        high: pd.Series = sample_ohlcv_df["high"]  # type: ignore[assignment]
        low: pd.Series = sample_ohlcv_df["low"]  # type: ignore[assignment]

        parkinson_vol = parkinson_volatility(high, low)
        features = calculate_har_features(
            parkinson_vol,
            daily_w=config.daily_window,
            weekly_w=config.weekly_window,
            monthly_w=config.monthly_window,
        )
        forecast = calculate_har_forecast(parkinson_vol, features, config.training_window)

        # training_window 이전은 NaN이어야 함
        assert forecast.iloc[: config.training_window].isna().all()

        # training_window 이후에 유효한 값이 존재
        valid_forecast = forecast.dropna()
        assert len(valid_forecast) > 0

    def test_rolling_ols_window(self, sample_ohlcv_df: pd.DataFrame):
        """Rolling OLS가 정해진 윈도우 크기를 사용하는지 확인."""
        config = HARVolConfig(
            daily_window=1,
            weekly_window=3,
            monthly_window=15,
            training_window=60,
        )
        high: pd.Series = sample_ohlcv_df["high"]  # type: ignore[assignment]
        low: pd.Series = sample_ohlcv_df["low"]  # type: ignore[assignment]

        parkinson_vol = parkinson_volatility(high, low)
        features = calculate_har_features(
            parkinson_vol,
            daily_w=config.daily_window,
            weekly_w=config.weekly_window,
            monthly_w=config.monthly_window,
        )
        forecast = calculate_har_forecast(parkinson_vol, features, config.training_window)

        # 첫 번째 유효한 forecast는 정확히 training_window 인덱스에서
        first_valid_idx = forecast.first_valid_index()
        assert first_valid_idx is not None
        first_valid_pos = forecast.index.get_loc(first_valid_idx)
        assert first_valid_pos >= config.training_window


class TestVolSurprise:
    """Vol surprise 테스트."""

    def test_vol_surprise_sign(self):
        """vol_surprise 부호 검증: realized > forecast → 양수."""
        realized = pd.Series([0.05, 0.03, 0.06])
        forecast = pd.Series([0.04, 0.04, 0.04])

        result = calculate_vol_surprise(realized, forecast)

        assert result.iloc[0] > 0  # 0.05 - 0.04 = 0.01
        assert result.iloc[1] < 0  # 0.03 - 0.04 = -0.01
        assert result.iloc[2] > 0  # 0.06 - 0.04 = 0.02


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, small_config)

        expected_cols = [
            "returns",
            "parkinson_vol",
            "rv_daily",
            "rv_weekly",
            "rv_monthly",
            "har_forecast",
            "vol_surprise",
            "realized_vol",
            "vol_scalar",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_missing_cols_raises(self, small_config: HARVolConfig):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, small_config)

    def test_numeric_conversion(self, small_config: HARVolConfig):
        """Decimal 등 비-float 타입이 변환되는지 확인."""
        from decimal import Decimal

        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "open": [Decimal(100)] * n,
                "high": [Decimal(110)] * n,
                "low": [Decimal(90)] * n,
                "close": [Decimal(100)] * n,
                "volume": [Decimal(1000)] * n,
            },
            index=idx,
        )

        result = preprocess(df, small_config)
        assert result["close"].dtype == np.float64

    def test_original_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, small_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)

    def test_vol_scalar_positive(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """vol_scalar는 항상 양수."""
        result = preprocess(sample_ohlcv_df, small_config)

        vol_scalar = result["vol_scalar"].dropna()
        assert len(vol_scalar) > 0
        assert (vol_scalar > 0).all()
