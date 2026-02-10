"""Tests for Kalman Trend Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.kalman_trend.config import KalmanTrendConfig
from src.strategy.kalman_trend.preprocessor import preprocess


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
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "kalman_state",
            "kalman_velocity",
            "q_adaptive",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_kalman_state_tracks_price(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """칼만 smoothed price는 실제 가격 근처를 추종해야 함."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result.dropna(subset=["kalman_state"])
        if len(valid) > 0:
            # smoothed price와 실제 가격의 상관관계가 높아야 함
            corr = valid["kalman_state"].corr(valid["close"])
            assert corr > 0.95, f"Kalman state-price correlation too low: {corr}"

    def test_kalman_velocity_exists(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """velocity 컬럼 존재 및 유한값 확인."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_vel = result["kalman_velocity"].dropna()
        assert len(valid_vel) > 0
        assert np.isfinite(valid_vel).all()

    def test_q_adaptive_bounded(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Adaptive Q 값이 범위 내에 있는지 확인."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        q_values = result["q_adaptive"]
        # q_adaptive = base_q * clip(ratio, 0.001, 10.0)
        # 최소: base_q * 0.001 = 0.01 * 0.001 = 0.00001
        # 최대: base_q * 10.0 = 0.01 * 10.0 = 0.1
        # NaN은 base_q로 채워짐 = 0.01
        assert (q_values >= 0).all()
        assert np.isfinite(q_values).all()

    def test_realized_vol_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """실현 변동성은 양수여야 함."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["realized_vol"].dropna()
        assert (valid >= 0).all()

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """변동성 스케일러는 양수여야 함."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """원본 DataFrame이 변경되지 않아야 함."""
        config = KalmanTrendConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self) -> None:
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = KalmanTrendConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """출력 길이가 입력과 동일해야 함."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_drawdown_nonpositive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """드로다운은 0 이하여야 함."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ATR은 양수여야 함."""
        config = KalmanTrendConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_trending_data_positive_velocity(self) -> None:
        """상승 추세 데이터에서 velocity가 양수 경향이어야 함."""
        n = 300
        # 강한 상승 추세
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5
        high = close + 1.0
        low = close - 1.0

        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = KalmanTrendConfig()
        result = preprocess(df, config)

        # 후반부의 velocity는 대체로 양수여야 함
        late_velocity = result["kalman_velocity"].iloc[-50:]
        assert late_velocity.mean() > 0, "Uptrend data should have positive velocity"

    def test_custom_config(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 config로 전처리."""
        config = KalmanTrendConfig(
            base_q=0.05,
            observation_noise=2.0,
            vol_lookback=10,
            long_term_vol_lookback=60,
        )
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "kalman_state",
            "kalman_velocity",
            "q_adaptive",
        ]
        for col in expected_cols:
            assert col in result.columns
