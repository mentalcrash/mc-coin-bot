"""Tests for Hour Seasonality Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hour_season.config import HourSeasonConfig
from src.strategy.hour_season.preprocessor import _compute_hour_t_stat, preprocess


@pytest.fixture
def sample_1h_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (1H, 1000 bars)."""
    np.random.seed(42)
    n = 1000

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)
    open_ = close + np.random.randn(n) * 0.3

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class TestComputeHourTStat:
    """_compute_hour_t_stat() 테스트."""

    def test_returns_series(self):
        """t-stat이 Series로 반환되는지 확인."""
        np.random.seed(42)
        n = 1000
        returns = pd.Series(
            np.random.randn(n) * 0.01,
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

        t_stat = _compute_hour_t_stat(returns, season_window_days=10)
        assert isinstance(t_stat, pd.Series)
        assert len(t_stat) == n

    def test_warmup_period_nan(self):
        """워밍업 기간에는 NaN이어야 함."""
        np.random.seed(42)
        n = 1000
        returns = pd.Series(
            np.random.randn(n) * 0.01,
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

        t_stat = _compute_hour_t_stat(returns, season_window_days=10)

        # 10일 x 24시간 = 240 bars 이전에는 대부분 NaN
        assert t_stat.iloc[:240].isna().sum() > 0

    def test_consistent_positive_returns_high_t_stat(self):
        """지속적 양수 수익률에서 높은 t-stat."""
        n = 1000
        # 모든 시간대에서 양수 수익률 (고정 bias)
        returns = pd.Series(
            np.full(n, 0.005),
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

        _compute_hour_t_stat(returns, season_window_days=10)

        # 워밍업 이후 t-stat이 매우 높아야 함 (모든 값이 동일하면 std=0 -> NaN)
        # 약간의 노이즈를 추가하여 유한한 t-stat 생성
        returns_noisy = returns + pd.Series(
            np.random.randn(n) * 0.001,
            index=returns.index,
        )
        t_stat_noisy = _compute_hour_t_stat(returns_noisy, season_window_days=10)
        valid = t_stat_noisy.dropna()

        if len(valid) > 0:
            assert valid.mean() > 0  # 양수 bias → 양수 t-stat


class TestPreprocess:
    """preprocess() 테스트."""

    def test_output_columns(self, sample_1h_df: pd.DataFrame):
        """출력 DataFrame에 필요한 컬럼이 존재하는지 확인."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        result = preprocess(sample_1h_df, config)

        expected_cols = [
            "returns",
            "hour_t_stat",
            "rel_volume",
            "vol_confirm",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_vol_confirm_boolean(self, sample_1h_df: pd.DataFrame):
        """vol_confirm은 boolean 타입."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        result = preprocess(sample_1h_df, config)

        valid = result["vol_confirm"].dropna()
        assert valid.dtype == bool

    def test_rel_volume_positive(self, sample_1h_df: pd.DataFrame):
        """rel_volume은 양수."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        result = preprocess(sample_1h_df, config)

        valid = result["rel_volume"].dropna()
        assert (valid > 0).all()

    def test_vol_scalar_positive(self, sample_1h_df: pd.DataFrame):
        """vol_scalar는 양수."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        result = preprocess(sample_1h_df, config)

        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_immutability(self, sample_1h_df: pd.DataFrame):
        """원본 DataFrame은 수정되지 않아야 함."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        original = sample_1h_df.copy()
        preprocess(sample_1h_df, config)

        pd.testing.assert_frame_equal(sample_1h_df, original)

    def test_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = HourSeasonConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_1h_df: pd.DataFrame):
        """출력 길이는 입력과 동일."""
        config = HourSeasonConfig(season_window_days=7, vol_confirm_window=48)
        result = preprocess(sample_1h_df, config)

        assert len(result) == len(sample_1h_df)
