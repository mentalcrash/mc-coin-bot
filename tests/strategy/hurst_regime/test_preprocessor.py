"""Tests for Hurst/ER Regime preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hurst_regime.config import HurstRegimeConfig
from src.strategy.hurst_regime.preprocessor import (
    _compute_rolling_hurst_numba,
    calculate_efficiency_ratio,
    calculate_rolling_hurst,
    preprocess,
)


@pytest.fixture
def sample_config() -> HurstRegimeConfig:
    """hurst_window=100 샘플 설정."""
    return HurstRegimeConfig(hurst_window=100)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (250일).

    hurst_window=100 기본값에서 non-NaN 값을 보장하기 위해
    최소 101일 이상의 데이터가 필요합니다. 250일로 설정.
    """
    np.random.seed(42)
    n = 250

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


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_returns_all_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HurstRegimeConfig,
    ):
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, sample_config)

        expected_cols = [
            "er",
            "hurst",
            "momentum",
            "z_score",
            "vol_scalar",
            "atr",
            "drawdown",
            "returns",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_missing_columns_raises(self, sample_config: HurstRegimeConfig):
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
        sample_config: HurstRegimeConfig,
    ):
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, sample_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)


class TestEfficiencyRatio:
    """Efficiency Ratio 계산 테스트."""

    def test_er_range_0_to_1(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HurstRegimeConfig,
    ):
        """ER 값은 0~1 범위 내."""
        result = preprocess(sample_ohlcv_df, sample_config)

        er = result["er"].dropna()
        assert len(er) > 0
        assert er.min() >= 0.0
        assert er.max() <= 1.0

    def test_er_trending_data(self):
        """순수 상승 추세 데이터에서 ER은 1에 가까워야 함."""
        close = pd.Series(
            np.arange(1, 51, dtype=float),
            index=pd.date_range("2024-01-01", periods=50, freq="D"),
        )
        er = calculate_efficiency_ratio(close, lookback=10)
        valid = er.dropna()
        # 완벽한 추세에서 ER ≈ 1.0
        assert valid.iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_er_name(self, sample_ohlcv_df: pd.DataFrame):
        """ER 시리즈 이름 확인."""
        close: pd.Series = sample_ohlcv_df["close"]  # type: ignore[assignment]
        er = calculate_efficiency_ratio(close, lookback=20)
        assert er.name == "er"


class TestRollingHurst:
    """Rolling Hurst exponent 계산 테스트."""

    def test_hurst_range(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HurstRegimeConfig,
    ):
        """Hurst 값은 유효한 범위 내 (0~1)."""
        result = preprocess(sample_ohlcv_df, sample_config)

        hurst = result["hurst"].dropna()
        assert len(hurst) > 0
        assert hurst.min() >= 0.0
        assert hurst.max() <= 1.0

    def test_hurst_name(self, sample_ohlcv_df: pd.DataFrame):
        """Hurst 시리즈 이름 확인."""
        close: pd.Series = sample_ohlcv_df["close"]  # type: ignore[assignment]
        returns = np.log(close / close.shift(1))
        hurst = calculate_rolling_hurst(returns, window=100)
        assert hurst.name == "hurst"

    def test_numba_hurst_unit_test(self):
        """_compute_rolling_hurst_numba를 직접 테스트.

        알려진 데이터로 Hurst exponent가 합리적인 범위인지 검증.
        """
        np.random.seed(123)
        # Random walk → Hurst ≈ 0.5
        n = 200
        returns = np.random.randn(n) * 0.01
        window = 100

        hurst_values = _compute_rolling_hurst_numba(returns, window)

        # window 이전은 NaN
        assert np.all(np.isnan(hurst_values[:window]))

        # window 이후에 유효한 값 존재
        valid = hurst_values[~np.isnan(hurst_values)]
        assert len(valid) > 0

        # Random walk의 Hurst는 대략 0.5 근처
        mean_hurst = np.mean(valid)
        assert 0.2 < mean_hurst < 0.8, f"Expected Hurst ~0.5, got {mean_hurst:.3f}"

    def test_numba_hurst_with_nan(self):
        """NaN이 포함된 데이터에서 _compute_rolling_hurst_numba가 NaN을 반환."""
        returns = np.full(150, 0.01)
        returns[50] = np.nan  # 중간에 NaN
        window = 100

        hurst_values = _compute_rolling_hurst_numba(returns, window)

        # NaN이 윈도우 안에 포함되는 인덱스에서는 NaN
        # index 100~149에서 segment[0:100]은 NaN 포함 (index 50)
        # → index 100~149는 NaN이어야 함
        assert np.isnan(hurst_values[100])


class TestMomentum:
    """모멘텀 계산 테스트."""

    def test_momentum_calculation(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HurstRegimeConfig,
    ):
        """모멘텀 값이 정상적으로 계산되는지 확인."""
        result = preprocess(sample_ohlcv_df, sample_config)

        momentum = result["momentum"].dropna()
        assert len(momentum) > 0
        # 모멘텀은 누적 수익률이므로 0 주변 값
        assert not np.isinf(momentum).any()


class TestZScore:
    """Z-Score 계산 테스트."""

    def test_z_score_calculation(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HurstRegimeConfig,
    ):
        """Z-Score 값이 정상적으로 계산되는지 확인."""
        result = preprocess(sample_ohlcv_df, sample_config)

        z_score = result["z_score"].dropna()
        assert len(z_score) > 0
        # Z-Score는 대체로 -3 ~ +3 범위
        assert z_score.abs().max() < 10.0


class TestDrawdown:
    """Drawdown 계산 테스트."""

    def test_drawdown_is_non_positive(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HurstRegimeConfig,
    ):
        """drawdown은 항상 0 이하."""
        result = preprocess(sample_ohlcv_df, sample_config)

        drawdown = result["drawdown"].dropna()
        assert (drawdown <= 0).all()
