"""Tests for Funding Rate Carry preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import funding_rate_ma, funding_zscore
from src.strategy.funding_carry.config import FundingCarryConfig
from src.strategy.funding_carry.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_with_funding() -> pd.DataFrame:
    """샘플 OHLCV + funding_rate DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200
    close = 50000.0 + np.cumsum(np.random.randn(n) * 300)
    funding_rate = np.random.randn(n) * 0.0003  # typical funding rate range

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestFundingRateMa:
    """funding_rate_ma 함수 테스트."""

    def test_basic_rolling_mean(self) -> None:
        """기본 Rolling Mean 계산."""
        fr = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005])
        avg_fr = funding_rate_ma(fr, window=3)

        # window=3이므로 처음 2개는 NaN
        assert pd.isna(avg_fr.iloc[0])
        assert pd.isna(avg_fr.iloc[1])
        # (0.001 + 0.002 + 0.003) / 3
        assert avg_fr.iloc[2] == pytest.approx(0.002, rel=1e-6)
        # (0.002 + 0.003 + 0.004) / 3
        assert avg_fr.iloc[3] == pytest.approx(0.003, rel=1e-6)

    def test_window_1(self) -> None:
        """window=1일 때 원본 값과 동일."""
        fr = pd.Series([0.001, -0.002, 0.003])
        avg_fr = funding_rate_ma(fr, window=1)

        assert avg_fr.iloc[0] == pytest.approx(0.001, rel=1e-6)
        assert avg_fr.iloc[1] == pytest.approx(-0.002, rel=1e-6)
        assert avg_fr.iloc[2] == pytest.approx(0.003, rel=1e-6)

    def test_returns_series(self) -> None:
        """반환 타입이 pd.Series인지 확인."""
        fr = pd.Series([0.001, 0.002, 0.003])
        avg_fr = funding_rate_ma(fr, window=2)
        assert isinstance(avg_fr, pd.Series)


class TestFundingZscore:
    """funding_zscore 함수 테스트."""

    def test_zscore_values(self) -> None:
        """Z-score 값 범위 확인."""
        np.random.seed(42)
        fr = pd.Series(np.random.randn(200) * 0.0003)
        result = funding_zscore(fr, ma_window=3, zscore_window=50)

        # NaN 제거 후 Z-score는 대략 -3 ~ 3 범위
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.abs().max() < 10  # 극단값이 아닌지 확인

    def test_zscore_nan_for_warmup(self) -> None:
        """워밍업 기간 동안 NaN 반환."""
        fr = pd.Series(np.random.randn(100) * 0.0003)
        result = funding_zscore(fr, ma_window=3, zscore_window=50)

        # ma_window + zscore_window 이전은 NaN
        assert result.iloc[:49].isna().all()

    def test_zscore_returns_series(self) -> None:
        """반환 타입이 pd.Series인지 확인."""
        fr = pd.Series(np.random.randn(100) * 0.0003)
        result = funding_zscore(fr, ma_window=3, zscore_window=50)
        assert isinstance(result, pd.Series)


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_columns(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 추가되는지 확인."""
        config = FundingCarryConfig()
        processed = preprocess(sample_ohlcv_with_funding, config)

        expected_cols = [
            "returns",
            "realized_vol",
            "avg_funding_rate",
            "funding_zscore",
            "vol_scalar",
            "atr",
        ]
        for col in expected_cols:
            assert col in processed.columns, f"Missing column: {col}"

    def test_missing_funding_rate_raises(self) -> None:
        """funding_rate 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100.0],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        config = FundingCarryConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "volume": [100.0],
                "funding_rate": [0.0001],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        config = FundingCarryConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """vol_scalar는 NaN이 아닌 곳에서 양수."""
        config = FundingCarryConfig()
        processed = preprocess(sample_ohlcv_with_funding, config)

        vol_scalar: pd.Series = processed["vol_scalar"]  # type: ignore[assignment]
        valid = vol_scalar.dropna()
        assert (valid > 0).all()

    def test_numeric_conversion(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """Decimal 타입이 float64로 변환되는지 확인."""
        from decimal import Decimal

        df = sample_ohlcv_with_funding.copy()
        df["close"] = df["close"].apply(Decimal)
        df["funding_rate"] = df["funding_rate"].apply(Decimal)

        config = FundingCarryConfig()
        processed = preprocess(df, config)

        assert processed["close"].dtype in [np.float64, float]

    def test_returns_calculated(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """수익률이 올바르게 계산되는지 확인."""
        config = FundingCarryConfig(use_log_returns=True)
        processed = preprocess(sample_ohlcv_with_funding, config)

        returns: pd.Series = processed["returns"]  # type: ignore[assignment]
        # 첫 번째 값은 NaN (shift 때문)
        assert pd.isna(returns.iloc[0])
        # 나머지는 값이 있어야 함
        assert returns.iloc[1:].notna().all()

    def test_preserves_original(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """원본 DataFrame이 수정되지 않는지 확인."""
        original_cols = set(sample_ohlcv_with_funding.columns)
        config = FundingCarryConfig()
        _processed = preprocess(sample_ohlcv_with_funding, config)

        assert set(sample_ohlcv_with_funding.columns) == original_cols

    def test_atr_calculated(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """ATR이 계산되는지 확인."""
        config = FundingCarryConfig()
        processed = preprocess(sample_ohlcv_with_funding, config)

        atr: pd.Series = processed["atr"]  # type: ignore[assignment]
        valid = atr.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()
