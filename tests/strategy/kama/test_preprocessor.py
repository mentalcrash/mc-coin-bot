"""Tests for KAMA Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.kama.config import KAMAConfig
from src.strategy.kama.preprocessor import (
    calculate_atr,
    calculate_kama,
    preprocess,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """상승 추세 패턴의 샘플 OHLCV DataFrame (200일)."""
    np.random.seed(42)
    n = 200

    # 상승 추세 패턴: 추세 추종에 적합한 데이터
    base_price = 50000.0
    trend = np.linspace(0, 5000, n)  # 상승 바이어스
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + trend + noise
    close = np.maximum(close, base_price * 0.8)  # 바닥 제한

    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestKAMA:
    """KAMA 계산 테스트."""

    def test_kama_follows_price(self, sample_ohlcv: pd.DataFrame):
        """KAMA 값이 가격에 근접해야 함."""
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        kama = calculate_kama(close, er_lookback=10, fast_period=2, slow_period=30)

        # KAMA는 가격의 이동평균이므로 가격과 유사한 범위에 있어야 함
        valid_kama = kama.dropna()
        valid_close = close.loc[valid_kama.index]

        # 가격 대비 KAMA의 평균 편차가 10% 이내
        mean_deviation = ((valid_kama - valid_close) / valid_close).abs().mean()
        assert mean_deviation < 0.10

    def test_kama_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        kama = calculate_kama(close, er_lookback=10, fast_period=2, slow_period=30)
        assert len(kama) == len(sample_ohlcv)

    def test_kama_first_value_equals_close(self, sample_ohlcv: pd.DataFrame):
        """KAMA 첫 값은 종가와 동일."""
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        kama = calculate_kama(close, er_lookback=10, fast_period=2, slow_period=30)
        assert kama.iloc[0] == close.iloc[0]


class TestATR:
    """ATR 계산 테스트."""

    def test_positive(self, sample_ohlcv: pd.DataFrame):
        """ATR은 항상 양수."""
        atr = calculate_atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], period=14
        )
        valid = atr.dropna()
        assert (valid > 0).all()


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame):
        """필수 출력 컬럼이 모두 존재."""
        config = KAMAConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "kama",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간 이후 NaN 없음."""
        config = KAMAConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["kama", "atr", "vol_scalar"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame):
        """원본 DataFrame이 수정되지 않음."""
        config = KAMAConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = KAMAConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame):
        """vol_scalar는 항상 양수."""
        config = KAMAConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
