"""Tests for BB+RSI Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.bb_rsi.config import BBRSIConfig
from src.strategy.bb_rsi.preprocessor import (
    calculate_atr,
    calculate_bb_position,
    calculate_bollinger_bands,
    calculate_rsi,
    preprocess,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """횡보 패턴의 샘플 OHLCV DataFrame (200일)."""
    np.random.seed(42)
    n = 200

    # 횡보 패턴: mean-reverting 가격
    base_price = 50000.0
    noise = np.cumsum(np.random.randn(n) * 300)
    # mean reversion force
    close = base_price + noise - noise.mean()
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


class TestBollingerBands:
    """볼린저밴드 계산 테스트."""

    def test_ordering(self, sample_ohlcv: pd.DataFrame):
        """lower < middle < upper 항상 성립."""
        upper, middle, lower = calculate_bollinger_bands(
            sample_ohlcv["close"], period=20, std_dev=2.0
        )
        valid = upper.dropna()
        assert (lower.loc[valid.index] <= middle.loc[valid.index]).all()
        assert (middle.loc[valid.index] <= upper.loc[valid.index]).all()

    def test_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        upper, middle, lower = calculate_bollinger_bands(
            sample_ohlcv["close"], period=20, std_dev=2.0
        )
        assert len(upper) == len(sample_ohlcv)
        assert len(middle) == len(sample_ohlcv)
        assert len(lower) == len(sample_ohlcv)

    def test_warmup_nan(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간에는 NaN."""
        upper, _, _ = calculate_bollinger_bands(sample_ohlcv["close"], period=20, std_dev=2.0)
        assert upper.iloc[:19].isna().all()
        assert upper.iloc[19:].notna().all()


class TestRSI:
    """RSI 계산 테스트."""

    def test_range(self, sample_ohlcv: pd.DataFrame):
        """RSI 값이 0-100 범위."""
        rsi = calculate_rsi(sample_ohlcv["close"], period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        rsi = calculate_rsi(sample_ohlcv["close"], period=14)
        assert len(rsi) == len(sample_ohlcv)


class TestATR:
    """ATR 계산 테스트."""

    def test_positive(self, sample_ohlcv: pd.DataFrame):
        """ATR은 항상 양수."""
        atr = calculate_atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], period=14
        )
        valid = atr.dropna()
        assert (valid > 0).all()


class TestBBPosition:
    """BB 위치 정규화 테스트."""

    def test_range_approximate(self, sample_ohlcv: pd.DataFrame):
        """bb_position이 대략 -1 ~ +1 범위 (대부분의 데이터)."""
        upper, middle, lower = calculate_bollinger_bands(
            sample_ohlcv["close"], period=20, std_dev=2.0
        )
        bb_pos = calculate_bb_position(sample_ohlcv["close"], upper, lower, middle)
        valid = bb_pos.dropna()
        # 대부분 -1 ~ +1 범위 (극단값은 밴드 밖)
        within_range = ((valid >= -1.5) & (valid <= 1.5)).mean()
        assert within_range > 0.90


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame):
        """필수 출력 컬럼이 모두 존재."""
        config = BBRSIConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "rsi",
            "bb_position",
            "atr",
            "returns",
            "realized_vol",
            "vol_scalar",
            "drawdown",
            "adx",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간 이후 NaN 없음."""
        config = BBRSIConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["bb_upper", "bb_middle", "bb_lower", "rsi", "vol_scalar"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame):
        """원본 DataFrame이 수정되지 않음."""
        config = BBRSIConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = BBRSIConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_adx_excluded_when_disabled(self, sample_ohlcv: pd.DataFrame):
        """ADX 필터 비활성화 시 adx 컬럼 미생성."""
        config = BBRSIConfig(use_adx_filter=False)
        result = preprocess(sample_ohlcv, config)
        assert "adx" not in result.columns

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame):
        """vol_scalar는 항상 양수."""
        config = BBRSIConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
