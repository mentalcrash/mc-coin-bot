"""Tests for RSI Crossover Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.rsi_crossover.config import RSICrossoverConfig
from src.strategy.rsi_crossover.preprocessor import preprocess


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 모두 추가되는지 확인."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = ["rsi", "returns", "realized_vol", "vol_scalar", "atr", "drawdown"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 OHLCV 컬럼이 보존되는지 확인."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns, f"Original column {col} missing"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 수정되지 않음."""
        config = RSICrossoverConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_output_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """출력 길이가 입력과 동일."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_rsi_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI 값이 0-100 범위 (워밍업 이후)."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)

        rsi_valid = result["rsi"].dropna()
        assert len(rsi_valid) > 0, "No valid RSI values"
        assert (rsi_valid >= 0).all(), f"RSI below 0: {rsi_valid.min()}"
        assert (rsi_valid <= 100).all(), f"RSI above 100: {rsi_valid.max()}"

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar는 항상 양수."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR은 항상 양수."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """Drawdown은 0 이하."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame) -> None:
        """워밍업 기간 이후 주요 컬럼에 NaN 없음."""
        config = RSICrossoverConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["rsi", "vol_scalar", "atr"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        config = RSICrossoverConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_single_column_raises(self) -> None:
        """하나의 필수 컬럼만 누락되어도 ValueError."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="4h")
        df = pd.DataFrame(
            {
                "open": np.random.randn(n),
                "high": np.random.randn(n),
                "low": np.random.randn(n),
                "close": np.random.randn(n),
                # volume 누락
            },
            index=dates,
        )
        config = RSICrossoverConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestPreprocessWithDifferentConfigs:
    """다른 설정에서의 전처리 테스트."""

    def test_different_rsi_period(self, sample_ohlcv: pd.DataFrame) -> None:
        """다른 RSI period에서도 정상 작동."""
        config = RSICrossoverConfig(rsi_period=10)
        result = preprocess(sample_ohlcv, config)
        assert "rsi" in result.columns

    def test_different_vol_window(self, sample_ohlcv: pd.DataFrame) -> None:
        """다른 vol_window에서도 정상 작동."""
        config = RSICrossoverConfig(vol_window=20)
        result = preprocess(sample_ohlcv, config)
        assert "realized_vol" in result.columns

    def test_higher_vol_target_higher_scalar(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_target이 높으면 vol_scalar도 높아짐."""
        config_low = RSICrossoverConfig(vol_target=0.10)
        config_high = RSICrossoverConfig(vol_target=0.50)

        result_low = preprocess(sample_ohlcv, config_low)
        result_high = preprocess(sample_ohlcv, config_high)

        avg_low = result_low["vol_scalar"].dropna().mean()
        avg_high = result_high["vol_scalar"].dropna().mean()
        assert avg_high > avg_low
