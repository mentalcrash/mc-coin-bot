"""Tests for src.market.utils."""

from decimal import Decimal

import numpy as np
import pandas as pd

from src.market.utils import coerce_ohlcv_to_float64


class TestCoerceOhlcvToFloat64:
    """coerce_ohlcv_to_float64 테스트."""

    def test_decimal_to_float64(self) -> None:
        """Decimal 값이 float64로 변환된다."""
        df = pd.DataFrame(
            {
                "open": [Decimal("100.5"), Decimal("101.0")],
                "high": [Decimal("102.0"), Decimal("103.0")],
                "low": [Decimal("99.0"), Decimal("100.0")],
                "close": [Decimal("101.0"), Decimal("102.0")],
                "volume": [Decimal(1000), Decimal(2000)],
            }
        )
        result = coerce_ohlcv_to_float64(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert result[col].dtype == np.float64

    def test_original_not_modified(self) -> None:
        """원본 DataFrame이 변경되지 않는다."""
        df = pd.DataFrame(
            {
                "open": [Decimal("100.5")],
                "close": [Decimal("101.0")],
                "volume": [Decimal(1000)],
            }
        )
        original_types = {col: df[col].dtype for col in df.columns}
        coerce_ohlcv_to_float64(df)
        for col in df.columns:
            assert df[col].dtype == original_types[col]

    def test_already_float64(self) -> None:
        """이미 float64인 경우 정상 작동."""
        df = pd.DataFrame(
            {
                "open": [100.5, 101.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 2000.0],
            }
        )
        result = coerce_ohlcv_to_float64(df)
        assert result["close"].dtype == np.float64

    def test_missing_columns_ignored(self) -> None:
        """OHLCV 중 일부 컬럼만 있어도 동작."""
        df = pd.DataFrame({"close": [Decimal("100.0")], "extra": ["abc"]})
        result = coerce_ohlcv_to_float64(df)
        assert result["close"].dtype == np.float64
        assert result["extra"].iloc[0] == "abc"

    def test_values_preserved(self) -> None:
        """변환 후 값이 보존된다."""
        df = pd.DataFrame({"close": [Decimal("123.456789")]})
        result = coerce_ohlcv_to_float64(df)
        assert abs(result["close"].iloc[0] - 123.456789) < 1e-6
