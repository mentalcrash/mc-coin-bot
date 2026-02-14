"""OHLCV data utilities.

전략 preprocessor에서 반복되는 Decimal→float64 변환 등 공통 유틸리티.
"""

from __future__ import annotations

import pandas as pd

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def coerce_ohlcv_to_float64(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 컬럼을 float64로 변환 (원본 비변경).

    Parquet에서 Decimal로 저장된 경우 ``np.log()`` 등이 작동하지 않으므로
    ``pd.to_numeric``으로 변환합니다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        float64로 변환된 복사본.
    """
    result = df.copy()
    for col in _OHLCV_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result
