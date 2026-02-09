"""Regime-Adaptive TSMOM Preprocessor.

레짐 컬럼 추가 + 기존 TSMOM 지표 계산을 결합합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Copy-on-write
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.regime.detector import add_regime_columns
from src.strategy.tsmom.preprocessor import preprocess as tsmom_preprocess

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.regime_tsmom.config import RegimeTSMOMConfig


def preprocess(
    df: pd.DataFrame,
    config: RegimeTSMOMConfig,
) -> pd.DataFrame:
    """Regime-Adaptive TSMOM 전처리.

    1. 레짐 컬럼 추가 (공유 인프라 사용)
    2. 기본 TSMOM 지표 계산 (기존 preprocessor 재사용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: RegimeTSMOMConfig 설정

    Returns:
        레짐 + TSMOM 지표가 추가된 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 1. 레짐 컬럼 추가
    result = add_regime_columns(df, config.regime)

    # 2. TSMOM 지표 계산 (기존 preprocessor 재사용)
    tsmom_config = config.to_tsmom_config()
    tsmom_result = tsmom_preprocess(result, tsmom_config)

    return tsmom_result
