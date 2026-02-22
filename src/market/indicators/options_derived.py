"""Options-derived indicators (IV-RV spread, IV percentile rank).

옵션 데이터 기반 파생 지표 함수.
모든 함수는 stateless, vectorized이며 ``pd.Series`` in → ``pd.Series`` out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def iv_rv_spread(
    implied_vol: pd.Series,
    realized_vol: pd.Series,
) -> pd.Series:
    """IV-RV Spread (Variance Risk Premium proxy).

    양수: IV > RV → 옵션 프리미엄 비쌈 (공포).
    음수: IV < RV → 옵션 프리미엄 저렴 (안도).

    Args:
        implied_vol: 내재변동성 시리즈 (예: DVOL).
        realized_vol: 실현변동성 시리즈.

    Returns:
        IV - RV 스프레드 시리즈.
    """
    result: pd.Series = implied_vol - realized_vol  # type: ignore[assignment]
    return result


def iv_percentile_rank(
    implied_vol: pd.Series,
    window: int = 365,
) -> pd.Series:
    """IV Percentile Rank — 현재 IV가 과거 분포에서 몇 번째 백분위인지.

    1에 가까울수록 역사적 고점, 0에 가까울수록 역사적 저점.

    Args:
        implied_vol: 내재변동성 시리즈 (예: DVOL).
        window: 백분위 계산 윈도우 (기본 365일).

    Returns:
        IV 백분위 순위 시리즈 (0~1).
    """
    result: pd.Series = implied_vol.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).rank(pct=True)
    return result
