"""VolumeMatrix — 백테스트용 일별 거래대금 사전 계산.

1m OHLCV에서 일별 quote volume을 집계하고,
rolling window 기반 거래대금 순위를 제공합니다.

Rules Applied:
    - #10 Python Standards: dataclass, type hints
    - #12 Data Engineering: UTC, DatetimeIndex
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd


@dataclass(frozen=True)
class VolumeMatrix:
    """일별 거래대금 매트릭스.

    Attributes:
        daily_volume: {symbol: Series(DatetimeIndex daily, values=quote_volume_usd)}
    """

    daily_volume: dict[str, pd.Series]

    @property
    def symbols(self) -> list[str]:
        """등록된 심볼 목록."""
        return list(self.daily_volume.keys())


def compute_volume_matrix(
    ohlcv_dict: dict[str, pd.DataFrame],
) -> VolumeMatrix:
    """1m OHLCV dict → 일별 quote volume 집계 → VolumeMatrix.

    Quote volume = close * volume (1m bar 기준).
    일별 합산으로 집계합니다.

    Args:
        ohlcv_dict: {symbol: DataFrame(index=DatetimeIndex, columns=[open,high,low,close,volume])}

    Returns:
        VolumeMatrix
    """
    import pandas as pd

    daily_volume: dict[str, pd.Series] = {}

    for symbol, df in ohlcv_dict.items():
        if df.empty:
            continue
        quote_vol = df["close"].astype(float) * df["volume"].astype(float)
        daily = pd.Series(quote_vol.resample("1D").sum())
        # NaN/0 제거 (거래 없는 날)
        daily = pd.Series(daily[daily > 0])
        if len(daily) > 0:
            daily_volume[symbol] = daily

    return VolumeMatrix(daily_volume=daily_volume)


def rank_at(
    matrix: VolumeMatrix,
    timestamp: datetime,
    rolling_window_days: int = 7,
    top_n: int = 20,
) -> list[str]:
    """특정 시점의 rolling volume 상위 N개 심볼 반환.

    Args:
        matrix: VolumeMatrix
        timestamp: 기준 시점
        rolling_window_days: rolling window (일)
        top_n: 상위 N개

    Returns:
        거래대금 내림차순 심볼 리스트
    """
    import pandas as pd

    ts = pd.Timestamp(timestamp)
    volumes: dict[str, float] = {}

    for symbol, daily in matrix.daily_volume.items():
        # timestamp 이전 rolling_window_days 기간의 합
        end = ts
        start = ts - pd.Timedelta(days=rolling_window_days)
        mask = (daily.index > start) & (daily.index <= end)
        window = daily[mask]
        if len(window) > 0:
            volumes[symbol] = float(window.sum())

    # 내림차순 정렬 → 상위 top_n
    ranked = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in ranked[:top_n]]
