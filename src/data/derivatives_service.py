"""Derivatives Data Service — Silver 로드 + OHLCV 병합.

Silver _deriv 파일을 로드하고 OHLCV DataFrame과 merge_asof로 정렬합니다.
Backtest/EDA 모두에서 사용됩니다.

Rules Applied:
    - Repository Pattern: 데이터 접근 추상화
    - #12 Data Engineering: Vectorized merge_asof
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.data.derivatives_storage import DerivativesSilverProcessor

if TYPE_CHECKING:
    from datetime import datetime

# 기본 enrich 컬럼
DEFAULT_DERIV_COLUMNS = [
    "funding_rate",
    "open_interest",
    "ls_ratio",
    "taker_ratio",
]


class DerivativesDataService:
    """파생상품 데이터 서비스.

    Silver _deriv 파일을 로드하고 OHLCV DataFrame에 merge_asof로 병합합니다.

    Example:
        >>> service = DerivativesDataService()
        >>> enriched = service.enrich(ohlcv_df, "BTC/USDT", start, end)
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        silver_processor: DerivativesSilverProcessor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._silver = silver_processor or DerivativesSilverProcessor(self._settings)

    def load(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Silver _deriv 파일 로드 + 기간 필터링.

        Args:
            symbol: 거래 심볼
            start: 시작 시각
            end: 종료 시각

        Returns:
            DatetimeIndex 기반 derivatives DataFrame
        """
        years = list(range(start.year, end.year + 1))
        dfs: list[pd.DataFrame] = []

        for year in years:
            if not self._silver.exists(symbol, year):
                logger.debug(f"No derivatives data for {symbol} {year}")
                continue
            df = self._silver.load(symbol, year)
            dfs.append(df)

        if not dfs:
            logger.warning(f"No derivatives data found for {symbol}")
            return pd.DataFrame()

        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # tz-aware 필터링
        if isinstance(combined.index, pd.DatetimeIndex) and combined.index.tz is None:
            combined.index = combined.index.tz_localize("UTC")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")

        filtered = combined.loc[start_ts:end_ts]
        logger.debug(
            f"Loaded derivatives for {symbol}: {len(filtered)} rows ({start.date()} ~ {end.date()})"
        )
        return filtered

    def enrich(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str,
        start: datetime,
        end: datetime,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """OHLCV DataFrame에 derivatives 데이터를 merge_asof로 병합.

        Args:
            ohlcv_df: OHLCV DataFrame (DatetimeIndex)
            symbol: 거래 심볼
            start: 시작 시각
            end: 종료 시각
            columns: 병합할 컬럼 (None이면 DEFAULT_DERIV_COLUMNS)

        Returns:
            병합된 DataFrame (원본 OHLCV + derivatives 컬럼)
        """
        cols = columns or DEFAULT_DERIV_COLUMNS
        deriv_df = self.load(symbol, start, end)

        if deriv_df.empty:
            logger.warning(f"No derivatives data for {symbol}, returning original OHLCV")
            return ohlcv_df

        # merge에 필요한 컬럼만 선택
        available_cols = [c for c in cols if c in deriv_df.columns]
        if not available_cols:
            return ohlcv_df

        deriv_subset = deriv_df[available_cols].copy()

        # merge_asof: ohlcv 타임스탬프 기준 backward-fill
        result = pd.merge_asof(
            ohlcv_df,
            deriv_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        logger.debug(f"Enriched OHLCV with {len(available_cols)} derivatives columns for {symbol}")
        return result

    def precompute(
        self,
        symbol: str,
        ohlcv_index: pd.DatetimeIndex,
        start: datetime,
        end: datetime,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """EDA용: ohlcv_index에 맞춰 derivatives 데이터 정렬.

        merge_asof를 사용하여 각 OHLCV 타임스탬프에 가장 가까운
        이전 derivatives 값을 배치합니다.

        Args:
            symbol: 거래 심볼
            ohlcv_index: OHLCV 타임스탬프 인덱스
            start: 시작 시각
            end: 종료 시각
            columns: 사용할 컬럼 (None이면 DEFAULT_DERIV_COLUMNS)

        Returns:
            ohlcv_index에 정렬된 derivatives DataFrame
        """
        cols = columns or DEFAULT_DERIV_COLUMNS
        deriv_df = self.load(symbol, start, end)

        if deriv_df.empty:
            return pd.DataFrame(index=ohlcv_index, columns=pd.Index(cols))

        available_cols = [c for c in cols if c in deriv_df.columns]
        if not available_cols:
            return pd.DataFrame(index=ohlcv_index, columns=pd.Index(cols))

        deriv_subset = deriv_df[available_cols].copy()

        # 빈 OHLCV 프레임 + merge_asof
        ohlcv_frame = pd.DataFrame(index=ohlcv_index)
        result = pd.merge_asof(
            ohlcv_frame,
            deriv_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        logger.debug(
            f"Precomputed derivatives for {symbol}: {len(result)} rows, {len(available_cols)} columns"
        )
        return result
