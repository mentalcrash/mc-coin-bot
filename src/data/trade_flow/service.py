"""Trade Flow Data Service — Silver 로드 + OHLCV enrichment.

Silver Parquet에 저장된 12H trade flow 피처를 로드하고,
OHLCV DataFrame의 인덱스에 맞춰 merge_asof로 정렬한다.
"""

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings

TFLOW_COLUMNS = [
    "tflow_cvd",
    "tflow_buy_ratio",
    "tflow_intensity",
    "tflow_large_ratio",
    "tflow_abs_order_imbalance",
    "tflow_vpin",
]


class TradeFlowService:
    """Trade flow Silver 데이터 로드 + OHLCV enrichment."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Silver Parquet 로드.

        Args:
            symbol: 거래 심볼 (e.g., "BTC/USDT")
            year: 연도

        Returns:
            Trade flow 피처 DataFrame (12H frequency)

        Raises:
            FileNotFoundError: Silver 파일이 없을 경우
        """
        path = self._settings.get_trade_flow_silver_path(symbol, year)
        if not path.exists():
            msg = f"Trade flow Silver not found: {path}"
            raise FileNotFoundError(msg)
        df = pd.read_parquet(path)
        logger.debug(f"Trade flow loaded: {path} ({len(df)} bars)")
        return df

    def precompute(
        self,
        ohlcv_index: pd.DatetimeIndex,
        symbol: str,
    ) -> pd.DataFrame:
        """OHLCV index에 맞춰 trade flow 피처 정렬.

        merge_asof(direction='backward')를 사용하여
        각 OHLCV bar에 가장 최근의 12H trade flow 피처를 매핑.

        Args:
            ohlcv_index: OHLCV DataFrame의 DatetimeIndex
            symbol: 거래 심볼

        Returns:
            ohlcv_index에 정렬된 trade flow DataFrame.
            데이터 없으면 빈 DataFrame(컬럼 없음) 반환.
        """
        if ohlcv_index.empty:
            return pd.DataFrame(index=ohlcv_index)

        # 연도 범위에서 데이터 로드
        start_year: int = ohlcv_index[0].year  # type: ignore[union-attr]
        end_year: int = ohlcv_index[-1].year  # type: ignore[union-attr]
        years = range(start_year, end_year + 1)

        dfs: list[pd.DataFrame] = []
        for year in years:
            try:
                df = self.load(symbol, year)
                dfs.append(df)
            except FileNotFoundError:
                logger.debug(f"Trade flow data not found: {symbol} {year}")

        if not dfs:
            return pd.DataFrame(index=ohlcv_index)

        # 연도별 병합
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # tflow 컬럼만 선택
        available_cols = [c for c in TFLOW_COLUMNS if c in combined.columns]
        if not available_cols:
            return pd.DataFrame(index=ohlcv_index)

        combined = combined[available_cols]
        assert isinstance(combined, pd.DataFrame)  # column selection always returns DataFrame

        # UTC 보장
        combined_idx = combined.index
        if isinstance(combined_idx, pd.DatetimeIndex) and combined_idx.tz is None:
            combined.index = combined_idx.tz_localize("UTC")

        ohlcv_idx = ohlcv_index
        if ohlcv_idx.tz is None:
            ohlcv_idx = ohlcv_idx.tz_localize("UTC")

        # merge_asof: 각 OHLCV bar에 가장 최근 12H trade flow 매핑
        ohlcv_df = pd.DataFrame(index=ohlcv_idx)
        result: pd.DataFrame = pd.merge_asof(
            ohlcv_df,
            combined,  # type: ignore[arg-type]
            left_index=True,
            right_index=True,
            direction="backward",
        )

        filled = int(result.notna().all(axis=1).sum())  # type: ignore[arg-type]
        logger.debug(
            "Trade flow precompute: {} columns, {}/{} rows filled",
            len(available_cols),
            filled,
            len(result),
        )

        return result
