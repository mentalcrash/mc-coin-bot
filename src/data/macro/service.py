"""Macro Data Service — batch SSOT, Silver load, OHLCV enrichment.

Batch definitions(source, name) 중앙 관리 + Silver 로드 + merge_asof.
CLI와 EDA/Backtest 모두에서 사용됩니다.

Rules Applied:
    - Repository Pattern: 데이터 접근 추상화
    - #12 Data Engineering: Vectorized merge_asof
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.data.macro.fetcher import COINGECKO_DATASETS, FRED_SERIES, YFINANCE_TICKERS
from src.data.macro.storage import MacroSilverProcessor

if TYPE_CHECKING:
    from src.catalog.store import DataCatalogStore

# ---------------------------------------------------------------------------
# Batch Definitions (SSOT)
# ---------------------------------------------------------------------------

_FRED_DEFS: list[tuple[str, str]] = [("fred", name) for name in FRED_SERIES]
_YFINANCE_DEFS: list[tuple[str, str]] = [("yfinance", name) for name in YFINANCE_TICKERS]
_COINGECKO_DEFS: list[tuple[str, str]] = [("coingecko", name) for name in COINGECKO_DATASETS]

MACRO_BATCH_DEFINITIONS: dict[str, list[tuple[str, str]]] = {
    "fred": _FRED_DEFS,
    "yfinance": _YFINANCE_DEFS,
    "coingecko": _COINGECKO_DEFS,
}

# ---------------------------------------------------------------------------
# Publication Lag
# ---------------------------------------------------------------------------

SOURCE_LAG_DAYS: dict[str, int] = {
    "fred": 1,
    "yfinance": 0,
    "coingecko": 0,
}

# ---------------------------------------------------------------------------
# Date Column Mapping
# ---------------------------------------------------------------------------

SOURCE_DATE_COLUMNS: dict[str, str] = {
    "fred": "date",
    "yfinance": "date",
    "coingecko": "date",
}


def get_date_col(source: str) -> str:
    """source에 해당하는 date column 이름 반환."""
    return SOURCE_DATE_COLUMNS.get(source, "date")


# ---------------------------------------------------------------------------
# Catalog (optional — fallback to hardcoded constants)
# ---------------------------------------------------------------------------


def _try_load_catalog() -> DataCatalogStore | None:
    """DataCatalogStore 로드 시도, 실패 시 None."""
    try:
        from src.catalog.store import DataCatalogStore

        store = DataCatalogStore()
        store.load_all()
    except Exception:
        return None
    else:
        return store


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MacroDataService:
    """Macro 데이터 서비스.

    - Batch definitions SSOT 제공
    - Silver 데이터 로드
    - OHLCV + macro merge_asof 병합

    Example:
        >>> service = MacroDataService()
        >>> defs = service.get_batch_definitions("fred")
        >>> df = service.load("fred", "dxy")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        silver_processor: MacroSilverProcessor | None = None,
        catalog: DataCatalogStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._silver = silver_processor or MacroSilverProcessor(self._settings)
        self._catalog = catalog or _try_load_catalog()

    def get_batch_definitions(self, batch_type: str) -> list[tuple[str, str]]:
        """batch_type에 해당하는 (source, name) 리스트 반환.

        Args:
            batch_type: "fred", "yfinance", "all"

        Returns:
            (source, name) 튜플 리스트

        Raises:
            ValueError: 알 수 없는 batch_type
        """
        if batch_type == "all":
            result: list[tuple[str, str]] = []
            for defs_list in MACRO_BATCH_DEFINITIONS.values():
                result.extend(defs_list)
            return result

        if batch_type not in MACRO_BATCH_DEFINITIONS:
            valid = ", ".join([*MACRO_BATCH_DEFINITIONS.keys(), "all"])
            msg = f"Unknown batch type: {batch_type}. Valid: {valid}"
            raise ValueError(msg)

        return list(MACRO_BATCH_DEFINITIONS[batch_type])

    def load(self, source: str, name: str) -> pd.DataFrame:
        """Silver 데이터 로드.

        Args:
            source: 데이터 소스
            name: 데이터 이름

        Returns:
            pandas DataFrame
        """
        return self._silver.load(source, name)

    def enrich(
        self,
        ohlcv_df: pd.DataFrame,
        source: str,
        name: str,
        columns: list[str] | None = None,
        lag_days: int | None = None,
    ) -> pd.DataFrame:
        """Macro 데이터를 OHLCV에 merge_asof로 병합.

        Args:
            ohlcv_df: OHLCV DataFrame (DatetimeIndex)
            source: 데이터 소스
            name: 데이터 이름
            columns: 병합할 컬럼 (None이면 전체)
            lag_days: publication lag 일수

        Returns:
            병합된 DataFrame
        """
        macro_df = self._silver.load(source, name)

        if macro_df.empty:
            logger.warning("No macro data for {}/{}, returning original OHLCV", source, name)
            return ohlcv_df

        date_col = get_date_col(source)
        if date_col in macro_df.columns:
            macro_df = macro_df.set_index(date_col)
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df.index = pd.to_datetime(macro_df.index, utc=True)

        # 컬럼 선택
        if columns is not None:
            available = [c for c in columns if c in macro_df.columns]
        else:
            exclude = {"source", "name", "series_id", "ticker"}
            available = [c for c in macro_df.columns if c not in exclude]

        if not available:
            return ohlcv_df

        macro_subset = macro_df[available].copy()
        macro_subset = macro_subset.sort_index()

        # Publication lag shift
        lag = lag_days if lag_days is not None else self._resolve_lag_days(source, name)
        if lag > 0:
            macro_subset.index = macro_subset.index + pd.Timedelta(days=lag)

        result = pd.merge_asof(
            ohlcv_df,
            macro_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        logger.debug(
            "Enriched OHLCV with {} macro columns from {}/{}", len(available), source, name
        )
        return result

    def precompute(
        self,
        ohlcv_index: pd.DatetimeIndex | pd.Index,
    ) -> pd.DataFrame:
        """EDA 백테스트용 macro 데이터 사전 계산.

        모든 macro 데이터는 global scope (symbol 무관 동일).
        Silver 데이터를 로드하여 macro_* prefix로 rename하고,
        ohlcv_index 기준으로 순차 merge_asof합니다.

        Args:
            ohlcv_index: OHLCV DatetimeIndex (merge 기준)

        Returns:
            macro_* 컬럼이 포함된 DataFrame (DatetimeIndex)
        """
        from src.eda.onchain_feed import MACRO_GLOBAL_SOURCES

        result = pd.DataFrame(index=ohlcv_index)
        for source, name, columns, rename_map in MACRO_GLOBAL_SOURCES:
            part = self._load_and_prepare(source, name, columns, rename_map)
            if part is None:
                continue
            result = pd.merge_asof(
                result,
                part,
                left_index=True,
                right_index=True,
                direction="backward",
            )

        if result.columns.empty:
            return pd.DataFrame(index=ohlcv_index)

        return result

    def _resolve_lag_days(self, source: str, name: str) -> int:
        """source/name의 publication lag 일수 조회.

        catalog의 dataset-level lag_days → source-level lag_days → 하드코딩 fallback 순서.
        """
        if self._catalog is not None:
            try:
                dataset_id = f"{source}_{name}"
                return self._catalog.get_lag_days(source, dataset_id=dataset_id)
            except KeyError:
                pass
        return SOURCE_LAG_DAYS.get(source, 0)

    def _load_and_prepare(
        self,
        source: str,
        name: str,
        columns: list[str],
        rename_map: dict[str, str],
    ) -> pd.DataFrame | None:
        """단일 source/name 데이터를 로드→컬럼 선택→rename→lag shift.

        Returns:
            준비된 DataFrame 또는 None (데이터 없음)
        """
        try:
            df = self._silver.load(source, name)
        except Exception:
            return None
        if df.empty:
            return None

        date_col = get_date_col(source)
        if date_col in df.columns:
            df = df.set_index(date_col)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # 컬럼 선택
        available = [c for c in columns if c in df.columns]
        if not available:
            return None

        subset: pd.DataFrame = df[available].copy()  # type: ignore[assignment]

        # macro_* prefix rename
        col_rename: dict[str, str] = {}
        for col in available:
            if col in rename_map:
                col_rename[col] = rename_map[col]
            else:
                col_rename[col] = f"macro_{col.lower()}"
        subset.columns = pd.Index([col_rename.get(c, c) for c in subset.columns])

        # Publication lag shift
        lag = self._resolve_lag_days(source, name)
        if lag > 0:
            subset.index = subset.index + pd.Timedelta(days=lag)

        subset = subset.sort_index()
        # float64 coercion
        for col in subset.columns:
            if subset[col].dtype == object:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")

        return subset
