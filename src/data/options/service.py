"""Options Data Service — batch SSOT, Silver load, OHLCV enrichment.

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
from src.data.options.fetcher import DERIBIT_DATASETS
from src.data.options.storage import OptionsSilverProcessor

if TYPE_CHECKING:
    from src.catalog.store import DataCatalogStore

# ---------------------------------------------------------------------------
# Batch Definitions (SSOT)
# ---------------------------------------------------------------------------

_DERIBIT_DEFS: list[tuple[str, str]] = [("deribit", name) for name in DERIBIT_DATASETS]

OPTIONS_BATCH_DEFINITIONS: dict[str, list[tuple[str, str]]] = {
    "deribit": _DERIBIT_DEFS,
}

# ---------------------------------------------------------------------------
# Publication Lag
# ---------------------------------------------------------------------------

SOURCE_LAG_DAYS: dict[str, int] = {
    "deribit": 0,  # Real-time
}

# ---------------------------------------------------------------------------
# Date Column Mapping
# ---------------------------------------------------------------------------

SOURCE_DATE_COLUMNS: dict[str, str] = {
    "deribit": "date",
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


class OptionsDataService:
    """Options 데이터 서비스.

    - Batch definitions SSOT 제공
    - Silver 데이터 로드
    - OHLCV + options merge_asof 병합

    Example:
        >>> service = OptionsDataService()
        >>> defs = service.get_batch_definitions("deribit")
        >>> df = service.load("deribit", "btc_dvol")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        silver_processor: OptionsSilverProcessor | None = None,
        catalog: DataCatalogStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._silver = silver_processor or OptionsSilverProcessor(self._settings)
        self._catalog = catalog or _try_load_catalog()

    def get_batch_definitions(self, batch_type: str) -> list[tuple[str, str]]:
        """batch_type에 해당하는 (source, name) 리스트 반환.

        Args:
            batch_type: "deribit", "all"

        Returns:
            (source, name) 튜플 리스트

        Raises:
            ValueError: 알 수 없는 batch_type
        """
        if batch_type == "all":
            result: list[tuple[str, str]] = []
            for defs_list in OPTIONS_BATCH_DEFINITIONS.values():
                result.extend(defs_list)
            return result

        if batch_type not in OPTIONS_BATCH_DEFINITIONS:
            valid = ", ".join([*OPTIONS_BATCH_DEFINITIONS.keys(), "all"])
            msg = f"Unknown batch type: {batch_type}. Valid: {valid}"
            raise ValueError(msg)

        return list(OPTIONS_BATCH_DEFINITIONS[batch_type])

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
        """Options 데이터를 OHLCV에 merge_asof로 병합.

        Args:
            ohlcv_df: OHLCV DataFrame (DatetimeIndex)
            source: 데이터 소스
            name: 데이터 이름
            columns: 병합할 컬럼 (None이면 전체)
            lag_days: publication lag 일수

        Returns:
            병합된 DataFrame
        """
        opt_df = self._silver.load(source, name)

        if opt_df.empty:
            logger.warning("No options data for {}/{}, returning original OHLCV", source, name)
            return ohlcv_df

        date_col = get_date_col(source)
        if date_col in opt_df.columns:
            opt_df = opt_df.set_index(date_col)
        if not isinstance(opt_df.index, pd.DatetimeIndex):
            opt_df.index = pd.to_datetime(opt_df.index, utc=True)

        # 컬럼 선택
        if columns is not None:
            available = [c for c in columns if c in opt_df.columns]
        else:
            exclude = {"source", "currency"}
            available = [c for c in opt_df.columns if c not in exclude]

        if not available:
            return ohlcv_df

        opt_subset = opt_df[available].copy()
        opt_subset = opt_subset.sort_index()

        # Publication lag shift
        lag = lag_days if lag_days is not None else self._resolve_lag_days(source)
        if lag > 0:
            opt_subset.index = opt_subset.index + pd.Timedelta(days=lag)

        result = pd.merge_asof(
            ohlcv_df,
            opt_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        logger.debug(
            "Enriched OHLCV with {} options columns from {}/{}", len(available), source, name
        )
        return result

    def precompute(
        self,
        ohlcv_index: pd.DatetimeIndex | pd.Index,
    ) -> pd.DataFrame:
        """EDA 백테스트용 options 데이터 사전 계산.

        모든 options 데이터는 global scope (symbol 무관 동일).
        Silver 데이터를 로드하여 opt_* prefix로 rename하고,
        ohlcv_index 기준으로 순차 merge_asof합니다.

        Args:
            ohlcv_index: OHLCV DatetimeIndex (merge 기준)

        Returns:
            opt_* 컬럼이 포함된 DataFrame (DatetimeIndex)
        """
        result = pd.DataFrame(index=ohlcv_index)

        # Precompute definitions: (source, name, columns, rename_map)
        precompute_defs: list[tuple[str, str, list[str], dict[str, str]]] = [
            ("deribit", "btc_dvol", ["close"], {"close": "opt_btc_dvol"}),
            ("deribit", "eth_dvol", ["close"], {"close": "opt_eth_dvol"}),
            ("deribit", "btc_pc_ratio", ["pc_ratio"], {"pc_ratio": "opt_btc_pc_ratio"}),
            ("deribit", "btc_hist_vol", ["vol_30d"], {"vol_30d": "opt_btc_rv30d"}),
            ("deribit", "btc_term_structure", ["slope"], {"slope": "opt_btc_term_slope"}),
        ]

        for source, name, columns, rename_map in precompute_defs:
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

    def _resolve_lag_days(self, source: str) -> int:
        """source의 publication lag 일수 조회."""
        if self._catalog is not None:
            try:
                return self._catalog.get_lag_days(source)
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

        # opt_* prefix rename
        col_rename: dict[str, str] = {}
        for col in available:
            if col in rename_map:
                col_rename[col] = rename_map[col]
            else:
                col_rename[col] = f"opt_{col.lower()}"
        subset.columns = pd.Index([col_rename.get(c, c) for c in subset.columns])

        # Publication lag shift
        lag = self._resolve_lag_days(source)
        if lag > 0:
            subset.index = subset.index + pd.Timedelta(days=lag)

        subset = subset.sort_index()
        # float64 coercion
        for col in subset.columns:
            if subset[col].dtype == object:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")

        return subset
