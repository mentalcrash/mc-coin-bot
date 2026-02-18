"""Extended Derivatives Data Service — batch SSOT, Silver load, OHLCV enrichment.

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
from src.data.deriv_ext.fetcher import COINALYZE_DATASETS, HYPERLIQUID_DATASETS
from src.data.deriv_ext.storage import DerivExtSilverProcessor

if TYPE_CHECKING:
    from src.catalog.store import DataCatalogStore

# ---------------------------------------------------------------------------
# Batch Definitions (SSOT)
# ---------------------------------------------------------------------------

_COINALYZE_DEFS: list[tuple[str, str]] = [("coinalyze", name) for name in COINALYZE_DATASETS]
_HYPERLIQUID_DEFS: list[tuple[str, str]] = [("hyperliquid", name) for name in HYPERLIQUID_DATASETS]

DERIV_EXT_BATCH_DEFINITIONS: dict[str, list[tuple[str, str]]] = {
    "coinalyze": _COINALYZE_DEFS,
    "hyperliquid": _HYPERLIQUID_DEFS,
}

# ---------------------------------------------------------------------------
# Publication Lag
# ---------------------------------------------------------------------------

SOURCE_LAG_DAYS: dict[str, int] = {
    "coinalyze": 0,  # Near real-time
    "hyperliquid": 0,  # Snapshot, no lag
}

# ---------------------------------------------------------------------------
# Date Column Mapping
# ---------------------------------------------------------------------------

SOURCE_DATE_COLUMNS: dict[str, str] = {
    "coinalyze": "date",
    "hyperliquid": "date",
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
# Per-asset precompute definitions
# ---------------------------------------------------------------------------

# (source, name, columns, rename_map)
_BTC_PRECOMPUTE: list[tuple[str, str, list[str], dict[str, str]]] = [
    ("coinalyze", "btc_agg_oi", ["close"], {"close": "dext_agg_oi_close"}),
    ("coinalyze", "btc_agg_funding", ["close"], {"close": "dext_agg_funding_close"}),
    (
        "coinalyze",
        "btc_liquidations",
        ["long_volume", "short_volume"],
        {"long_volume": "dext_liq_long_vol", "short_volume": "dext_liq_short_vol"},
    ),
    ("coinalyze", "btc_cvd", ["buy_volume"], {"buy_volume": "dext_cvd_buy_vol"}),
    # Hyperliquid (coin-filtered at load time)
    (
        "hyperliquid",
        "hl_asset_contexts",
        ["open_interest", "funding"],
        {"open_interest": "dext_hl_oi", "funding": "dext_hl_funding"},
    ),
]

_ETH_PRECOMPUTE: list[tuple[str, str, list[str], dict[str, str]]] = [
    ("coinalyze", "eth_agg_oi", ["close"], {"close": "dext_agg_oi_close"}),
    ("coinalyze", "eth_agg_funding", ["close"], {"close": "dext_agg_funding_close"}),
    (
        "coinalyze",
        "eth_liquidations",
        ["long_volume", "short_volume"],
        {"long_volume": "dext_liq_long_vol", "short_volume": "dext_liq_short_vol"},
    ),
    ("coinalyze", "eth_cvd", ["buy_volume"], {"buy_volume": "dext_cvd_buy_vol"}),
    # Hyperliquid (coin-filtered at load time)
    (
        "hyperliquid",
        "hl_asset_contexts",
        ["open_interest", "funding"],
        {"open_interest": "dext_hl_oi", "funding": "dext_hl_funding"},
    ),
]

ASSET_PRECOMPUTE_DEFS: dict[str, list[tuple[str, str, list[str], dict[str, str]]]] = {
    "BTC": _BTC_PRECOMPUTE,
    "ETH": _ETH_PRECOMPUTE,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DerivExtDataService:
    """Extended Derivatives 데이터 서비스.

    - Batch definitions SSOT 제공
    - Silver 데이터 로드
    - OHLCV + deriv_ext merge_asof 병합

    Example:
        >>> service = DerivExtDataService()
        >>> defs = service.get_batch_definitions("coinalyze")
        >>> df = service.load("coinalyze", "btc_agg_oi")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        silver_processor: DerivExtSilverProcessor | None = None,
        catalog: DataCatalogStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._silver = silver_processor or DerivExtSilverProcessor(self._settings)
        self._catalog = catalog or _try_load_catalog()

    def get_batch_definitions(self, batch_type: str) -> list[tuple[str, str]]:
        """batch_type에 해당하는 (source, name) 리스트 반환.

        Args:
            batch_type: "coinalyze", "all"

        Returns:
            (source, name) 튜플 리스트

        Raises:
            ValueError: 알 수 없는 batch_type
        """
        if batch_type == "all":
            result: list[tuple[str, str]] = []
            for defs_list in DERIV_EXT_BATCH_DEFINITIONS.values():
                result.extend(defs_list)
            return result

        if batch_type not in DERIV_EXT_BATCH_DEFINITIONS:
            valid = ", ".join([*DERIV_EXT_BATCH_DEFINITIONS.keys(), "all"])
            msg = f"Unknown batch type: {batch_type}. Valid: {valid}"
            raise ValueError(msg)

        return list(DERIV_EXT_BATCH_DEFINITIONS[batch_type])

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
        """Extended Derivatives 데이터를 OHLCV에 merge_asof로 병합.

        Args:
            ohlcv_df: OHLCV DataFrame (DatetimeIndex)
            source: 데이터 소스
            name: 데이터 이름
            columns: 병합할 컬럼 (None이면 전체)
            lag_days: publication lag 일수

        Returns:
            병합된 DataFrame
        """
        dext_df = self._silver.load(source, name)

        if dext_df.empty:
            logger.warning("No deriv_ext data for {}/{}, returning original OHLCV", source, name)
            return ohlcv_df

        date_col = get_date_col(source)
        if date_col in dext_df.columns:
            dext_df = dext_df.set_index(date_col)
        if not isinstance(dext_df.index, pd.DatetimeIndex):
            dext_df.index = pd.to_datetime(dext_df.index, utc=True)

        # 컬럼 선택
        if columns is not None:
            available = [c for c in columns if c in dext_df.columns]
        else:
            exclude = {"source", "symbol"}
            available = [c for c in dext_df.columns if c not in exclude]

        if not available:
            return ohlcv_df

        dext_subset = dext_df[available].copy()
        dext_subset = dext_subset.sort_index()

        # Publication lag shift
        lag = lag_days if lag_days is not None else self._resolve_lag_days(source)
        if lag > 0:
            dext_subset.index = dext_subset.index + pd.Timedelta(days=lag)

        result = pd.merge_asof(
            ohlcv_df,
            dext_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        logger.debug(
            "Enriched OHLCV with {} deriv_ext columns from {}/{}", len(available), source, name
        )
        return result

    def precompute(
        self,
        ohlcv_index: pd.DatetimeIndex | pd.Index,
        asset: str = "BTC",
    ) -> pd.DataFrame:
        """EDA 백테스트용 deriv_ext 데이터 사전 계산.

        asset에 맞는 데이터셋을 로드하여 dext_* prefix로 rename하고,
        ohlcv_index 기준으로 순차 merge_asof합니다.

        Args:
            ohlcv_index: OHLCV DatetimeIndex (merge 기준)
            asset: 대상 자산 ("BTC" or "ETH")

        Returns:
            dext_* 컬럼이 포함된 DataFrame (DatetimeIndex)
        """
        result = pd.DataFrame(index=ohlcv_index)

        precompute_defs = ASSET_PRECOMPUTE_DEFS.get(asset.upper(), _BTC_PRECOMPUTE)

        for source, name, columns, rename_map in precompute_defs:
            part = self._load_and_prepare(source, name, columns, rename_map, asset=asset.upper())
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
        asset: str | None = None,
    ) -> pd.DataFrame | None:
        """단일 source/name 데이터를 로드→컬럼 선택→rename→lag shift.

        Args:
            source: 데이터 소스
            name: 데이터 이름
            columns: 선택할 컬럼
            rename_map: 컬럼 rename 맵
            asset: coin 필터 (Hyperliquid 등 multi-row snapshot용)

        Returns:
            준비된 DataFrame 또는 None (데이터 없음)
        """
        df = self._load_silver(source, name, asset)
        if df is None:
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

        # dext_* prefix rename
        col_rename = {col: rename_map.get(col, f"dext_{col.lower()}") for col in available}
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

    def _load_silver(self, source: str, name: str, asset: str | None) -> pd.DataFrame | None:
        """Silver 데이터 로드 + coin 필터링.

        Returns:
            DataFrame 또는 None (데이터 없음/로드 실패)
        """
        try:
            df = self._silver.load(source, name)
        except Exception:
            return None
        if df.empty:
            return None

        # Multi-row snapshot (e.g., Hyperliquid): filter by coin
        if asset and "coin" in df.columns:
            df = df.loc[df["coin"] == asset].copy()
            if df.empty:
                return None

        return df
