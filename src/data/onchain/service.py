"""On-chain Data Service — batch SSOT, Silver load, OHLCV enrichment.

Batch definitions(source, name) 중앙 관리 + Silver 로드 + merge_asof.
CLI와 EDA/Backtest 모두에서 사용됩니다.

Rules Applied:
    - Repository Pattern: 데이터 접근 추상화
    - #12 Data Engineering: Vectorized merge_asof
"""

from __future__ import annotations

import time

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.data.onchain.fetcher import (
    BC_CHARTS,
    CM_ASSETS,
    DEFI_CHAINS,
    STABLECOIN_IDS,
    OnchainFetcher,
)
from src.data.onchain.storage import OnchainSilverProcessor

# ---------------------------------------------------------------------------
# Batch Definitions (SSOT)
# ---------------------------------------------------------------------------

_STABLECOIN_DEFS: list[tuple[str, str]] = [
    ("defillama", "stablecoin_total"),
    *[("defillama", f"stablecoin_chain_{c}") for c in DEFI_CHAINS],
    *[("defillama", f"stablecoin_{name.lower()}") for name in STABLECOIN_IDS],
]

_TVL_DEFS: list[tuple[str, str]] = [
    ("defillama", "tvl_total"),
    *[("defillama", f"tvl_chain_{c}") for c in DEFI_CHAINS],
]

_DEX_DEFS: list[tuple[str, str]] = [
    ("defillama", "dex_volume"),
]

_COINMETRICS_DEFS: list[tuple[str, str]] = [
    *[("coinmetrics", f"{asset}_metrics") for asset in CM_ASSETS],
]

_SENTIMENT_DEFS: list[tuple[str, str]] = [
    ("alternative_me", "fear_greed"),
]

_BLOCKCHAIN_DEFS: list[tuple[str, str]] = [
    *[("blockchain_com", f"bc_{chart}") for chart in BC_CHARTS],
]

_ETHERSCAN_DEFS: list[tuple[str, str]] = [
    ("etherscan", "eth_supply"),
]

_MEMPOOL_DEFS: list[tuple[str, str]] = [
    ("mempool_space", "mining"),
]

ONCHAIN_BATCH_DEFINITIONS: dict[str, list[tuple[str, str]]] = {
    "stablecoin": _STABLECOIN_DEFS,
    "tvl": _TVL_DEFS,
    "dex": _DEX_DEFS,
    "coinmetrics": _COINMETRICS_DEFS,
    "sentiment": _SENTIMENT_DEFS,
    "blockchain": _BLOCKCHAIN_DEFS,
    "etherscan": _ETHERSCAN_DEFS,
    "mempool": _MEMPOOL_DEFS,
}


# ---------------------------------------------------------------------------
# Fetch Router
# ---------------------------------------------------------------------------


async def _route_defillama_stablecoin(fetcher: OnchainFetcher, name: str) -> pd.DataFrame | None:
    """DeFiLlama stablecoin 라우팅."""
    if name == "stablecoin_total":
        return await fetcher.fetch_stablecoin_total()
    if name.startswith("stablecoin_chain_"):
        chain = name.removeprefix("stablecoin_chain_")
        return await fetcher.fetch_stablecoin_by_chain(chain)
    if name.startswith("stablecoin_"):
        sc_name = name.removeprefix("stablecoin_").upper()
        sc_id = STABLECOIN_IDS.get(sc_name)
        if sc_id is None:
            msg = f"Unknown stablecoin: {sc_name}"
            raise ValueError(msg)
        return await fetcher.fetch_stablecoin_individual(sc_id, sc_name)
    return None


async def _route_defillama(fetcher: OnchainFetcher, name: str) -> pd.DataFrame | None:
    """DeFiLlama 데이터 라우팅 (내부 헬퍼).

    Returns:
        DataFrame or None if name not recognized.
    """
    if name.startswith("stablecoin"):
        return await _route_defillama_stablecoin(fetcher, name)
    if name == "tvl_total":
        return await fetcher.fetch_tvl()
    if name.startswith("tvl_chain_"):
        chain = name.removeprefix("tvl_chain_")
        return await fetcher.fetch_tvl(chain)
    if name == "dex_volume":
        return await fetcher.fetch_dex_volume()
    return None


# Source별 Silver processing 시 date column 이름 매핑
SOURCE_DATE_COLUMNS: dict[str, str] = {
    "defillama": "date",
    "coinmetrics": "time",
    "alternative_me": "timestamp",
    "blockchain_com": "timestamp",
    "etherscan": "timestamp",
    "mempool_space": "timestamp",
}

# Source별 publication lag (일): T+lag 이후 데이터 접근 가능 (look-ahead bias 방지)
SOURCE_LAG_DAYS: dict[str, int] = {
    "defillama": 1,  # TVL/Stablecoin: T+1 ~06:00 UTC 확정
    "coinmetrics": 1,  # T+1 ~00:00-04:00 UTC 가용
    "alternative_me": 1,  # 같은 날이지만 안전하게 T+1
    "blockchain_com": 1,  # T+1 ~12:00 UTC
    "etherscan": 0,  # 스냅샷 (near real-time)
    "mempool_space": 0,  # near real-time
}


def get_date_col(source: str) -> str:
    """source에 해당하는 date column 이름 반환."""
    return SOURCE_DATE_COLUMNS.get(source, "date")


async def _do_route_fetch(fetcher: OnchainFetcher, source: str, name: str) -> pd.DataFrame:
    """(source, name) 쌍을 적절한 fetcher 메서드로 라우팅 (내부 헬퍼).

    Raises:
        ValueError: 알 수 없는 source/name 조합
    """
    if source == "defillama":
        result = await _route_defillama(fetcher, name)
        if result is not None:
            return result
    elif source == "coinmetrics":
        asset = name.removesuffix("_metrics")
        return await fetcher.fetch_coinmetrics(asset)
    elif source == "alternative_me":
        if name == "fear_greed":
            return await fetcher.fetch_fear_greed()
    elif source == "blockchain_com":
        if name.startswith("bc_"):
            chart_name = name.removeprefix("bc_")
            return await fetcher.fetch_blockchain_chart(chart_name)
    elif source == "etherscan":
        if name == "eth_supply":
            api_key = get_settings().etherscan_api_key.get_secret_value()
            return await fetcher.fetch_eth_supply(api_key)
    elif source == "mempool_space":
        if name == "mining":
            return await fetcher.fetch_mempool_mining()

    msg = f"Unknown route: {source}/{name}"
    raise ValueError(msg)


async def route_fetch(
    fetcher: OnchainFetcher,
    source: str,
    name: str,
    metrics_callback: object | None = None,
) -> pd.DataFrame:
    """(source, name) 쌍을 적절한 fetcher 메서드로 라우팅.

    Args:
        fetcher: OnchainFetcher 인스턴스
        source: 데이터 소스 (defillama, coinmetrics, alternative_me, ...)
        name: 데이터 이름 (stablecoin_total, tvl_chain_Ethereum, ...)
        metrics_callback: OnchainMetricsCallback 구현체 (선택적 계측)

    Returns:
        Fetched DataFrame

    Raises:
        ValueError: 알 수 없는 source/name 조합
    """
    start = time.monotonic()
    try:
        df = await _do_route_fetch(fetcher, source, name)
    except Exception:
        elapsed = time.monotonic() - start
        if metrics_callback is not None:
            metrics_callback.on_fetch(source, name, elapsed, "failure", 0)  # type: ignore[union-attr]
        raise
    else:
        elapsed = time.monotonic() - start
        if metrics_callback is not None:
            status = "empty" if df.empty else "success"
            metrics_callback.on_fetch(source, name, elapsed, status, len(df))  # type: ignore[union-attr]
        return df


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OnchainDataService:
    """On-chain 데이터 서비스.

    - Batch definitions SSOT 제공
    - Silver 데이터 로드
    - OHLCV + on-chain merge_asof 병합

    Example:
        >>> service = OnchainDataService()
        >>> defs = service.get_batch_definitions("stablecoin")
        >>> df = service.load("defillama", "stablecoin_total")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        silver_processor: OnchainSilverProcessor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._silver = silver_processor or OnchainSilverProcessor(self._settings)

    def get_batch_definitions(self, batch_type: str) -> list[tuple[str, str]]:
        """batch_type에 해당하는 (source, name) 리스트 반환.

        Args:
            batch_type: "stablecoin", "tvl", "dex", "coinmetrics", "all"

        Returns:
            (source, name) 튜플 리스트

        Raises:
            ValueError: 알 수 없는 batch_type
        """
        if batch_type == "all":
            result: list[tuple[str, str]] = []
            for defs in ONCHAIN_BATCH_DEFINITIONS.values():
                result.extend(defs)
            return result

        if batch_type not in ONCHAIN_BATCH_DEFINITIONS:
            valid = ", ".join([*ONCHAIN_BATCH_DEFINITIONS.keys(), "all"])
            msg = f"Unknown batch type: {batch_type}. Valid: {valid}"
            raise ValueError(msg)

        return list(ONCHAIN_BATCH_DEFINITIONS[batch_type])

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
        """On-chain 데이터를 OHLCV에 merge_asof로 병합.

        Args:
            ohlcv_df: OHLCV DataFrame (DatetimeIndex)
            source: 데이터 소스
            name: 데이터 이름
            columns: 병합할 컬럼 (None이면 전체)
            lag_days: publication lag 일수 (None이면 SOURCE_LAG_DAYS 기본값)

        Returns:
            병합된 DataFrame
        """
        onchain_df = self._silver.load(source, name)

        if onchain_df.empty:
            logger.warning(f"No on-chain data for {source}/{name}, returning original OHLCV")
            return ohlcv_df

        # date 컬럼을 인덱스로 설정 (merge_asof용)
        date_col = get_date_col(source)
        if date_col in onchain_df.columns:
            onchain_df = onchain_df.set_index(date_col)
        if not isinstance(onchain_df.index, pd.DatetimeIndex):
            onchain_df.index = pd.to_datetime(onchain_df.index, utc=True)

        # 병합할 컬럼 선택
        if columns is not None:
            available = [c for c in columns if c in onchain_df.columns]
        else:
            # source/name/date 등 메타 컬럼 제외
            exclude = {
                "source",
                "name",
                "chain",
                "stablecoin_id",
                "asset",
                "chart_name",
                "classification",
                "metric_name",
            }
            available = [c for c in onchain_df.columns if c not in exclude]

        if not available:
            return ohlcv_df

        onchain_subset = onchain_df[available].copy()
        onchain_subset = onchain_subset.sort_index()

        # Publication lag shift: T일 데이터는 T+lag일 이후에 접근 가능
        lag = lag_days if lag_days is not None else SOURCE_LAG_DAYS.get(source, 0)
        if lag > 0:
            onchain_subset.index = onchain_subset.index + pd.Timedelta(days=lag)

        result = pd.merge_asof(
            ohlcv_df,
            onchain_subset,
            left_index=True,
            right_index=True,
            direction="backward",
        )

        logger.debug(f"Enriched OHLCV with {len(available)} on-chain columns from {source}/{name}")
        return result

    def precompute(
        self,
        symbol: str,
        ohlcv_index: pd.DatetimeIndex | pd.Index,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """EDA 백테스트용 on-chain 데이터 사전 계산.

        Silver 데이터를 로드하여 oc_* prefix로 rename하고,
        ohlcv_index 기준으로 순차 merge_asof합니다.
        Publication lag가 자동 적용됩니다.

        Args:
            symbol: 거래 심볼 (e.g. "BTC/USDT")
            ohlcv_index: OHLCV DatetimeIndex (merge 기준)
            start: 시작일 (미사용, 인터페이스 호환)
            end: 종료일 (미사용, 인터페이스 호환)

        Returns:
            oc_* 컬럼이 포함된 DataFrame (DatetimeIndex)
        """
        from src.eda.onchain_feed import build_precompute_map

        precompute_map = build_precompute_map([symbol])
        sources = precompute_map.get(symbol, [])
        if not sources:
            return pd.DataFrame(index=ohlcv_index)

        result = pd.DataFrame(index=ohlcv_index)
        for source, name, columns, rename_map in sources:
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

        # 비어있으면 빈 DataFrame 반환
        if result.columns.empty:
            return pd.DataFrame(index=ohlcv_index)

        return result

    def _load_and_prepare(
        self,
        source: str,
        name: str,
        columns: list[str],
        rename_map: dict[str, str],
    ) -> pd.DataFrame | None:
        """단일 source/name 데이터를 로드→컬럼 선택→oc_ rename→lag shift.

        Returns:
            준비된 DataFrame 또는 None (데이터 없음)
        """
        try:
            df = self._silver.load(source, name)
        except Exception:
            return None
        if df.empty:
            return None

        # date 컬럼 → DatetimeIndex
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

        # oc_ prefix rename
        col_rename: dict[str, str] = {}
        for col in available:
            if col in rename_map:
                col_rename[col] = rename_map[col]
            else:
                col_rename[col] = f"oc_{col.lower()}"
        subset.columns = pd.Index([col_rename.get(c, c) for c in subset.columns])

        # Publication lag shift
        lag = SOURCE_LAG_DAYS.get(source, 0)
        if lag > 0:
            subset.index = subset.index + pd.Timedelta(days=lag)

        subset = subset.sort_index()
        # float64 coercion
        for col in subset.columns:
            if subset[col].dtype == object:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")

        return subset
