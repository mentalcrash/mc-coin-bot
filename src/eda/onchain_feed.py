"""OnchainProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 on-chain DataFrame을 merge_asof로 병합
Live: Silver 데이터 캐시 + 주기적 refresh

Rules Applied:
    - DerivativesProviderPort 패턴 복제
    - OnchainProviderPort 프로토콜 구현
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pandas as pd
from loguru import logger

from src.data.onchain.fetcher import CM_RENAME_MAP


class BacktestOnchainProvider:
    """Backtest용 OnchainProvider — 사전 계산된 데이터 사용.

    OnchainDataService.precompute()로 생성된 DataFrame을
    merge_asof로 OHLCV에 병합합니다.
    """

    def __init__(self, precomputed: dict[str, pd.DataFrame]) -> None:
        """초기화.

        Args:
            precomputed: {symbol: onchain_df} — 각 df는 DatetimeIndex, oc_* 컬럼
        """
        self._precomputed = precomputed

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Precomputed on-chain 데이터를 merge_asof로 병합."""
        onchain = self._precomputed.get(symbol)
        if onchain is None or onchain.empty:
            return df

        return pd.merge_asof(
            df,
            onchain,
            left_index=True,
            right_index=True,
            direction="backward",
        )

    def get_onchain_columns(self, symbol: str) -> dict[str, float] | None:
        """Backtest 모드에서는 None (precomputed 사용)."""
        return None


# Daily refresh 주기 (초)
_REFRESH_INTERVAL = 86400


def _inc_cache_refresh(status: str) -> None:
    """Cache refresh counter 증가 (monitoring 의존 선택적)."""
    try:
        from src.monitoring.metrics import onchain_cache_refresh_total

        onchain_cache_refresh_total.labels(status=status).inc()
    except ImportError:
        pass


class LiveOnchainFeed:
    """Live on-chain 데이터 feed — Silver 캐시 + 주기적 refresh.

    OnchainProviderPort 프로토콜을 구현합니다.
    Silver 데이터의 최신 행을 캐시하여 전략에 제공합니다.
    """

    def __init__(
        self,
        symbols: list[str],
        refresh_interval: int = _REFRESH_INTERVAL,
    ) -> None:
        self._symbols = symbols
        self._refresh_interval = refresh_interval
        self._cache: dict[str, dict[str, float]] = {}
        self._task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._notification_queue: Any = None  # 후주입 (NotificationQueue)

    async def start(self) -> None:
        """Silver에서 초기 캐시 로드 + background refresh task 시작."""
        self._shutdown.clear()
        self._load_cache()
        self._task = asyncio.create_task(self._periodic_refresh())
        logger.info(
            "LiveOnchainFeed started for {} symbols ({} cached)",
            len(self._symbols),
            len(self._cache),
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._shutdown.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("LiveOnchainFeed stopped")

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Live 모드: 캐시된 값을 전체 행에 broadcast."""
        cols = self._cache.get(symbol)
        if cols is None:
            return df
        for col_name, col_val in cols.items():
            df[col_name] = col_val
        return df

    def get_onchain_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 값 반환."""
        return self._cache.get(symbol)

    def _load_cache(self) -> None:
        """Silver 데이터에서 최신 행을 캐시에 로드."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService()
        precompute_map = build_precompute_map(self._symbols)

        for symbol, sources in precompute_map.items():
            cache: dict[str, float] = {}
            for source, name, columns, rename_map in sources:
                try:
                    df = service.load(source, name)
                except Exception:
                    logger.debug("No Silver data for {}/{}, skipping", source, name)
                    continue
                if df.empty:
                    continue
                # 최신 행에서 값 추출
                last_row = df.iloc[-1]
                for col in columns:
                    if col in last_row.index:
                        oc_name = rename_map.get(col, f"oc_{col.lower()}")
                        val = last_row[col]
                        try:
                            cache[oc_name] = float(val)
                        except (ValueError, TypeError):
                            continue
            if cache:
                self._cache[symbol] = cache

    def get_health_status(self) -> dict[str, int]:
        """Heartbeat용 on-chain 캐시 상태."""
        return {
            "symbols_cached": len(self._cache),
            "total_columns": sum(len(c) for c in self._cache.values()),
        }

    def update_cache_metrics(self) -> None:
        """Prometheus gauge에 캐시 크기 갱신."""
        try:
            from src.monitoring.metrics import onchain_cache_size_gauge

            for symbol, cache in self._cache.items():
                onchain_cache_size_gauge.labels(symbol=symbol).set(len(cache))
        except ImportError:
            pass

    async def _send_alert(self, message: str) -> None:
        """Notification queue에 on-chain 알림 전송."""
        if self._notification_queue is None:
            return
        try:
            from src.notification.health_formatters import format_onchain_alert_embed
            from src.notification.models import ChannelRoute, NotificationItem, Severity

            embed = format_onchain_alert_embed(message, "LiveOnchainFeed")
            item = NotificationItem(
                severity=Severity.WARNING,
                channel=ChannelRoute.ALERTS,
                embed=embed,
                spam_key="onchain_refresh_fail",
            )
            await self._notification_queue.enqueue(item)
        except Exception:
            logger.debug("Failed to send onchain alert notification")

    async def _periodic_refresh(self) -> None:
        """주기적으로 Silver 데이터를 다시 로드."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._refresh_interval,
                )
                break
            except TimeoutError:
                try:
                    self._load_cache()
                    _inc_cache_refresh("success")
                    logger.debug("LiveOnchainFeed cache refreshed")
                except Exception:
                    _inc_cache_refresh("failure")
                    logger.exception("OnchainFeed cache refresh failed")
                    await self._send_alert("Cache refresh failed")


# ---------------------------------------------------------------------------
# Precompute helpers — symbol → source mapping
# ---------------------------------------------------------------------------

# 모든 심볼이 받는 글로벌 데이터: (source, name, columns, rename_map)
# deprecated: catalog로 마이그레이션 중 — fallback용으로 유지
_GLOBAL_SOURCES: list[tuple[str, str, list[str], dict[str, str]]] = [
    (
        "defillama",
        "stablecoin_total",
        ["total_circulating_usd"],
        {"total_circulating_usd": "oc_stablecoin_total_usd"},
    ),
    ("defillama", "tvl_total", ["tvl_usd"], {"tvl_usd": "oc_tvl_usd"}),
    ("defillama", "dex_volume", ["volume_usd"], {"volume_usd": "oc_dex_volume_usd"}),
    ("alternative_me", "fear_greed", ["value"], {"value": "oc_fear_greed"}),
]

# Asset별 추가 데이터
# deprecated: catalog로 마이그레이션 중 — fallback용으로 유지
_ASSET_SOURCES: dict[str, list[tuple[str, str, list[str], dict[str, str]]]] = {
    "BTC": [
        (
            "coinmetrics",
            "btc_metrics",
            ["CapMVRVCur", "CapMrktCurUSD", "FlowInExUSD", "FlowOutExUSD"],
            CM_RENAME_MAP,
        ),
        ("blockchain_com", "bc_hash-rate", ["value"], {"value": "oc_hash_rate"}),
        ("mempool_space", "mining", ["avg_hashrate", "difficulty"], {}),
    ],
    "ETH": [
        (
            "coinmetrics",
            "eth_metrics",
            ["CapMVRVCur", "CapMrktCurUSD", "FlowInExUSD", "FlowOutExUSD"],
            CM_RENAME_MAP,
        ),
        ("etherscan", "eth_supply", ["eth_supply", "eth2_staking"], {}),
    ],
}


def _try_catalog_precompute(
    symbols: list[str],
) -> dict[str, list[tuple[str, str, list[str], dict[str, str]]]] | None:
    """Catalog에서 precompute map 로드 시도, 실패 시 None."""
    try:
        from src.catalog.store import DataCatalogStore

        store = DataCatalogStore()
        return store.build_precompute_map(symbols)
    except Exception:
        return None


def build_precompute_map(
    symbols: list[str],
) -> dict[str, list[tuple[str, str, list[str], dict[str, str]]]]:
    """Symbol 리스트에서 symbol→sources 매핑 생성.

    Catalog 우선, 실패 시 hardcoded constants fallback.
    """
    catalog_result = _try_catalog_precompute(symbols)
    if catalog_result is not None:
        return catalog_result

    result: dict[str, list[tuple[str, str, list[str], dict[str, str]]]] = {}
    for symbol in symbols:
        asset = symbol.split("/")[0].upper()
        sources = list(_GLOBAL_SOURCES)
        if asset in _ASSET_SOURCES:
            sources.extend(_ASSET_SOURCES[asset])
        result[symbol] = sources
    return result
