"""OnchainProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 on-chain DataFrame을 merge_asof로 병합
Live: Silver 초기 로드 + API 직접 polling (5+ 소스, 혼합 scope)

Rules Applied:
    - DerivativesProviderPort 패턴 복제
    - OnchainProviderPort 프로토콜 구현
    - LiveDerivativesFeed 참조 패턴 (API polling)
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from src.data.onchain.fetcher import CM_RENAME_MAP

if TYPE_CHECKING:
    from src.data.onchain.client import AsyncOnchainClient
    from src.data.onchain.fetcher import OnchainFetcher


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


# Polling 주기 (초)
_DEFILLAMA_POLL_INTERVAL = 21600  # 6h
_SENTIMENT_POLL_INTERVAL = 21600  # 6h
_COINMETRICS_POLL_INTERVAL = 43200  # 12h
_BTC_MINING_POLL_INTERVAL = 21600  # 6h
_ETH_SUPPLY_POLL_INTERVAL = 43200  # 12h


def _inc_cache_refresh(status: str) -> None:
    """Cache refresh counter 증가 (monitoring 의존 선택적)."""
    try:
        from src.monitoring.metrics import onchain_cache_refresh_total

        onchain_cache_refresh_total.labels(status=status).inc()
    except ImportError:
        pass


class LiveOnchainFeed:
    """Live on-chain 데이터 feed — Silver 초기 로드 + API polling.

    OnchainProviderPort 프로토콜을 구현합니다.
    5+ 소스에서 GLOBAL + PER-ASSET 혼합 polling.

    Startup: Silver 디스크에서 초기 캐시 로드 (cold start 방지)
    Background: DeFiLlama/Alternative.me/CoinMetrics/mempool/Etherscan API polling
    Graceful Degradation: API 실패 시 마지막 캐시 값 유지
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        poll_interval_defillama: int = _DEFILLAMA_POLL_INTERVAL,
        poll_interval_sentiment: int = _SENTIMENT_POLL_INTERVAL,
        poll_interval_coinmetrics: int = _COINMETRICS_POLL_INTERVAL,
        poll_interval_btc_mining: int = _BTC_MINING_POLL_INTERVAL,
        poll_interval_eth_supply: int = _ETH_SUPPLY_POLL_INTERVAL,
        refresh_interval: int = 0,
    ) -> None:
        self._symbols = symbols
        self._poll_interval_defillama = poll_interval_defillama
        self._poll_interval_sentiment = poll_interval_sentiment
        self._poll_interval_coinmetrics = poll_interval_coinmetrics
        self._poll_interval_btc_mining = poll_interval_btc_mining
        self._poll_interval_eth_supply = poll_interval_eth_supply
        self._cache: dict[str, dict[str, float]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()
        self._notification_queue: Any = None  # 후주입 (NotificationQueue)
        self._clients: list[AsyncOnchainClient] = []
        self._fetcher: OnchainFetcher | None = None
        # Legacy compat (ignored)
        _ = refresh_interval

    async def start(self) -> None:
        """Silver 초기 로드 + API clients 생성 + polling tasks 시작."""
        self._shutdown.clear()
        self._load_cache()

        from src.data.onchain.client import AsyncOnchainClient as _Client
        from src.data.onchain.fetcher import OnchainFetcher as _Fetcher

        # 단일 client (범용, rate limit 소스별 관리)
        client = _Client("onchain")
        await client.__aenter__()
        self._clients.append(client)
        self._fetcher = _Fetcher(client)

        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(self._poll_defillama()),
            asyncio.create_task(self._poll_sentiment()),
            asyncio.create_task(self._poll_coinmetrics()),
        ]

        # BTC mining (BTC 심볼 있을 때만)
        if self._symbol_for_asset("BTC") is not None:
            tasks.append(asyncio.create_task(self._poll_btc_mining()))

        # ETH supply (ETH 심볼 + ETHERSCAN_API_KEY 있을 때만)
        etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "")
        if self._symbol_for_asset("ETH") is not None and etherscan_key:
            tasks.append(asyncio.create_task(self._poll_eth_supply()))
        elif self._symbol_for_asset("ETH") is not None:
            logger.warning("ETHERSCAN_API_KEY not set — ETH supply polling disabled")

        self._tasks = tasks
        logger.info(
            "LiveOnchainFeed started for {} symbols ({} cached, {} polling tasks)",
            len(self._symbols),
            len(self._cache),
            len(self._tasks),
        )

    async def stop(self) -> None:
        """Graceful shutdown — tasks cancel + clients close."""
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for client in self._clients:
            await client.__aexit__(None, None, None)
        self._clients.clear()
        self._fetcher = None
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
                    if source in ("fred", "yfinance"):
                        from src.data.macro.storage import MacroSilverProcessor

                        df = MacroSilverProcessor().load(source, name)
                    else:
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

    # === Helper ===

    def _symbol_for_asset(self, asset: str) -> str | None:
        """asset → symbol 매핑 (BTC → BTC/USDT)."""
        for s in self._symbols:
            if s.split("/")[0].upper() == asset.upper():
                return s
        return None

    def _set_global_cache(self, key: str, value: float) -> None:
        """GLOBAL 값을 모든 심볼의 캐시에 설정."""
        for symbol in self._symbols:
            cache = self._cache.setdefault(symbol, {})
            cache[key] = value

    # === Polling tasks ===

    async def _poll_defillama(self) -> None:
        """DeFiLlama stablecoin + TVL + DEX volume polling (6h 주기, GLOBAL)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_defillama()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_defillama,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_defillama(self) -> None:
        """DeFiLlama 단일 사이클."""
        assert self._fetcher is not None

        # Stablecoin Total
        try:
            df = await self._fetcher.fetch_stablecoin_total()
            if not df.empty:
                self._set_global_cache(
                    "oc_stablecoin_total_usd",
                    float(df.iloc[-1]["total_circulating_usd"]),
                )
        except Exception as e:
            logger.warning("DeFiLlama stablecoin polling error: {}", e)

        # TVL Total
        try:
            df = await self._fetcher.fetch_tvl("")
            if not df.empty:
                self._set_global_cache("oc_tvl_usd", float(df.iloc[-1]["tvl_usd"]))
        except Exception as e:
            logger.warning("DeFiLlama TVL polling error: {}", e)

        # DEX Volume
        try:
            df = await self._fetcher.fetch_dex_volume()
            if not df.empty:
                self._set_global_cache("oc_dex_volume_usd", float(df.iloc[-1]["volume_usd"]))
        except Exception as e:
            logger.warning("DeFiLlama DEX volume polling error: {}", e)

        _inc_cache_refresh("success")
        logger.debug("LiveOnchainFeed DeFiLlama poll done")

    async def _poll_sentiment(self) -> None:
        """Fear & Greed Index polling (6h 주기, GLOBAL)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_sentiment()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_sentiment,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_sentiment(self) -> None:
        """Fear & Greed 단일 사이클."""
        assert self._fetcher is not None

        try:
            df = await self._fetcher.fetch_fear_greed()
            if not df.empty:
                self._set_global_cache("oc_fear_greed", float(df.iloc[-1]["value"]))
        except Exception as e:
            logger.warning("Fear & Greed polling error: {}", e)

        logger.debug("LiveOnchainFeed sentiment poll done")

    async def _poll_coinmetrics(self) -> None:
        """Coin Metrics BTC/ETH polling (12h 주기, PER-ASSET)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_coinmetrics()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_coinmetrics,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_coinmetrics(self) -> None:
        """Coin Metrics 단일 사이클."""
        assert self._fetcher is not None
        from datetime import UTC, datetime, timedelta

        start = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
        end = datetime.now(UTC).strftime("%Y-%m-%d")

        for asset_lower in ("btc", "eth"):
            asset_upper = asset_lower.upper()
            sym = self._symbol_for_asset(asset_upper)
            if sym is None:
                continue

            try:
                df = await self._fetcher.fetch_coinmetrics(
                    asset_lower, start=start, end=end
                )
                if not df.empty:
                    last = df.iloc[-1]
                    cache = self._cache.setdefault(sym, {})
                    for col, oc_name in CM_RENAME_MAP.items():
                        if col in last.index and last[col] is not None:
                            try:
                                cache[oc_name] = float(last[col])
                            except (ValueError, TypeError):
                                continue
            except Exception as e:
                logger.warning("CoinMetrics {} polling error: {}", asset_upper, e)

        logger.debug("LiveOnchainFeed CoinMetrics poll done")

    async def _poll_btc_mining(self) -> None:
        """mempool.space BTC mining polling (6h 주기, BTC only)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_btc_mining()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_btc_mining,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_btc_mining(self) -> None:
        """mempool.space 단일 사이클."""
        assert self._fetcher is not None

        sym = self._symbol_for_asset("BTC")
        if sym is None:
            return

        try:
            df = await self._fetcher.fetch_mempool_mining(interval="1m")
            if not df.empty:
                last = df.iloc[-1]
                cache = self._cache.setdefault(sym, {})
                if "avg_hashrate" in last.index:
                    cache["oc_avg_hashrate"] = float(last["avg_hashrate"])
                if "difficulty" in last.index:
                    cache["oc_difficulty"] = float(last["difficulty"])
        except Exception as e:
            logger.warning("mempool.space mining polling error: {}", e)

        logger.debug("LiveOnchainFeed BTC mining poll done")

    async def _poll_eth_supply(self) -> None:
        """Etherscan ETH supply polling (12h 주기, ETH only)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_eth_supply()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_eth_supply,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_eth_supply(self) -> None:
        """Etherscan 단일 사이클."""
        assert self._fetcher is not None

        sym = self._symbol_for_asset("ETH")
        if sym is None:
            return

        api_key = os.environ.get("ETHERSCAN_API_KEY", "")
        if not api_key:
            return

        try:
            df = await self._fetcher.fetch_eth_supply(api_key)
            if not df.empty:
                last = df.iloc[-1]
                cache = self._cache.setdefault(sym, {})
                if "eth_supply" in last.index:
                    cache["oc_eth_supply"] = float(last["eth_supply"])
                if "eth2_staking" in last.index:
                    cache["oc_eth2_staking"] = float(last["eth2_staking"])
        except Exception as e:
            logger.warning("Etherscan ETH supply polling error: {}", e)

        logger.debug("LiveOnchainFeed ETH supply poll done")


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

# Macro 글로벌 데이터 (FRED + yfinance) — fallback용
# deprecated: catalog로 마이그레이션 중 — fallback용으로 유지
MACRO_GLOBAL_SOURCES: list[tuple[str, str, list[str], dict[str, str]]] = [
    ("fred", "dxy", ["value"], {"value": "macro_dxy"}),
    ("fred", "gold", ["value"], {"value": "macro_gold"}),
    ("fred", "dgs10", ["value"], {"value": "macro_dgs10"}),
    ("fred", "dgs2", ["value"], {"value": "macro_dgs2"}),
    ("fred", "t10y2y", ["value"], {"value": "macro_yield_curve"}),
    ("fred", "vix", ["value"], {"value": "macro_vix"}),
    ("fred", "m2", ["value"], {"value": "macro_m2"}),
    ("yfinance", "spy", ["close"], {"close": "macro_spy_close"}),
    ("yfinance", "qqq", ["close"], {"close": "macro_qqq_close"}),
    ("yfinance", "gld", ["close"], {"close": "macro_gld_close"}),
    ("yfinance", "tlt", ["close"], {"close": "macro_tlt_close"}),
    ("yfinance", "uup", ["close"], {"close": "macro_uup_close"}),
    ("yfinance", "hyg", ["close"], {"close": "macro_hyg_close"}),
    # CoinGecko
    ("coingecko", "global_metrics", ["btc_dominance"], {"btc_dominance": "macro_btc_dom"}),
    (
        "coingecko",
        "global_metrics",
        ["total_market_cap_usd"],
        {"total_market_cap_usd": "macro_total_mcap"},
    ),
    ("coingecko", "defi_global", ["defi_dominance"], {"defi_dominance": "macro_defi_dom"}),
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
