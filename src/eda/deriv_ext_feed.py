"""DerivExtProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 deriv_ext DataFrame을 merge_asof로 병합 (PER-ASSET)
Live: Silver 초기 로드 + Coinalyze/Hyperliquid API 직접 polling

Rules Applied:
    - OnchainProviderPort/DerivativesProviderPort 패턴 복제
    - DerivExtProviderPort 프로토콜 구현
    - LiveDerivativesFeed 참조 패턴 (API polling)
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.data.deriv_ext.client import AsyncCoinalyzeClient, AsyncHyperliquidClient
    from src.data.deriv_ext.fetcher import CoinalyzeFetcher, HyperliquidFetcher


class BacktestDerivExtProvider:
    """Backtest용 DerivExtProvider — 사전 계산된 PER-ASSET 데이터 사용.

    DerivExtDataService.precompute()로 생성된 dict[symbol, DataFrame]을
    merge_asof로 OHLCV에 병합합니다.
    """

    def __init__(self, precomputed: dict[str, pd.DataFrame]) -> None:
        """초기화.

        Args:
            precomputed: {symbol: deriv_ext_df} — 각 df는 DatetimeIndex, dext_* 컬럼
        """
        self._precomputed = precomputed

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Precomputed deriv_ext 데이터를 merge_asof로 병합."""
        dext = self._precomputed.get(symbol)
        if dext is None or dext.empty:
            return df

        return pd.merge_asof(
            df,
            dext,
            left_index=True,
            right_index=True,
            direction="backward",
        )

    def get_deriv_ext_columns(self, symbol: str) -> dict[str, float] | None:
        """Backtest 모드에서는 None (precomputed 사용)."""
        return None


# Polling 주기 (초)
_COINALYZE_POLL_INTERVAL = 3600  # 1h
_HYPERLIQUID_POLL_INTERVAL = 300  # 5min

# Coinalyze: asset → (cache_key_prefix, fetcher methods)
_CA_FETCH_DEFS: list[tuple[str, str, str]] = [
    # (asset, method_suffix, cache_key)
    ("BTC", "agg_oi", "dext_agg_oi_close"),
    ("ETH", "agg_oi", "dext_agg_oi_close"),
    ("BTC", "agg_funding", "dext_agg_funding_close"),
    ("ETH", "agg_funding", "dext_agg_funding_close"),
    ("BTC", "liquidations", ""),
    ("ETH", "liquidations", ""),
    ("BTC", "cvd", "dext_cvd_buy_vol"),
    ("ETH", "cvd", "dext_cvd_buy_vol"),
]


class LiveDerivExtFeed:
    """Live deriv_ext 데이터 feed — Silver 초기 로드 + API polling.

    DerivExtProviderPort 프로토콜을 구현합니다.
    PER-ASSET scope: 각 symbol별로 독립된 캐시를 유지합니다.

    Startup: Silver 디스크에서 초기 캐시 로드 (cold start 방지)
    Background: Coinalyze/Hyperliquid API 직접 polling
    Graceful Degradation: API 실패 시 마지막 캐시 값 유지
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        poll_interval_coinalyze: int = _COINALYZE_POLL_INTERVAL,
        poll_interval_hyperliquid: int = _HYPERLIQUID_POLL_INTERVAL,
        refresh_interval: int = 0,
    ) -> None:
        self._symbols = symbols
        self._poll_interval_coinalyze = poll_interval_coinalyze
        self._poll_interval_hyperliquid = poll_interval_hyperliquid
        self._cache: dict[str, dict[str, float]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()
        self._notification_queue: Any = None
        self._ca_client: AsyncCoinalyzeClient | None = None
        self._hl_client: AsyncHyperliquidClient | None = None
        self._ca_fetcher: CoinalyzeFetcher | None = None
        self._hl_fetcher: HyperliquidFetcher | None = None
        # Legacy compat (ignored)
        _ = refresh_interval

    async def start(self) -> None:
        """Silver 초기 로드 + API clients 생성 + polling tasks 시작."""
        self._shutdown.clear()
        self._load_cache()

        from src.data.deriv_ext.client import (
            AsyncCoinalyzeClient as _CAClient,
            AsyncHyperliquidClient as _HLClient,
        )
        from src.data.deriv_ext.fetcher import (
            CoinalyzeFetcher as _CAFetcher,
            HyperliquidFetcher as _HLFetcher,
        )

        tasks: list[asyncio.Task[None]] = []

        # Coinalyze (API 키 필요)
        ca_api_key = os.environ.get("COINALYZE_API_KEY", "")
        if ca_api_key:
            self._ca_client = _CAClient("coinalyze", api_key=ca_api_key)
            await self._ca_client.__aenter__()
            self._ca_fetcher = _CAFetcher(self._ca_client)
            tasks.append(asyncio.create_task(self._poll_coinalyze()))
        else:
            logger.warning("COINALYZE_API_KEY not set — Coinalyze polling disabled")

        # Hyperliquid (인증 불필요)
        self._hl_client = _HLClient()
        await self._hl_client.__aenter__()
        self._hl_fetcher = _HLFetcher(self._hl_client)
        tasks.append(asyncio.create_task(self._poll_hyperliquid()))

        self._tasks = tasks
        logger.info(
            "LiveDerivExtFeed started for {} symbols ({} cached, {} polling tasks)",
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

        if self._hl_client is not None:
            await self._hl_client.__aexit__(None, None, None)
            self._hl_client = None
        if self._ca_client is not None:
            await self._ca_client.__aexit__(None, None, None)
            self._ca_client = None
        self._ca_fetcher = None
        self._hl_fetcher = None
        logger.info("LiveDerivExtFeed stopped")

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Live 모드: 캐시된 값을 전체 행에 broadcast."""
        cols = self._cache.get(symbol)
        if cols is None:
            return df
        for col_name, col_val in cols.items():
            df[col_name] = col_val
        return df

    def get_deriv_ext_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 값 반환."""
        return self._cache.get(symbol)

    def _load_cache(self) -> None:
        """Silver 데이터에서 최신 행을 캐시에 로드."""
        from src.data.deriv_ext.service import ASSET_PRECOMPUTE_DEFS
        from src.data.deriv_ext.storage import DerivExtSilverProcessor

        silver = DerivExtSilverProcessor()

        for symbol in self._symbols:
            asset = symbol.split("/")[0].upper()
            defs = ASSET_PRECOMPUTE_DEFS.get(asset)
            if defs is None:
                continue

            cache: dict[str, float] = {}
            for source, name, columns, rename_map in defs:
                try:
                    df = silver.load(source, name)
                except Exception:
                    logger.debug("No deriv_ext Silver data for {}/{}, skipping", source, name)
                    continue
                if df.empty:
                    continue

                # Multi-row snapshot (e.g., Hyperliquid): filter by coin
                if "coin" in df.columns:
                    df = df.loc[df["coin"] == asset]
                    if df.empty:
                        continue

                last_row = df.iloc[-1]
                for col in columns:
                    if col in last_row.index:
                        dext_name = rename_map.get(col, f"dext_{col.lower()}")
                        val = last_row[col]
                        try:
                            cache[dext_name] = float(val)
                        except (ValueError, TypeError):
                            continue

            if cache:
                self._cache[symbol] = cache

    # === Polling tasks ===

    def _symbol_for_asset(self, asset: str) -> str | None:
        """asset → symbol 매핑 (BTC → BTC/USDT)."""
        for s in self._symbols:
            if s.split("/")[0].upper() == asset.upper():
                return s
        return None

    async def _poll_coinalyze(self) -> None:
        """Coinalyze OI/Funding/Liq/CVD polling (1h 주기)."""
        while not self._shutdown.is_set():
            if self._ca_fetcher is not None:
                await self._fetch_coinalyze()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_coinalyze,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_coinalyze(self) -> None:
        """Coinalyze 단일 사이클 — BTC/ETH 4종 x 2."""
        assert self._ca_fetcher is not None
        from datetime import UTC, datetime, timedelta

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        start = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")

        for asset in ("BTC", "ETH"):
            sym = self._symbol_for_asset(asset)
            if sym is None:
                continue
            cache = self._cache.setdefault(sym, {})

            # Agg OI
            try:
                df = await self._ca_fetcher.fetch_agg_oi(asset, start=start, end=today)
                if not df.empty:
                    cache["dext_agg_oi_close"] = float(df.iloc[-1]["close"])
            except Exception as e:
                logger.warning("Coinalyze agg_oi {} error: {}", asset, e)

            # Agg Funding
            try:
                df = await self._ca_fetcher.fetch_agg_funding(asset, start=start, end=today)
                if not df.empty:
                    cache["dext_agg_funding_close"] = float(df.iloc[-1]["close"])
            except Exception as e:
                logger.warning("Coinalyze agg_funding {} error: {}", asset, e)

            # Liquidations
            try:
                df = await self._ca_fetcher.fetch_liquidations(asset, start=start, end=today)
                if not df.empty:
                    last = df.iloc[-1]
                    cache["dext_liq_long_vol"] = float(last["long_volume"])
                    cache["dext_liq_short_vol"] = float(last["short_volume"])
            except Exception as e:
                logger.warning("Coinalyze liquidations {} error: {}", asset, e)

            # CVD
            try:
                df = await self._ca_fetcher.fetch_cvd(asset, start=start, end=today)
                if not df.empty:
                    cache["dext_cvd_buy_vol"] = float(df.iloc[-1]["buy_volume"])
            except Exception as e:
                logger.warning("Coinalyze cvd {} error: {}", asset, e)

        logger.debug("LiveDerivExtFeed Coinalyze poll done")

    async def _poll_hyperliquid(self) -> None:
        """Hyperliquid asset_contexts + predicted_fundings polling (5min 주기)."""
        while not self._shutdown.is_set():
            if self._hl_fetcher is not None:
                await self._fetch_hyperliquid()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_hyperliquid,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_hyperliquid(self) -> None:
        """Hyperliquid 단일 사이클 — coin별 필터 → per-symbol cache."""
        assert self._hl_fetcher is not None

        # Asset Contexts
        try:
            df = await self._hl_fetcher.fetch_asset_contexts()
            if not df.empty:
                for _, row in df.iterrows():
                    coin = str(row["coin"])
                    sym = self._symbol_for_asset(coin)
                    if sym is None:
                        continue
                    cache = self._cache.setdefault(sym, {})
                    cache["dext_hl_oi"] = float(row["open_interest"])
                    cache["dext_hl_funding"] = float(row["funding"])
        except Exception as e:
            logger.warning("Hyperliquid asset_contexts polling error: {}", e)

        logger.debug("LiveDerivExtFeed Hyperliquid poll done")
