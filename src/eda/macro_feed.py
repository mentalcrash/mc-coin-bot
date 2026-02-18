"""MacroProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 macro DataFrame을 merge_asof로 병합
Live: Silver 초기 로드 + FRED/yfinance/CoinGecko API 직접 polling

Rules Applied:
    - OnchainProviderPort 패턴 복제
    - MacroProviderPort 프로토콜 구현
    - LiveDerivativesFeed 참조 패턴 (API polling)
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.data.macro.client import AsyncCoinGeckoClient, AsyncMacroClient
    from src.data.macro.fetcher import MacroFetcher


class BacktestMacroProvider:
    """Backtest용 MacroProvider — 사전 계산된 GLOBAL 데이터 사용.

    MacroDataService.precompute()로 생성된 단일 DataFrame을
    merge_asof로 OHLCV에 병합합니다 (symbol 무관).
    """

    def __init__(self, precomputed: pd.DataFrame) -> None:
        """초기화.

        Args:
            precomputed: GLOBAL macro DataFrame (DatetimeIndex, macro_* 컬럼)
        """
        self._precomputed = precomputed

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Precomputed macro 데이터를 merge_asof로 병합 (symbol 무관, GLOBAL)."""
        if self._precomputed.empty:
            return df

        return pd.merge_asof(
            df,
            self._precomputed,
            left_index=True,
            right_index=True,
            direction="backward",
        )

    def get_macro_columns(self, symbol: str) -> dict[str, float] | None:
        """Backtest 모드에서는 None (precomputed 사용)."""
        return None


# Polling 주기 (초)
_FRED_POLL_INTERVAL = 21600  # 6h
_YFINANCE_POLL_INTERVAL = 21600  # 6h
_COINGECKO_POLL_INTERVAL = 900  # 15min

# FRED series → cache key 매핑
_FRED_CACHE_KEYS: dict[str, str] = {
    "dxy": "macro_dxy",
    "dgs10": "macro_dgs10",
    "dgs2": "macro_dgs2",
    "t10y2y": "macro_yield_curve",
    "vix": "macro_vix",
    "m2": "macro_m2",
}

# yfinance ticker → cache key 매핑
_YF_CACHE_KEYS: dict[str, str] = {
    "spy": "macro_spy_close",
    "qqq": "macro_qqq_close",
    "gld": "macro_gld_close",
    "tlt": "macro_tlt_close",
    "uup": "macro_uup_close",
    "hyg": "macro_hyg_close",
}


class LiveMacroFeed:
    """Live macro 데이터 feed — Silver 초기 로드 + API polling.

    MacroProviderPort 프로토콜을 구현합니다.
    GLOBAL scope: 모든 symbol에 동일한 값을 제공합니다.

    Startup: Silver 디스크에서 초기 캐시 로드 (cold start 방지)
    Background: FRED/yfinance/CoinGecko API 직접 polling
    Graceful Degradation: API 실패 시 마지막 캐시 값 유지
    """

    def __init__(
        self,
        *,
        poll_interval_fred: int = _FRED_POLL_INTERVAL,
        poll_interval_yfinance: int = _YFINANCE_POLL_INTERVAL,
        poll_interval_coingecko: int = _COINGECKO_POLL_INTERVAL,
        refresh_interval: int = 0,
    ) -> None:
        self._poll_interval_fred = poll_interval_fred
        self._poll_interval_yfinance = poll_interval_yfinance
        self._poll_interval_coingecko = poll_interval_coingecko
        self._cache: dict[str, float] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()
        self._notification_queue: Any = None
        self._fred_client: AsyncMacroClient | None = None
        self._cg_client: AsyncCoinGeckoClient | None = None
        self._fetcher: MacroFetcher | None = None
        # Legacy compat (ignored)
        _ = refresh_interval

    async def start(self) -> None:
        """Silver 초기 로드 + API clients 생성 + polling tasks 시작."""
        self._shutdown.clear()
        self._load_cache()

        from src.data.macro.client import (
            AsyncCoinGeckoClient as _CGClient,
            AsyncMacroClient as _FredClient,
        )
        from src.data.macro.fetcher import MacroFetcher as _Fetcher

        fred_api_key = os.environ.get("FRED_API_KEY", "")

        # FRED client (API 키 필요)
        if fred_api_key:
            self._fred_client = _FredClient("fred")
            await self._fred_client.__aenter__()
        else:
            logger.warning("FRED_API_KEY not set — FRED polling disabled")

        # CoinGecko client (인증 불필요)
        cg_api_key = os.environ.get("COINGECKO_API_KEY", "")
        self._cg_client = _CGClient(api_key=cg_api_key)
        await self._cg_client.__aenter__()

        # Fetcher 생성 (FRED client가 없으면 더미로 생성 — FRED poll은 건너뜀)
        if self._fred_client is not None:
            self._fetcher = _Fetcher(
                self._fred_client,
                api_key=fred_api_key,
                coingecko_client=self._cg_client,
            )
        else:
            # fred_client 없어도 fetcher 자체는 yfinance/coingecko용으로 필요
            # → dummy fred client로 생성 (FRED poll은 조건문으로 건너뜀)
            dummy_fred = _FredClient("fred")
            await dummy_fred.__aenter__()
            self._fred_client = dummy_fred
            self._fetcher = _Fetcher(
                dummy_fred,
                api_key="",
                coingecko_client=self._cg_client,
            )

        tasks: list[asyncio.Task[None]] = []
        if fred_api_key:
            tasks.append(asyncio.create_task(self._poll_fred()))
        tasks.append(asyncio.create_task(self._poll_yfinance()))
        tasks.append(asyncio.create_task(self._poll_coingecko()))
        self._tasks = tasks

        logger.info(
            "LiveMacroFeed started ({} columns cached, {} polling tasks)",
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

        if self._cg_client is not None:
            await self._cg_client.__aexit__(None, None, None)
            self._cg_client = None
        if self._fred_client is not None:
            await self._fred_client.__aexit__(None, None, None)
            self._fred_client = None
        self._fetcher = None
        logger.info("LiveMacroFeed stopped")

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Live 모드: 캐시된 GLOBAL 값을 전체 행에 broadcast."""
        if not self._cache:
            return df
        for col_name, col_val in self._cache.items():
            df[col_name] = col_val
        return df

    def get_macro_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 GLOBAL 값 반환."""
        return self._cache if self._cache else None

    def _load_cache(self) -> None:
        """Silver 데이터에서 최신 행을 캐시에 로드."""
        from src.data.macro.storage import MacroSilverProcessor
        from src.eda.onchain_feed import MACRO_GLOBAL_SOURCES

        silver = MacroSilverProcessor()
        cache: dict[str, float] = {}

        for source, name, columns, rename_map in MACRO_GLOBAL_SOURCES:
            try:
                df = silver.load(source, name)
            except Exception:
                logger.debug("No macro Silver data for {}/{}, skipping", source, name)
                continue
            if df.empty:
                continue

            last_row = df.iloc[-1]
            for col in columns:
                if col in last_row.index:
                    macro_name = rename_map.get(col, f"macro_{col.lower()}")
                    val = last_row[col]
                    try:
                        cache[macro_name] = float(val)
                    except (ValueError, TypeError):
                        continue

        self._cache = cache

    # === Polling tasks ===

    async def _poll_fred(self) -> None:
        """FRED 7 시리즈 polling (6h 주기)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_fred()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_fred,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_fred(self) -> None:
        """FRED 단일 사이클."""
        assert self._fetcher is not None
        from datetime import UTC, datetime, timedelta

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        start = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")

        for name, cache_key in _FRED_CACHE_KEYS.items():
            try:
                df = await self._fetcher.fetch_fred_series(name, start=start, end=today)
                if not df.empty:
                    # FRED "." → None은 fetch_fred_series에서 처리됨
                    last = df.iloc[-1]
                    if last["value"] is not None:
                        self._cache[cache_key] = float(last["value"])
            except Exception as e:
                logger.warning("FRED {} polling error: {}", name, e)

        logger.debug("LiveMacroFeed FRED poll done")

    async def _poll_yfinance(self) -> None:
        """yfinance 6 ETF polling (6h 주기, asyncio.to_thread)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_yfinance()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_yfinance,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_yfinance(self) -> None:
        """yfinance 단일 사이클."""
        assert self._fetcher is not None
        from datetime import UTC, datetime, timedelta

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        start = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")

        for name, cache_key in _YF_CACHE_KEYS.items():
            try:
                df = await self._fetcher.fetch_yfinance_ticker(name, start=start, end=today)
                if not df.empty:
                    self._cache[cache_key] = float(df.iloc[-1]["close"])
            except Exception as e:
                logger.warning("yfinance {} polling error: {}", name, e)

        logger.debug("LiveMacroFeed yfinance poll done")

    async def _poll_coingecko(self) -> None:
        """CoinGecko global + defi polling (15min 주기)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_coingecko()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_coingecko,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_coingecko(self) -> None:
        """CoinGecko 단일 사이클."""
        assert self._fetcher is not None

        # Global metrics
        try:
            df = await self._fetcher.fetch_coingecko_global()
            if not df.empty:
                last = df.iloc[-1]
                self._cache["macro_btc_dom"] = float(last["btc_dominance"])
                self._cache["macro_total_mcap"] = float(last["total_market_cap_usd"])
        except Exception as e:
            logger.warning("CoinGecko global polling error: {}", e)

        # DeFi metrics
        try:
            df = await self._fetcher.fetch_coingecko_defi()
            if not df.empty:
                last = df.iloc[-1]
                self._cache["macro_defi_dom"] = float(last["defi_dominance"])
        except Exception as e:
            logger.warning("CoinGecko defi polling error: {}", e)

        logger.debug("LiveMacroFeed CoinGecko poll done")
