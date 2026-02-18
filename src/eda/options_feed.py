"""OptionsProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 options DataFrame을 merge_asof로 병합
Live: Silver 초기 로드 + Deribit API 직접 polling

Rules Applied:
    - OnchainProviderPort 패턴 복제
    - OptionsProviderPort 프로토콜 구현
    - LiveDerivativesFeed 참조 패턴 (API polling)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.data.options.client import AsyncOptionsClient
    from src.data.options.fetcher import OptionsFetcher

# Precompute definitions: (source, name, columns, rename_map)
OPTIONS_PRECOMPUTE_DEFS: list[tuple[str, str, list[str], dict[str, str]]] = [
    ("deribit", "btc_dvol", ["close"], {"close": "opt_btc_dvol"}),
    ("deribit", "eth_dvol", ["close"], {"close": "opt_eth_dvol"}),
    ("deribit", "btc_pc_ratio", ["pc_ratio"], {"pc_ratio": "opt_btc_pc_ratio"}),
    ("deribit", "btc_hist_vol", ["vol_30d"], {"vol_30d": "opt_btc_rv30d"}),
    ("deribit", "btc_term_structure", ["slope"], {"slope": "opt_btc_term_slope"}),
]


class BacktestOptionsProvider:
    """Backtest용 OptionsProvider — 사전 계산된 GLOBAL 데이터 사용.

    OptionsDataService.precompute()로 생성된 단일 DataFrame을
    merge_asof로 OHLCV에 병합합니다 (symbol 무관).
    """

    def __init__(self, precomputed: pd.DataFrame) -> None:
        """초기화.

        Args:
            precomputed: GLOBAL options DataFrame (DatetimeIndex, opt_* 컬럼)
        """
        self._precomputed = precomputed

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Precomputed options 데이터를 merge_asof로 병합 (symbol 무관, GLOBAL)."""
        if self._precomputed.empty:
            return df

        return pd.merge_asof(
            df,
            self._precomputed,
            left_index=True,
            right_index=True,
            direction="backward",
        )

    def get_options_columns(self, symbol: str) -> dict[str, float] | None:
        """Backtest 모드에서는 None (precomputed 사용)."""
        return None


# Polling 주기 (초)
_DVOL_PCR_POLL_INTERVAL = 900  # 15min
_VOL_TERM_POLL_INTERVAL = 3600  # 1h


class LiveOptionsFeed:
    """Live options 데이터 feed — Silver 초기 로드 + Deribit API polling.

    OptionsProviderPort 프로토콜을 구현합니다.
    GLOBAL scope: 모든 symbol에 동일한 값을 제공합니다.

    Startup: Silver 디스크에서 초기 캐시 로드 (cold start 방지)
    Background: Deribit Public API 직접 polling
    Graceful Degradation: API 실패 시 마지막 캐시 값 유지
    """

    def __init__(
        self,
        *,
        poll_interval_dvol_pcr: int = _DVOL_PCR_POLL_INTERVAL,
        poll_interval_vol_term: int = _VOL_TERM_POLL_INTERVAL,
        refresh_interval: int = 0,
    ) -> None:
        self._poll_interval_dvol_pcr = poll_interval_dvol_pcr
        self._poll_interval_vol_term = poll_interval_vol_term
        self._cache: dict[str, float] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()
        self._notification_queue: Any = None
        self._client: AsyncOptionsClient | None = None
        self._fetcher: OptionsFetcher | None = None
        # Legacy compat (ignored)
        _ = refresh_interval

    async def start(self) -> None:
        """Silver 초기 로드 + Deribit client 생성 + polling tasks 시작."""
        self._shutdown.clear()
        self._load_cache()

        # Create Deribit client + fetcher
        from src.data.options.client import AsyncOptionsClient as _Client
        from src.data.options.fetcher import OptionsFetcher as _Fetcher

        self._client = _Client("deribit")
        await self._client.__aenter__()
        self._fetcher = _Fetcher(self._client)

        self._tasks = [
            asyncio.create_task(self._poll_dvol_pcr()),
            asyncio.create_task(self._poll_vol_term()),
        ]
        logger.info(
            "LiveOptionsFeed started ({} columns cached, 2 polling tasks)",
            len(self._cache),
        )

    async def stop(self) -> None:
        """Graceful shutdown — tasks cancel + client close."""
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
        self._fetcher = None
        logger.info("LiveOptionsFeed stopped")

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Live 모드: 캐시된 GLOBAL 값을 전체 행에 broadcast."""
        if not self._cache:
            return df
        for col_name, col_val in self._cache.items():
            df[col_name] = col_val
        return df

    def get_options_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 GLOBAL 값 반환."""
        return self._cache if self._cache else None

    def _load_cache(self) -> None:
        """Silver 데이터에서 최신 행을 캐시에 로드."""
        from src.data.options.storage import OptionsSilverProcessor

        silver = OptionsSilverProcessor()
        cache: dict[str, float] = {}

        for source, name, columns, rename_map in OPTIONS_PRECOMPUTE_DEFS:
            try:
                df = silver.load(source, name)
            except Exception:
                logger.debug("No options Silver data for {}/{}, skipping", source, name)
                continue
            if df.empty:
                continue

            last_row = df.iloc[-1]
            for col in columns:
                if col in last_row.index:
                    opt_name = rename_map.get(col, f"opt_{col.lower()}")
                    val = last_row[col]
                    try:
                        cache[opt_name] = float(val)
                    except (ValueError, TypeError):
                        continue

        self._cache = cache

    # === Polling tasks ===

    async def _poll_dvol_pcr(self) -> None:
        """DVOL (BTC/ETH) + Put/Call Ratio polling (15min 주기)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_dvol_pcr()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_dvol_pcr,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_dvol_pcr(self) -> None:
        """DVOL BTC/ETH + PC Ratio 단일 사이클."""
        assert self._fetcher is not None
        from datetime import UTC, datetime, timedelta

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        start = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")

        # BTC DVOL
        try:
            df = await self._fetcher.fetch_dvol("BTC", start=start, end=today)
            if not df.empty:
                self._cache["opt_btc_dvol"] = float(df.iloc[-1]["close"])
        except Exception as e:
            logger.warning("Options DVOL BTC polling error: {}", e)

        # ETH DVOL
        try:
            df = await self._fetcher.fetch_dvol("ETH", start=start, end=today)
            if not df.empty:
                self._cache["opt_eth_dvol"] = float(df.iloc[-1]["close"])
        except Exception as e:
            logger.warning("Options DVOL ETH polling error: {}", e)

        # PC Ratio
        try:
            df = await self._fetcher.fetch_pc_ratio("BTC")
            if not df.empty:
                self._cache["opt_btc_pc_ratio"] = float(df.iloc[-1]["pc_ratio"])
        except Exception as e:
            logger.warning("Options PC Ratio polling error: {}", e)

        logger.debug("LiveOptionsFeed DVOL/PCR poll done")

    async def _poll_vol_term(self) -> None:
        """Historical Vol + Term Structure polling (1h 주기)."""
        while not self._shutdown.is_set():
            if self._fetcher is not None:
                await self._fetch_vol_term()
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_vol_term,
                )
                break
            except TimeoutError:
                continue

    async def _fetch_vol_term(self) -> None:
        """Hist Vol + Term Structure 단일 사이클."""
        assert self._fetcher is not None

        # Historical Volatility
        try:
            df = await self._fetcher.fetch_hist_vol("BTC")
            if not df.empty:
                last = df.iloc[-1]
                if "vol_30d" in last.index:
                    self._cache["opt_btc_rv30d"] = float(last["vol_30d"])
        except Exception as e:
            logger.warning("Options Hist Vol polling error: {}", e)

        # Term Structure
        try:
            df = await self._fetcher.fetch_term_structure("BTC")
            if not df.empty:
                last = df.iloc[-1]
                if "slope" in last.index:
                    self._cache["opt_btc_term_slope"] = float(last["slope"])
        except Exception as e:
            logger.warning("Options Term Structure polling error: {}", e)

        logger.debug("LiveOptionsFeed Vol/Term poll done")
