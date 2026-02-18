"""MacroProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 macro DataFrame을 merge_asof로 병합
Live: Silver 데이터 캐시 + 주기적 refresh

Rules Applied:
    - OnchainProviderPort 패턴 복제
    - MacroProviderPort 프로토콜 구현
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pandas as pd
from loguru import logger


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


# Daily refresh 주기 (초)
_REFRESH_INTERVAL = 86400


class LiveMacroFeed:
    """Live macro 데이터 feed — Silver 캐시 + 주기적 refresh.

    MacroProviderPort 프로토콜을 구현합니다.
    GLOBAL scope: 모든 symbol에 동일한 값을 제공합니다.
    """

    def __init__(
        self,
        refresh_interval: int = _REFRESH_INTERVAL,
    ) -> None:
        self._refresh_interval = refresh_interval
        self._cache: dict[str, float] = {}
        self._task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._notification_queue: Any = None

    async def start(self) -> None:
        """Silver에서 초기 캐시 로드 + background refresh task 시작."""
        self._shutdown.clear()
        self._load_cache()
        self._task = asyncio.create_task(self._periodic_refresh())
        logger.info(
            "LiveMacroFeed started ({} columns cached)",
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
                    logger.debug("LiveMacroFeed cache refreshed")
                except Exception:
                    logger.exception("MacroFeed cache refresh failed")
