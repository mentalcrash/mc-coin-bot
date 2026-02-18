"""OptionsProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 options DataFrame을 merge_asof로 병합
Live: Silver 데이터 캐시 + 주기적 refresh

Rules Applied:
    - OnchainProviderPort 패턴 복제
    - OptionsProviderPort 프로토콜 구현
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pandas as pd
from loguru import logger

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


# Daily refresh 주기 (초)
_REFRESH_INTERVAL = 86400


class LiveOptionsFeed:
    """Live options 데이터 feed — Silver 캐시 + 주기적 refresh.

    OptionsProviderPort 프로토콜을 구현합니다.
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
            "LiveOptionsFeed started ({} columns cached)",
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
                    logger.debug("LiveOptionsFeed cache refreshed")
                except Exception:
                    logger.exception("OptionsFeed cache refresh failed")
