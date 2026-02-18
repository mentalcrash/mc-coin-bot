"""DerivExtProviderPort 구현 — Backtest / Live.

Backtest: 사전 계산된 deriv_ext DataFrame을 merge_asof로 병합 (PER-ASSET)
Live: Silver 데이터 캐시 + 주기적 refresh

Rules Applied:
    - OnchainProviderPort/DerivativesProviderPort 패턴 복제
    - DerivExtProviderPort 프로토콜 구현
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pandas as pd
from loguru import logger


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


# Daily refresh 주기 (초)
_REFRESH_INTERVAL = 86400


class LiveDerivExtFeed:
    """Live deriv_ext 데이터 feed — Silver 캐시 + 주기적 refresh.

    DerivExtProviderPort 프로토콜을 구현합니다.
    PER-ASSET scope: 각 symbol별로 독립된 캐시를 유지합니다.
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
        self._notification_queue: Any = None

    async def start(self) -> None:
        """Silver에서 초기 캐시 로드 + background refresh task 시작."""
        self._shutdown.clear()
        self._load_cache()
        self._task = asyncio.create_task(self._periodic_refresh())
        logger.info(
            "LiveDerivExtFeed started for {} symbols ({} cached)",
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
                    logger.debug("LiveDerivExtFeed cache refreshed")
                except Exception:
                    logger.exception("DerivExtFeed cache refresh failed")
