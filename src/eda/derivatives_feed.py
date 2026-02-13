"""LiveDerivativesFeed — 실시간 파생상품 데이터 REST polling.

주기적으로 Binance Futures REST API를 polling하여
funding rate, OI, LS ratio, taker ratio를 캐시합니다.

Backtest용 precomputed 모드도 지원합니다 (DerivativesProviderPort 프로토콜).

Rules Applied:
    - EDA Component Pattern: start/stop 라이프사이클
    - DerivativesProviderPort 프로토콜 구현
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.exchange.binance_futures_client import BinanceFuturesClient

# Polling 주기 (초)
_FR_POLL_INTERVAL = 28800  # 8h
_OI_POLL_INTERVAL = 3600  # 1h
_RATIO_POLL_INTERVAL = 3600  # 1h


class BacktestDerivativesProvider:
    """Backtest용 DerivativesProvider — 사전 계산된 데이터 사용.

    DerivativesDataService.precompute()로 생성된 DataFrame을
    merge_asof로 OHLCV에 병합합니다.
    """

    def __init__(self, precomputed: dict[str, pd.DataFrame]) -> None:
        """초기화.

        Args:
            precomputed: {symbol: derivatives_df} — 각 df는 DatetimeIndex 기반
        """
        self._precomputed = precomputed

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Precomputed derivatives를 merge_asof로 병합."""
        deriv = self._precomputed.get(symbol)
        if deriv is None or deriv.empty:
            return df

        return pd.merge_asof(
            df,
            deriv,
            left_index=True,
            right_index=True,
            direction="backward",
        )

    def get_derivatives_columns(self, symbol: str) -> dict[str, float] | None:
        """Backtest 모드에서는 None (precomputed 사용)."""
        return None


class LiveDerivativesFeed:
    """Live 파생상품 데이터 REST polling feed.

    DerivativesProviderPort 프로토콜을 구현합니다.
    주기적으로 Binance Futures REST API를 polling하여 최신 값을 캐시합니다.
    """

    def __init__(
        self,
        symbols: list[str],
        futures_client: BinanceFuturesClient,
        *,
        poll_interval_fr: int = _FR_POLL_INTERVAL,
        poll_interval_oi: int = _OI_POLL_INTERVAL,
        poll_interval_ratios: int = _RATIO_POLL_INTERVAL,
    ) -> None:
        self._symbols = symbols
        self._client = futures_client
        self._poll_interval_fr = poll_interval_fr
        self._poll_interval_oi = poll_interval_oi
        self._poll_interval_ratios = poll_interval_ratios

        # 캐시: {symbol: {column_name: value}}
        self._cache: dict[str, dict[str, float]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        """Background polling tasks 시작."""
        self._shutdown.clear()
        self._tasks = [
            asyncio.create_task(self._poll_funding_rates()),
            asyncio.create_task(self._poll_open_interest()),
            asyncio.create_task(self._poll_ratios()),
        ]
        logger.info(
            "LiveDerivativesFeed started for {} symbols",
            len(self._symbols),
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("LiveDerivativesFeed stopped")

    # === DerivativesProviderPort 프로토콜 ===

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Live 모드: 캐시된 값을 전체 행에 broadcast."""
        cols = self._cache.get(symbol)
        if cols is None:
            return df
        for col_name, col_val in cols.items():
            df[col_name] = col_val
        return df

    def get_derivatives_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 값 반환."""
        return self._cache.get(symbol)

    # === Internal polling ===

    async def _poll_funding_rates(self) -> None:
        """Funding Rate polling (8h 주기)."""
        while not self._shutdown.is_set():
            for symbol in self._symbols:
                try:
                    raw = await self._client.fetch_funding_rate_history(
                        symbol, limit=1
                    )
                    if raw:
                        item = raw[-1]
                        cache = self._cache.setdefault(symbol, {})
                        cache["funding_rate"] = float(item.get("fundingRate", 0))
                        cache["mark_price"] = float(item.get("markPrice", item.get("price", 0)))
                except Exception as e:
                    logger.warning("FR polling error for {}: {}", symbol, e)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_fr,
                )
                break
            except TimeoutError:
                continue

    async def _poll_open_interest(self) -> None:
        """Open Interest polling (1h 주기)."""
        while not self._shutdown.is_set():
            for symbol in self._symbols:
                try:
                    raw = await self._client.fetch_open_interest_history(
                        symbol, period="1h", limit=1
                    )
                    if raw:
                        item = raw[-1]
                        cache = self._cache.setdefault(symbol, {})
                        cache["open_interest"] = float(item.get("sumOpenInterest", 0))
                        cache["oi_value"] = float(item.get("sumOpenInterestValue", 0))
                except Exception as e:
                    logger.warning("OI polling error for {}: {}", symbol, e)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_oi,
                )
                break
            except TimeoutError:
                continue

    async def _poll_ratios(self) -> None:
        """LS Ratio + Taker Ratio polling (1h 주기)."""
        while not self._shutdown.is_set():
            for symbol in self._symbols:
                try:
                    # LS Ratio
                    ls_raw = await self._client.fetch_long_short_ratio(
                        symbol, period="1h", limit=1
                    )
                    if ls_raw:
                        item = ls_raw[-1]
                        cache = self._cache.setdefault(symbol, {})
                        cache["ls_ratio"] = float(item.get("longShortRatio", 0))
                        cache["long_pct"] = float(item.get("longAccount", 0))
                        cache["short_pct"] = float(item.get("shortAccount", 0))

                    # Taker Ratio
                    tk_raw = await self._client.fetch_taker_buy_sell_ratio(
                        symbol, period="1h", limit=1
                    )
                    if tk_raw:
                        item = tk_raw[-1]
                        cache = self._cache.setdefault(symbol, {})
                        cache["taker_ratio"] = float(item.get("buySellRatio", 0))
                        cache["taker_buy_vol"] = float(item.get("buyVol", 0))
                        cache["taker_sell_vol"] = float(item.get("sellVol", 0))
                except Exception as e:
                    logger.warning("Ratio polling error for {}: {}", symbol, e)
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._poll_interval_ratios,
                )
                break
            except TimeoutError:
                continue
