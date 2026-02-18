"""DerivativesSnapshotFetcher — 최신 파생상품 데이터 1건 조회.

LIVE 모드: BinanceFuturesClient 재사용
Paper/Shadow 모드: 내부 ccxt 인스턴스 (public endpoints only, API key 불필요)

Rules Applied:
    - #10 Python Standards: Async patterns, type hints
    - Exchange Rules: ccxt lifecycle 관리
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import ccxt.pro as ccxt
from loguru import logger

from src.notification.health_models import SymbolDerivativesSnapshot

if TYPE_CHECKING:
    from src.exchange.binance_futures_client import BinanceFuturesClient

# Funding rate -> annualized: 8h funding * 3 * 365
_ANNUALIZE_FACTOR = 3 * 365 * 100  # % 단위


class DerivativesSnapshotFetcher:
    """파생상품 데이터 스냅샷 조회.

    Args:
        futures_client: BinanceFuturesClient (LIVE 모드에서 재사용, None이면 내부 생성)
    """

    def __init__(self, futures_client: BinanceFuturesClient | None = None) -> None:
        self._futures_client = futures_client
        self._own_exchange: ccxt.binance | None = None  # Paper/Shadow용

    async def start(self) -> None:
        """내부 ccxt 인스턴스 생성 (futures_client 없을 때만)."""
        if self._futures_client is not None:
            return

        self._own_exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        await self._own_exchange.load_markets()
        logger.info("DerivativesSnapshotFetcher: internal ccxt client ready (public only)")

    async def stop(self) -> None:
        """내부 ccxt 인스턴스 정리."""
        if self._own_exchange is not None:
            await self._own_exchange.close()
            self._own_exchange = None

    async def fetch_symbol(self, symbol: str) -> SymbolDerivativesSnapshot | None:
        """단일 심볼의 최신 파생상품 스냅샷 조회.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")

        Returns:
            SymbolDerivativesSnapshot 또는 조회 실패 시 None
        """
        try:
            if self._futures_client is not None:
                return await self._fetch_via_futures_client(symbol)
            return await self._fetch_via_own_exchange(symbol)
        except Exception:
            logger.exception("Failed to fetch derivatives snapshot for {}", symbol)
            return None

    async def fetch_all(self, symbols: list[str]) -> list[SymbolDerivativesSnapshot]:
        """여러 심볼의 스냅샷 조회 (순차 실행, rate limit 안전).

        Args:
            symbols: 심볼 리스트

        Returns:
            성공한 스냅샷 리스트
        """
        results: list[SymbolDerivativesSnapshot] = []
        for symbol in symbols:
            snap = await self.fetch_symbol(symbol)
            if snap is not None:
                results.append(snap)
            # rate limit 보호: 심볼 간 0.5초 간격
            if symbol != symbols[-1]:
                await asyncio.sleep(0.5)
        return results

    async def _fetch_via_futures_client(self, symbol: str) -> SymbolDerivativesSnapshot | None:
        """BinanceFuturesClient를 통한 데이터 조회."""
        assert self._futures_client is not None

        # 순차 호출 (rate limit 안전)
        ticker = await self._futures_client.fetch_ticker(symbol)
        price = float(ticker.get("last", 0))

        fr_history = await self._futures_client.fetch_funding_rate_history(symbol, limit=1)
        funding_rate = float(fr_history[0].get("fundingRate", 0)) if fr_history else 0.0

        oi_history = await self._futures_client.fetch_open_interest_history(symbol, limit=1)
        oi_value = float(oi_history[0].get("sumOpenInterestValue", 0)) if oi_history else 0.0

        ls_history = await self._futures_client.fetch_long_short_ratio(symbol, limit=1)
        ls_ratio = float(ls_history[0].get("longShortRatio", 1.0)) if ls_history else 1.0

        taker_history = await self._futures_client.fetch_taker_buy_sell_ratio(symbol, limit=1)
        taker_ratio = float(taker_history[0].get("buySellRatio", 1.0)) if taker_history else 1.0

        top_acct = await self._futures_client.fetch_top_long_short_account_ratio(symbol, limit=1)
        top_acct_ls = float(top_acct[0].get("longShortRatio", 1.0)) if top_acct else 1.0

        top_pos = await self._futures_client.fetch_top_long_short_position_ratio(symbol, limit=1)
        top_pos_ls = float(top_pos[0].get("longShortRatio", 1.0)) if top_pos else 1.0

        return SymbolDerivativesSnapshot(
            symbol=symbol,
            price=price,
            funding_rate=funding_rate,
            funding_rate_annualized=funding_rate * _ANNUALIZE_FACTOR,
            open_interest=oi_value,
            ls_ratio=ls_ratio,
            taker_ratio=taker_ratio,
            top_acct_ls_ratio=top_acct_ls,
            top_pos_ls_ratio=top_pos_ls,
        )

    async def _fetch_via_own_exchange(self, symbol: str) -> SymbolDerivativesSnapshot | None:
        """내부 ccxt를 통한 public endpoint 조회."""
        if self._own_exchange is None:
            logger.warning("DerivativesSnapshotFetcher not started")
            return None

        exchange = self._own_exchange
        bare_symbol = symbol.replace("/", "").replace(":USDT", "")
        futures_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol

        # 1. Ticker
        ticker: dict[str, Any] = await exchange.fetch_ticker(futures_symbol)  # type: ignore[assignment]
        price = float(ticker.get("last", 0))

        # 2. Funding rate
        fr_data: list[dict[str, Any]] = await exchange.fetch_funding_rate_history(  # type: ignore[assignment]
            futures_symbol, limit=1
        )
        funding_rate = float(fr_data[0].get("fundingRate", 0)) if fr_data else 0.0

        # 3. Open Interest history
        oi_data: list[dict[str, Any]] = await exchange.fapipublic_get_futures_data_openinteresthist(  # type: ignore[assignment,attr-defined]
            {"symbol": bare_symbol, "period": "1h", "limit": 1}
        )
        oi_value = float(oi_data[0].get("sumOpenInterestValue", 0)) if oi_data else 0.0

        # 4. Long/Short ratio
        ls_data: list[
            dict[str, Any]
        ] = await exchange.fapipublic_get_futures_data_globallongshortaccountratio(  # type: ignore[assignment,attr-defined]
            {"symbol": bare_symbol, "period": "1h", "limit": 1}
        )
        ls_ratio = float(ls_data[0].get("longShortRatio", 1.0)) if ls_data else 1.0

        # 5. Taker ratio
        taker_data: list[
            dict[str, Any]
        ] = await exchange.fapipublic_get_futures_data_takerlongshortratio(  # type: ignore[assignment,attr-defined]
            {"symbol": bare_symbol, "period": "1h", "limit": 1}
        )
        taker_ratio = float(taker_data[0].get("buySellRatio", 1.0)) if taker_data else 1.0

        # 6. Top Trader Account ratio
        top_acct_data: list[
            dict[str, Any]
        ] = await exchange.fapipublic_get_futures_data_toplongshortaccountratio(  # type: ignore[assignment,attr-defined]
            {"symbol": bare_symbol, "period": "1h", "limit": 1}
        )
        top_acct_ls = float(top_acct_data[0].get("longShortRatio", 1.0)) if top_acct_data else 1.0

        # 7. Top Trader Position ratio
        top_pos_data: list[
            dict[str, Any]
        ] = await exchange.fapipublic_get_futures_data_toplongshortpositionratio(  # type: ignore[assignment,attr-defined]
            {"symbol": bare_symbol, "period": "1h", "limit": 1}
        )
        top_pos_ls = float(top_pos_data[0].get("longShortRatio", 1.0)) if top_pos_data else 1.0

        return SymbolDerivativesSnapshot(
            symbol=symbol,
            price=price,
            funding_rate=funding_rate,
            funding_rate_annualized=funding_rate * _ANNUALIZE_FACTOR,
            open_interest=oi_value,
            ls_ratio=ls_ratio,
            taker_ratio=taker_ratio,
            top_acct_ls_ratio=top_acct_ls,
            top_pos_ls_ratio=top_pos_ls,
        )
