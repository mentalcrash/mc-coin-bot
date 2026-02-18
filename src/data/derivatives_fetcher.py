"""Derivatives data fetcher — Funding Rate, OI, LS Ratio, Taker Ratio.

BinanceFuturesClient를 사용하여 파생상품 데이터를 비동기 배치로 수집합니다.
기존 DataFetcher 패턴 미러링 (페이지네이션 + rate limiting + progress).

Rules Applied:
    - #14 CCXT Trading: Async, retry (BinanceFuturesClient._retry_with_backoff)
    - #15 Logging: Structured logging with context
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from ccxt.base.errors import BadRequest as CcxtBadRequest, ExchangeError as CcxtExchangeError
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.models.derivatives import (
    DerivativesBatch,
    FundingRateRecord,
    LongShortRatioRecord,
    OpenInterestRecord,
    TakerRatioRecord,
    TopTraderAccountRatioRecord,
    TopTraderPositionRatioRecord,
)

if TYPE_CHECKING:
    from src.exchange.binance_futures_client import BinanceFuturesClient

# 8시간 = 28,800,000 ms
_MS_PER_8H = 8 * 60 * 60 * 1000
# 1시간 = 3,600,000 ms
_MS_PER_1H = 60 * 60 * 1000
# Rate limit: 기본 대기 시간 (초)
_DEFAULT_SLEEP = 0.5
# Rate limit: N 요청마다 긴 대기
_LONG_SLEEP_EVERY = 50
_LONG_SLEEP_SECONDS = 2.0


def _get_year_timestamps(year: int) -> tuple[int, int]:
    """해당 연도의 시작/끝 타임스탬프 (Unix ms)."""
    start_dt = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC)
    end_dt = datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


class DerivativesFetcher:
    """파생상품 데이터 비동기 수집기.

    BinanceFuturesClient를 재사용하며, 페이지네이션으로
    대량 데이터를 수집합니다.

    Example:
        >>> async with BinanceFuturesClient() as client:
        ...     fetcher = DerivativesFetcher(client=client)
        ...     batch = await fetcher.fetch_year("BTC/USDT", 2024)
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        settings: IngestionSettings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()
        self._request_count = 0

    async def _rate_limit_sleep(self) -> None:
        """Rate limiting: 요청 간 대기."""
        self._request_count += 1
        if self._request_count % _LONG_SLEEP_EVERY == 0:
            await asyncio.sleep(_LONG_SLEEP_SECONDS)
        else:
            await asyncio.sleep(_DEFAULT_SLEEP)

    async def fetch_funding_rates(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> list[FundingRateRecord]:
        """Funding Rate 히스토리 수집 (페이지네이션).

        CCXT native, limit=500 (8h*500 = ~166일/요청).

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            start_ts: 시작 시각 (Unix ms)
            end_ts: 종료 시각 (Unix ms)

        Returns:
            FundingRateRecord 리스트
        """
        records: list[FundingRateRecord] = []
        since = start_ts

        while since < end_ts:
            raw = await self._client.fetch_funding_rate_history(symbol, since=since, limit=500)
            if not raw:
                break

            for item in raw:
                ts = int(item.get("timestamp", 0))
                if ts > end_ts:
                    break
                records.append(
                    FundingRateRecord(
                        symbol=symbol,
                        timestamp=ts,
                        funding_rate=Decimal(str(item.get("fundingRate", 0))),
                        mark_price=Decimal(str(item.get("markPrice", item.get("price", 1)))),
                    )
                )

            # 다음 페이지 since 설정
            last_ts = int(raw[-1].get("timestamp", 0))
            if last_ts <= since:
                break
            since = last_ts + 1
            await self._rate_limit_sleep()

        logger.info(
            "Fetched {} funding rate records for {}",
            len(records),
            symbol,
        )
        return records

    async def fetch_open_interest(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        period: str = "1h",
    ) -> list[OpenInterestRecord]:
        """Open Interest 히스토리 수집 (Binance-specific, limit=30).

        1h*30 = 30시간/요청 → 1년 ~300 요청.

        Args:
            symbol: 거래 심볼
            start_ts: 시작 시각 (Unix ms)
            end_ts: 종료 시각 (Unix ms)
            period: 기간 (기본 "1h")

        Returns:
            OpenInterestRecord 리스트
        """
        records: list[OpenInterestRecord] = []
        since = start_ts
        page_limit = 30

        while since < end_ts:
            try:
                raw = await self._client.fetch_open_interest_history(
                    symbol, period=period, since=since, limit=page_limit
                )
            except (CcxtBadRequest, CcxtExchangeError) as e:
                logger.warning(
                    "OI history unavailable for {} (startTime={}): {} — Binance only provides last 30 days",
                    symbol,
                    since,
                    str(e)[:80],
                )
                break
            if not raw:
                break

            for item in raw:
                ts = int(item.get("timestamp", 0))
                if ts > end_ts:
                    break
                records.append(
                    OpenInterestRecord(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                        sum_open_interest=Decimal(str(item.get("sumOpenInterest", 0))),
                        sum_open_interest_value=Decimal(str(item.get("sumOpenInterestValue", 0))),
                    )
                )

            last_ts = int(raw[-1].get("timestamp", 0))
            if last_ts <= since:
                break
            since = last_ts + 1
            await self._rate_limit_sleep()

        logger.info(
            "Fetched {} open interest records for {}",
            len(records),
            symbol,
        )
        return records

    async def fetch_long_short_ratio(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        period: str = "1h",
    ) -> list[LongShortRatioRecord]:
        """Long/Short Ratio 히스토리 수집 (Binance-specific, limit=30).

        Args:
            symbol: 거래 심볼
            start_ts: 시작 시각 (Unix ms)
            end_ts: 종료 시각 (Unix ms)
            period: 기간 (기본 "1h")

        Returns:
            LongShortRatioRecord 리스트
        """
        records: list[LongShortRatioRecord] = []
        since = start_ts
        page_limit = 30

        while since < end_ts:
            try:
                raw = await self._client.fetch_long_short_ratio(
                    symbol, period=period, since=since, limit=page_limit
                )
            except (CcxtBadRequest, CcxtExchangeError) as e:
                logger.warning(
                    "LS ratio history unavailable for {} (startTime={}): {} — Binance only provides last 30 days",
                    symbol,
                    since,
                    str(e)[:80],
                )
                break
            if not raw:
                break

            for item in raw:
                ts = int(item.get("timestamp", 0))
                if ts > end_ts:
                    break
                records.append(
                    LongShortRatioRecord(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                        long_account=Decimal(str(item.get("longAccount", 0))),
                        short_account=Decimal(str(item.get("shortAccount", 0))),
                        long_short_ratio=Decimal(str(item.get("longShortRatio", 0))),
                    )
                )

            last_ts = int(raw[-1].get("timestamp", 0))
            if last_ts <= since:
                break
            since = last_ts + 1
            await self._rate_limit_sleep()

        logger.info(
            "Fetched {} LS ratio records for {}",
            len(records),
            symbol,
        )
        return records

    async def fetch_taker_ratio(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        period: str = "1h",
    ) -> list[TakerRatioRecord]:
        """Taker Buy/Sell Ratio 히스토리 수집 (Binance-specific, limit=30).

        Args:
            symbol: 거래 심볼
            start_ts: 시작 시각 (Unix ms)
            end_ts: 종료 시각 (Unix ms)
            period: 기간 (기본 "1h")

        Returns:
            TakerRatioRecord 리스트
        """
        records: list[TakerRatioRecord] = []
        since = start_ts
        page_limit = 30

        while since < end_ts:
            try:
                raw = await self._client.fetch_taker_buy_sell_ratio(
                    symbol, period=period, since=since, limit=page_limit
                )
            except (CcxtBadRequest, CcxtExchangeError) as e:
                logger.warning(
                    "Taker ratio history unavailable for {} (startTime={}): {} — Binance only provides last 30 days",
                    symbol,
                    since,
                    str(e)[:80],
                )
                break
            if not raw:
                break

            for item in raw:
                ts = int(item.get("timestamp", 0))
                if ts > end_ts:
                    break
                records.append(
                    TakerRatioRecord(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                        buy_vol=Decimal(str(item.get("buyVol", 0))),
                        sell_vol=Decimal(str(item.get("sellVol", 0))),
                        buy_sell_ratio=Decimal(str(item.get("buySellRatio", 0))),
                    )
                )

            last_ts = int(raw[-1].get("timestamp", 0))
            if last_ts <= since:
                break
            since = last_ts + 1
            await self._rate_limit_sleep()

        logger.info(
            "Fetched {} taker ratio records for {}",
            len(records),
            symbol,
        )
        return records

    async def fetch_top_account_ratio(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        period: str = "1h",
    ) -> list[TopTraderAccountRatioRecord]:
        """Top Trader Account Ratio 히스토리 수집 (Binance-specific, limit=30).

        Args:
            symbol: 거래 심볼
            start_ts: 시작 시각 (Unix ms)
            end_ts: 종료 시각 (Unix ms)
            period: 기간 (기본 "1h")

        Returns:
            TopTraderAccountRatioRecord 리스트
        """
        records: list[TopTraderAccountRatioRecord] = []
        since = start_ts
        page_limit = 30

        while since < end_ts:
            try:
                raw = await self._client.fetch_top_long_short_account_ratio(
                    symbol, period=period, since=since, limit=page_limit
                )
            except (CcxtBadRequest, CcxtExchangeError) as e:
                logger.warning(
                    "Top acct ratio history unavailable for {} (startTime={}): {}",
                    symbol,
                    since,
                    str(e)[:80],
                )
                break
            if not raw:
                break

            for item in raw:
                ts = int(item.get("timestamp", 0))
                if ts > end_ts:
                    break
                records.append(
                    TopTraderAccountRatioRecord(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                        long_account=Decimal(str(item.get("longAccount", 0))),
                        short_account=Decimal(str(item.get("shortAccount", 0))),
                        long_short_ratio=Decimal(str(item.get("longShortRatio", 0))),
                    )
                )

            last_ts = int(raw[-1].get("timestamp", 0))
            if last_ts <= since:
                break
            since = last_ts + 1
            await self._rate_limit_sleep()

        logger.info(
            "Fetched {} top acct ratio records for {}",
            len(records),
            symbol,
        )
        return records

    async def fetch_top_position_ratio(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        period: str = "1h",
    ) -> list[TopTraderPositionRatioRecord]:
        """Top Trader Position Ratio 히스토리 수집 (Binance-specific, limit=30).

        Args:
            symbol: 거래 심볼
            start_ts: 시작 시각 (Unix ms)
            end_ts: 종료 시각 (Unix ms)
            period: 기간 (기본 "1h")

        Returns:
            TopTraderPositionRatioRecord 리스트
        """
        records: list[TopTraderPositionRatioRecord] = []
        since = start_ts
        page_limit = 30

        while since < end_ts:
            try:
                raw = await self._client.fetch_top_long_short_position_ratio(
                    symbol, period=period, since=since, limit=page_limit
                )
            except (CcxtBadRequest, CcxtExchangeError) as e:
                logger.warning(
                    "Top pos ratio history unavailable for {} (startTime={}): {}",
                    symbol,
                    since,
                    str(e)[:80],
                )
                break
            if not raw:
                break

            for item in raw:
                ts = int(item.get("timestamp", 0))
                if ts > end_ts:
                    break
                records.append(
                    TopTraderPositionRatioRecord(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                        long_account=Decimal(str(item.get("longAccount", 0))),
                        short_account=Decimal(str(item.get("shortAccount", 0))),
                        long_short_ratio=Decimal(str(item.get("longShortRatio", 0))),
                    )
                )

            last_ts = int(raw[-1].get("timestamp", 0))
            if last_ts <= since:
                break
            since = last_ts + 1
            await self._rate_limit_sleep()

        logger.info(
            "Fetched {} top pos ratio records for {}",
            len(records),
            symbol,
        )
        return records

    async def fetch_year(self, symbol: str, year: int) -> DerivativesBatch:
        """지정 연도의 전체 파생상품 데이터 수집.

        6종 데이터를 순차적으로 수집하여 DerivativesBatch로 반환합니다.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 수집 연도

        Returns:
            DerivativesBatch 객체
        """
        start_ts, end_ts = _get_year_timestamps(year)
        logger.info(
            "Fetching derivatives data for {} year {}",
            symbol,
            year,
        )

        funding_rates = await self.fetch_funding_rates(symbol, start_ts, end_ts)
        oi = await self.fetch_open_interest(symbol, start_ts, end_ts)
        ls_ratio = await self.fetch_long_short_ratio(symbol, start_ts, end_ts)
        taker = await self.fetch_taker_ratio(symbol, start_ts, end_ts)
        top_acct = await self.fetch_top_account_ratio(symbol, start_ts, end_ts)
        top_pos = await self.fetch_top_position_ratio(symbol, start_ts, end_ts)

        batch = DerivativesBatch(
            symbol=symbol,
            funding_rates=tuple(funding_rates),
            open_interest=tuple(oi),
            long_short_ratios=tuple(ls_ratio),
            taker_ratios=tuple(taker),
            top_acct_ratios=tuple(top_acct),
            top_pos_ratios=tuple(top_pos),
        )

        logger.info(
            "Derivatives fetch complete for {} {}: {} total records (FR={}, OI={}, LS={}, Taker={}, TopAcct={}, TopPos={})",
            symbol,
            year,
            batch.total_records,
            len(funding_rates),
            len(oi),
            len(ls_ratio),
            len(taker),
            len(top_acct),
            len(top_pos),
        )

        return batch
