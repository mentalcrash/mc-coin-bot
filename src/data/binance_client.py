"""Binance API client using CCXT library."""

import asyncio
from typing import Any

import ccxt.async_support as ccxt
from loguru import logger

from src.data.schemas import RawBinanceKline, TickerInfo


class BinanceClient:
    """바이낸스 API 클라이언트 (CCXT 래핑).

    Rate limit을 준수하며 캔들 데이터를 수집합니다.
    """

    def __init__(self, rate_limit: bool = True) -> None:
        """BinanceClient 초기화.

        Args:
            rate_limit: Rate limit 활성화 여부 (기본값: True)
        """
        self._exchange: ccxt.binance | None = None
        self._rate_limit = rate_limit

    async def __aenter__(self) -> "BinanceClient":
        """Async context manager 진입."""
        self._exchange = ccxt.binance(
            {
                "enableRateLimit": self._rate_limit,
                "options": {
                    "defaultType": "spot",
                },
            }
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager 종료."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    @property
    def exchange(self) -> ccxt.binance:
        """CCXT exchange 인스턴스 반환."""
        if self._exchange is None:
            msg = "BinanceClient must be used as async context manager"
            raise RuntimeError(msg)
        return self._exchange

    async def get_top_usdt_tickers(self, top_n: int = 100) -> list[str]:
        """거래대금 기준 상위 USDT 페어 티커 목록 조회.

        Args:
            top_n: 상위 N개 티커 수 (기본값: 100)

        Returns:
            상위 티커 심볼 리스트 (예: ['BTCUSDT', 'ETHUSDT', ...])
        """
        logger.info(f"Fetching top {top_n} USDT tickers by quote volume...")

        # 24시간 티커 데이터 조회
        tickers = await self.exchange.fetch_tickers()

        # USDT 페어만 필터링하고 거래대금 기준 정렬
        usdt_tickers: list[TickerInfo] = []
        for symbol, ticker in tickers.items():
            if symbol.endswith("/USDT") and ticker.get("quoteVolume"):
                usdt_tickers.append(
                    TickerInfo(
                        symbol=symbol.replace("/", ""),  # CCXT는 "BTC/USDT" 형식 사용
                        quote_volume=float(ticker["quoteVolume"]),
                    )
                )

        # 거래대금 내림차순 정렬
        usdt_tickers.sort(key=lambda x: x.quote_volume, reverse=True)

        # 상위 N개 선택
        top_symbols = [t.symbol for t in usdt_tickers[:top_n]]

        logger.info(f"Found {len(top_symbols)} USDT tickers")
        logger.debug(f"Top 10: {top_symbols[:10]}")

        return top_symbols

    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int = 1000,
    ) -> list[RawBinanceKline]:
        """캔들 데이터 조회.

        Args:
            symbol: 심볼 (예: 'BTCUSDT')
            timeframe: 타임프레임 (기본값: '1m')
            since: 시작 타임스탬프 (밀리초)
            limit: 조회 개수 (최대 1000)

        Returns:
            RawBinanceKline 리스트
        """
        # CCXT는 "BTC/USDT" 형식 사용
        ccxt_symbol = symbol[:-4] + "/" + symbol[-4:] if "USDT" in symbol else symbol

        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            # CCXT 응답을 RawBinanceKline으로 변환
            # CCXT fetch_ohlcv 응답: [timestamp, open, high, low, close, volume]
            # 바이낸스 원본 필드 중 일부가 누락되므로 직접 API 호출 필요
            return [
                RawBinanceKline(
                    open_time=candle[0],
                    open=str(candle[1]),
                    high=str(candle[2]),
                    low=str(candle[3]),
                    close=str(candle[4]),
                    volume=str(candle[5]),
                    close_time=candle[0] + 60000 - 1,  # 1분봉 기준
                    quote_volume="0",  # CCXT에서 제공하지 않음
                    trades=0,  # CCXT에서 제공하지 않음
                    taker_buy_volume="0",  # CCXT에서 제공하지 않음
                    taker_buy_quote_volume="0",  # CCXT에서 제공하지 않음
                )
                for candle in ohlcv
            ]

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            raise

    async def fetch_klines_raw(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[RawBinanceKline]:
        """바이낸스 API 직접 호출하여 모든 필드가 포함된 캔들 데이터 조회.

        Args:
            symbol: 심볼 (예: 'BTCUSDT')
            interval: 타임프레임 (기본값: '1m')
            start_time: 시작 타임스탬프 (밀리초)
            end_time: 종료 타임스탬프 (밀리초)
            limit: 조회 개수 (최대 1000)

        Returns:
            RawBinanceKline 리스트 (모든 필드 포함)
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            # CCXT의 publicGetKlines 메서드 사용 (바이낸스 원본 API)
            response = await self.exchange.public_get_klines(params)

            return [RawBinanceKline.from_binance_response(candle) for candle in response]

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            raise

    async def fetch_klines_paginated(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        interval: str = "1m",
        limit: int = 1000,
        delay_ms: int = 25,
    ) -> list[RawBinanceKline]:
        """페이지네이션을 사용하여 대량의 캔들 데이터 조회.

        Args:
            symbol: 심볼 (예: 'BTCUSDT')
            start_time: 시작 타임스탬프 (밀리초)
            end_time: 종료 타임스탬프 (밀리초)
            interval: 타임프레임 (기본값: '1m')
            limit: 요청당 조회 개수 (최대 1000)
            delay_ms: 요청 간 딜레이 (밀리초)

        Returns:
            RawBinanceKline 리스트
        """
        all_klines: list[RawBinanceKline] = []
        current_start = start_time

        while current_start < end_time:
            klines = await self.fetch_klines_raw(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=limit,
            )

            if not klines:
                break

            all_klines.extend(klines)

            # 다음 요청의 시작 시간 설정 (마지막 캔들 시간 + 1분)
            last_time = klines[-1].open_time
            current_start = last_time + 60000  # 1분 = 60000ms

            # Rate limit 준수를 위한 딜레이
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

            # 진행 상황 로깅 (1000개마다)
            if len(all_klines) % 10000 == 0:
                logger.info(f"{symbol}: Fetched {len(all_klines)} candles...")

        logger.info(f"{symbol}: Total {len(all_klines)} candles fetched")
        return all_klines
