"""CCXT Pro Binance client for async market data fetching.

This module provides a high-level async interface to Binance exchange
using CCXT Pro (WebSocket-first architecture). It handles connection
management, rate limiting, and error handling.

Features:
    - Async context manager for proper resource cleanup
    - Automatic load_markets() on initialization
    - Rate limit handling with configurable parameters
    - String types for precision-sensitive values

Rules Applied:
    - #14 CCXT Trading: ccxt.pro, async context manager, load_markets()
    - #10 Python Standards: Async/await, TaskGroup
"""

from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType
from typing import TYPE_CHECKING, Any

import ccxt.pro as ccxt
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
)
from src.models.ohlcv import OHLCVBatch, OHLCVCandle

if TYPE_CHECKING:
    from ccxt.pro import binance


class BinanceClient:
    """Binance 거래소 비동기 클라이언트.

    CCXT Pro를 사용하여 Binance API와 통신합니다.
    async context manager 패턴을 사용하여 연결 수명 주기를 관리합니다.

    Attributes:
        settings: 설정 객체
        exchange: CCXT Binance 인스턴스 (초기화 전 None)

    Example:
        >>> async with BinanceClient() as client:
        ...     batch = await client.fetch_ohlcv("BTC/USDT", "1m", since_ts)
        ...     print(f"Fetched {batch.candle_count} candles")
    """

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        """BinanceClient 초기화.

        Args:
            settings: 설정 객체 (None이면 기본 설정 사용)
        """
        self.settings = settings or get_settings()
        self._exchange: binance | None = None
        self._initialized = False

    async def __aenter__(self) -> "BinanceClient":
        """비동기 컨텍스트 매니저 진입.

        거래소 인스턴스를 생성하고 markets 정보를 로드합니다.

        Returns:
            초기화된 BinanceClient 인스턴스

        Raises:
            AuthenticationError: API 키가 유효하지 않을 경우
            NetworkError: 네트워크 연결 실패 시
        """
        await self._initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """비동기 컨텍스트 매니저 종료.

        거래소 연결을 정리합니다.
        """
        await self._cleanup()

    async def _initialize(self) -> None:
        """거래소 인스턴스 초기화 및 마켓 로드.

        Raises:
            AuthenticationError: API 키가 유효하지 않을 경우
            NetworkError: 네트워크 연결 실패 시
        """
        if self._initialized:
            return

        # CCXT Pro 설정 (WebSocket-first)
        exchange_config: dict[str, Any] = {
            "enableRateLimit": True,
            "rateLimit": 60000 // self.settings.rate_limit_per_minute,  # ms per request
            "timeout": self.settings.request_timeout * 1000,  # ms
            "options": {
                "defaultType": "spot",  # spot 마켓 기본
                "adjustForTimeDifference": True,  # 시간 동기화
            },
        }

        # API 자격 증명이 있으면 추가
        if self.settings.has_api_credentials():
            exchange_config["apiKey"] = self.settings.binance_api_key.get_secret_value()
            exchange_config["secret"] = self.settings.binance_secret.get_secret_value()

        try:
            self._exchange = ccxt.binance(exchange_config)  # type: ignore[arg-type]
            # [중요] 마켓 정보 로드 - 정밀도 정보 필수
            await self._exchange.load_markets()
            self._initialized = True
            markets: dict[str, Any] | None = self._exchange.markets  # type: ignore[assignment]
            logger.info(
                "Binance client initialized",
                extra={"markets_loaded": len(markets) if markets else 0},
            )
        except ccxt.AuthenticationError as e:
            # 초기화 실패 시 리소스 정리
            await self._cleanup()
            raise AuthenticationError(
                "Binance API authentication failed",
                context={"error": str(e)},
            ) from e
        except ccxt.NetworkError as e:
            # 초기화 실패 시 리소스 정리
            await self._cleanup()
            raise NetworkError(
                "Failed to connect to Binance",
                context={"error": str(e)},
            ) from e
        except Exception:
            # 예상치 못한 예외 시에도 리소스 정리
            await self._cleanup()
            raise

    async def _cleanup(self) -> None:
        """리소스 정리 및 연결 종료."""
        if self._exchange is not None:
            try:
                await self._exchange.close()
                logger.debug("Binance client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Binance connection: {e}")
            finally:
                self._exchange = None
                self._initialized = False

    @property
    def exchange(self) -> "binance":  # type: ignore[name-defined]
        """초기화된 거래소 인스턴스 반환.

        Returns:
            CCXT Binance 인스턴스

        Raises:
            RuntimeError: 초기화되지 않은 경우
        """
        if self._exchange is None:
            msg = "BinanceClient not initialized. Use 'async with BinanceClient() as client:'"
            raise RuntimeError(msg)
        return self._exchange

    async def fetch_ohlcv_raw(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        """원본 OHLCV 데이터 페칭 (CCXT 응답 그대로).

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            timeframe: 타임프레임 (기본: "1m")
            since: 시작 타임스탬프 (Unix ms)
            limit: 최대 캔들 수 (기본: settings.batch_size)

        Returns:
            [[timestamp, open, high, low, close, volume], ...] 형태의 리스트

        Raises:
            NetworkError: 네트워크 오류 시
            RateLimitError: 레이트 리밋 초과 시
        """
        limit = limit or self.settings.batch_size

        try:
            result: list[list[Any]] = await self.exchange.fetch_ohlcv(  # type: ignore[assignment]
                symbol,
                timeframe,
                since=since,
                limit=limit,
            )
        except ccxt.RateLimitExceeded as e:
            raise RateLimitError(
                "Binance rate limit exceeded",
                retry_after=60.0,  # 기본 1분 대기
                context={"symbol": symbol, "error": str(e)},
            ) from e
        except ccxt.NetworkError as e:
            raise NetworkError(
                "Network error during OHLCV fetch",
                context={"symbol": symbol, "error": str(e)},
            ) from e
        else:
            return result

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> OHLCVBatch:
        """OHLCV 데이터를 Pydantic 모델로 변환하여 반환.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            timeframe: 타임프레임 (기본: "1m")
            since: 시작 타임스탬프 (Unix ms)
            limit: 최대 캔들 수

        Returns:
            검증된 OHLCVBatch 객체

        Raises:
            NetworkError: 네트워크 오류 시
            RateLimitError: 레이트 리밋 초과 시
            DataValidationError: 데이터 검증 실패 시
        """
        raw_candles = await self.fetch_ohlcv_raw(symbol, timeframe, since, limit)

        # CCXT 응답을 Pydantic 모델로 변환
        candles = tuple(
            OHLCVCandle(
                timestamp=c[0],  # Unix ms -> datetime (validator에서 변환)
                open=Decimal(str(c[1])),
                high=Decimal(str(c[2])),
                low=Decimal(str(c[3])),
                close=Decimal(str(c[4])),
                volume=Decimal(str(c[5])),
            )
            for c in raw_candles
        )

        return OHLCVBatch(
            symbol=symbol,
            timeframe=timeframe,
            exchange="binance",
            candles=candles,
            fetched_at=datetime.now(UTC),
        )

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """심볼의 마켓 정보 반환.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")

        Returns:
            마켓 정보 딕셔너리 또는 None
        """
        markets: dict[str, Any] | None = self.exchange.markets  # type: ignore[assignment]
        if markets is None:
            return None
        return markets.get(symbol)  # pyright: ignore[reportUnknownVariableType]

    def is_symbol_valid(self, symbol: str) -> bool:
        """심볼이 유효한지 확인.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")

        Returns:
            유효하면 True
        """
        markets: dict[str, Any] | None = self.exchange.markets  # type: ignore[assignment]
        if markets is None:
            return False
        return symbol in markets

    async def fetch_top_symbols(
        self,
        quote: str = "USDT",
        limit: int = 100,
    ) -> list[str]:
        """24시간 거래대금(quoteVolume) 기준 상위 N개 심볼 반환.

        Binance의 24시간 거래 통계를 조회하여 Quote Volume 기준으로
        상위 거래량 종목을 선정합니다.

        Args:
            quote: Quote 통화 (기본: "USDT")
            limit: 반환할 심볼 수 (기본: 100)

        Returns:
            거래대금 기준 상위 N개 심볼 리스트 (예: ["BTC/USDT", "ETH/USDT", ...])

        Raises:
            NetworkError: 네트워크 오류 시

        Example:
            >>> async with BinanceClient() as client:
            ...     top_symbols = await client.fetch_top_symbols(limit=10)
            ...     print(top_symbols)
            ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', ...]
        """
        try:
            tickers: dict[str, Any] = await self.exchange.fetch_tickers()  # type: ignore[assignment]
        except ccxt.NetworkError as e:
            raise NetworkError(
                "Failed to fetch tickers from Binance",
                context={"error": str(e)},
            ) from e

        # Quote 통화 페어만 필터링 + quoteVolume 기준 정렬
        quote_suffix = f"/{quote}"
        filtered_tickers: list[tuple[str, float]] = []

        for symbol, data in tickers.items():
            if not symbol.endswith(quote_suffix):
                continue
            quote_volume = data.get("quoteVolume")
            if quote_volume is None or quote_volume <= 0:
                continue
            filtered_tickers.append((symbol, float(quote_volume)))

        # 거래대금 내림차순 정렬
        filtered_tickers.sort(key=lambda x: x[1], reverse=True)

        # 상위 N개 심볼만 추출
        top_symbols = [symbol for symbol, _ in filtered_tickers[:limit]]

        logger.info(
            f"Fetched top {len(top_symbols)} symbols by quote volume",
            extra={
                "quote": quote,
                "requested": limit,
                "available": len(filtered_tickers),
                "top_3": top_symbols[:3] if top_symbols else [],
            },
        )

        return top_symbols
