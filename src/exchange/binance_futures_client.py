"""Binance USDT-M Futures 비동기 클라이언트.

Hedge Mode + Cross Margin + 1x Leverage 기본 설정.
async context manager 패턴. 주문 실행 전용 (데이터 스트리밍은 BinanceClient 사용).

Rules Applied:
    - #14 CCXT Trading: ccxt.pro, async context manager, load_markets()
    - String Protocol: amount_to_precision / price_to_precision 필수
    - Idempotency: client_order_id (newClientOrderId) 사용
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import ccxt.pro as ccxt
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import (
    AuthenticationError,
    InsufficientFundsError,
    NetworkError,
    OrderExecutionError,
    RateLimitError,
)

if TYPE_CHECKING:
    from types import TracebackType

    from ccxt.pro import binance

# 재시도 가능한 CCXT 예외 타입
_RETRYABLE_EXCEPTIONS = (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RequestTimeout)

# 재시도 설정
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds


class BinanceFuturesClient:
    """Binance USDT-M Futures 비동기 클라이언트.

    Hedge Mode + Cross Margin + 1x Leverage 기본 설정.
    async context manager 패턴. 주문 실행 전용 (데이터 스트리밍은 BinanceClient 사용).

    Example:
        >>> async with BinanceFuturesClient() as client:
        ...     result = await client.create_order(
        ...         symbol="BTC/USDT:USDT", side="buy", amount=0.001,
        ...         position_side="LONG",
        ...     )
    """

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._exchange: binance | None = None
        self._initialized = False

    async def __aenter__(self) -> BinanceFuturesClient:
        await self._initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _initialize(self) -> None:
        """거래소 인스턴스 초기화 및 마켓 로드."""
        if self._initialized:
            return

        exchange_config: dict[str, Any] = {
            "enableRateLimit": True,
            "rateLimit": 60000 // self._settings.rate_limit_per_minute,
            "timeout": self._settings.request_timeout * 1000,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }

        if not self._settings.has_api_credentials():
            msg = "Binance API credentials required for futures trading"
            raise AuthenticationError(msg)

        exchange_config["apiKey"] = self._settings.binance_api_key.get_secret_value()
        exchange_config["secret"] = self._settings.binance_secret.get_secret_value()

        try:
            self._exchange = ccxt.binance(exchange_config)  # type: ignore[arg-type]
            await self._exchange.load_markets()
            self._initialized = True
            markets: dict[str, Any] | None = self._exchange.markets  # type: ignore[assignment]
            logger.info(
                "Binance Futures client initialized",
                extra={"markets_loaded": len(markets) if markets else 0},
            )
        except ccxt.AuthenticationError as e:
            await self.close()
            raise AuthenticationError(
                "Binance Futures API authentication failed",
                context={"error": str(e)},
            ) from e
        except ccxt.NetworkError as e:
            await self.close()
            raise NetworkError(
                "Failed to connect to Binance Futures",
                context={"error": str(e)},
            ) from e
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """리소스 정리 및 연결 종료."""
        if self._exchange is not None:
            try:
                await self._exchange.close()
                logger.debug("Binance Futures client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Binance Futures connection: {e}")
            finally:
                self._exchange = None
                self._initialized = False

    @property
    def exchange(self) -> binance:  # type: ignore[name-defined]
        """초기화된 거래소 인스턴스."""
        if self._exchange is None:
            msg = "BinanceFuturesClient not initialized. Use 'async with BinanceFuturesClient() as client:'"
            raise RuntimeError(msg)
        return self._exchange

    async def setup_account(self, symbols: list[str], *, leverage: int = 1) -> None:
        """계정 설정: Hedge Mode + Cross Margin + 지정 Leverage.

        이미 설정된 경우 무시합니다 (idempotent).

        Args:
            symbols: 거래할 심볼 리스트 (예: ["BTC/USDT:USDT"])
            leverage: 심볼별 레버리지 배수 (기본 1x)
        """
        # Hedge Mode 설정
        try:
            await self.exchange.set_position_mode(hedged=True)  # type: ignore[arg-type]
            logger.info("Hedge Mode enabled")
        except ccxt.ExchangeError as e:
            if "No need to change" in str(e):
                logger.debug("Hedge Mode already enabled")
            else:
                raise OrderExecutionError(
                    "Failed to set Hedge Mode",
                    context={"error": str(e)},
                ) from e

        # 심볼별 leverage + margin mode 설정
        for symbol in symbols:
            futures_symbol = self.to_futures_symbol(symbol)
            try:
                await self.exchange.set_leverage(leverage, futures_symbol)  # type: ignore[arg-type]
                logger.info("Leverage set to {}x for {}", leverage, futures_symbol)
            except ccxt.ExchangeError as e:
                if "No need to change" in str(e):
                    pass
                else:
                    logger.warning("Failed to set leverage for {}: {}", futures_symbol, e)

            try:
                await self.exchange.set_margin_mode("cross", futures_symbol)  # type: ignore[arg-type]
            except ccxt.ExchangeError as e:
                if "No need to change" in str(e):
                    pass
                else:
                    logger.warning("Failed to set margin mode for {}: {}", futures_symbol, e)

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_side: str,
        *,
        price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """주문 생성.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT:USDT")
            side: "buy" 또는 "sell"
            amount: 주문 수량
            position_side: "LONG" 또는 "SHORT"
            price: 지정가 (None이면 시장가)
            reduce_only: 청산 전용 주문 여부
            client_order_id: 멱등성 키

        Returns:
            CCXT 주문 응답 dict

        Raises:
            InsufficientFundsError: 잔고 부족
            OrderExecutionError: 주문 실행 실패
        """
        order_type = "limit" if price is not None else "market"
        safe_amount = self.exchange.amount_to_precision(symbol, amount)  # type: ignore[no-untyped-call]

        params: dict[str, Any] = {
            "positionSide": position_side,
            "recvWindow": 5000,
        }
        if reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        safe_price: str | None = None
        if price is not None:
            safe_price = self.exchange.price_to_precision(symbol, price)  # type: ignore[no-untyped-call]

        async def _do_create() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.create_order(  # type: ignore[assignment,arg-type]
                symbol,
                order_type,
                side,  # pyright: ignore[reportArgumentType]
                safe_amount,  # pyright: ignore[reportArgumentType]
                safe_price,
                params,
            )
            return result

        try:
            return await self._retry_with_backoff(_do_create)
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                "Insufficient funds for order",
                context={"symbol": symbol, "side": side, "amount": amount, "error": str(e)},
            ) from e
        except ccxt.InvalidOrder as e:
            raise OrderExecutionError(
                "Invalid order parameters",
                context={"symbol": symbol, "side": side, "amount": amount, "error": str(e)},
            ) from e
        except ccxt.ExchangeError as e:
            raise OrderExecutionError(
                "Order execution failed",
                context={"symbol": symbol, "side": side, "error": str(e)},
            ) from e

    async def fetch_positions(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """오픈 포지션 조회.

        Args:
            symbols: 조회할 심볼 리스트 (None이면 전체)

        Returns:
            포지션 리스트 (size > 0인 것만)
        """

        async def _do_fetch() -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = await self.exchange.fetch_positions(symbols)  # type: ignore[assignment]
            return result

        positions = await self._retry_with_backoff(_do_fetch)
        return [p for p in positions if abs(float(p.get("contracts", 0))) > 0]

    async def fetch_balance(self) -> dict[str, Any]:
        """USDT 잔고 조회.

        Returns:
            잔고 정보 dict (total, free, used)
        """

        async def _do_fetch() -> dict[str, Any]:
            balance: dict[str, Any] = await self.exchange.fetch_balance()  # type: ignore[assignment]
            return balance

        return await self._retry_with_backoff(_do_fetch)

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """주문 취소.

        Args:
            order_id: 주문 ID
            symbol: 거래 심볼

        Returns:
            취소 결과 dict
        """

        async def _do_cancel() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.cancel_order(order_id, symbol)  # type: ignore[assignment]
            return result

        return await self._retry_with_backoff(_do_cancel)

    @staticmethod
    def to_futures_symbol(symbol: str) -> str:
        """Spot 심볼 → Futures 심볼 변환.

        Args:
            symbol: "BTC/USDT" 또는 이미 "BTC/USDT:USDT" 형태

        Returns:
            "BTC/USDT:USDT" 형태의 Futures 심볼
        """
        if ":USDT" in symbol:
            return symbol
        return f"{symbol}:USDT"

    @staticmethod
    async def _retry_with_backoff(
        func: Any,
        max_retries: int = _MAX_RETRIES,
        base_backoff: float = _BASE_BACKOFF,
    ) -> Any:
        """재시도 래퍼 (network/rate limit만 재시도).

        Args:
            func: 실행할 async 함수
            max_retries: 최대 재시도 횟수
            base_backoff: 기본 백오프 시간 (초)

        Returns:
            함수 실행 결과
        """
        last_exc: Exception | None = None
        is_rate_limit = False
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except ccxt.RateLimitExceeded as e:
                # RateLimitExceeded는 NetworkError의 서브클래스이므로 먼저 체크
                last_exc = e
                is_rate_limit = True
                if attempt < max_retries:
                    wait = base_backoff * (2 ** (attempt + 1))
                    logger.warning(
                        "Rate limited (attempt {}/{}), waiting {:.1f}s",
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    break
            except _RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                if attempt < max_retries:
                    wait = base_backoff * (2**attempt)
                    logger.warning(
                        "Retryable error (attempt {}/{}): {}, waiting {:.1f}s",
                        attempt + 1,
                        max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    break

        assert last_exc is not None
        if is_rate_limit:
            raise RateLimitError(
                "Rate limit exceeded after retries",
                retry_after=60.0,
                context={"error": str(last_exc)},
            ) from last_exc
        raise NetworkError(
            "Network error after retries",
            context={"error": str(last_exc)},
        ) from last_exc
