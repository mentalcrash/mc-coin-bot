"""Binance Spot 비동기 클라이언트.

Spot 현물 거래 전용. async context manager 패턴.
주문 실행 전용 (데이터 스트리밍은 BinanceClient 사용).

Rules Applied:
    - #14 CCXT Trading: ccxt.pro, async context manager, load_markets()
    - String Protocol: amount_to_precision / price_to_precision / cost_to_precision 필수
    - Idempotency: client_order_id (newClientOrderId) 사용
"""

from __future__ import annotations

import asyncio
import random
import time
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

    from src.monitoring.metrics import ApiMetricsCallback

# 재시도 가능한 CCXT 예외 타입
_RETRYABLE_EXCEPTIONS = (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RequestTimeout)

# 재시도 설정
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds


class BinanceSpotClient:
    """Binance Spot 비동기 클라이언트.

    async context manager 패턴. 주문 실행 전용 (데이터 스트리밍은 BinanceClient 사용).

    Example:
        >>> async with BinanceSpotClient() as client:
        ...     result = await client.create_market_buy(
        ...         symbol="BTC/USDT", quote_amount=100.0,
        ...         client_order_id="spot_BTC_USDT_123",
        ...     )
    """

    # API 연속 실패 시 주문 차단
    _MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        metrics_callback: ApiMetricsCallback | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._exchange: binance | None = None
        self._initialized = False
        self._consecutive_failures = 0
        self._metrics_callback = metrics_callback

    async def __aenter__(self) -> BinanceSpotClient:
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
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            },
        }

        if not self._settings.has_api_credentials():
            msg = "Binance API credentials required for spot trading"
            raise AuthenticationError(msg)

        exchange_config["apiKey"] = self._settings.binance_api_key.get_secret_value()
        exchange_config["secret"] = self._settings.binance_secret.get_secret_value()

        try:
            self._exchange = ccxt.binance(exchange_config)  # type: ignore[arg-type]
            await self._exchange.load_markets()
            self._initialized = True
            markets: dict[str, Any] | None = self._exchange.markets  # type: ignore[assignment]
            logger.info(
                "Binance Spot client initialized",
                extra={"markets_loaded": len(markets) if markets else 0},
            )
        except ccxt.AuthenticationError as e:
            await self.close()
            raise AuthenticationError(
                "Binance Spot API authentication failed",
                context={"error": str(e)},
            ) from e
        except ccxt.NetworkError as e:
            await self.close()
            raise NetworkError(
                "Failed to connect to Binance Spot",
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
                logger.debug("Binance Spot client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Binance Spot connection: {e}")
            finally:
                self._exchange = None
                self._initialized = False

    @property
    def exchange(self) -> binance:  # type: ignore[name-defined]
        """초기화된 거래소 인스턴스."""
        if self._exchange is None:
            msg = (
                "BinanceSpotClient not initialized. Use 'async with BinanceSpotClient() as client:'"
            )
            raise RuntimeError(msg)
        return self._exchange

    @property
    def is_api_healthy(self) -> bool:
        """API 연속 실패가 임계값 미만인지 확인."""
        return self._consecutive_failures < self._MAX_CONSECUTIVE_FAILURES

    @property
    def consecutive_failures(self) -> int:
        """연속 API 실패 횟수."""
        return self._consecutive_failures

    def _record_success(self) -> None:
        """API 호출 성공 → 실패 카운터 리셋."""
        if self._consecutive_failures > 0:
            logger.debug(
                "API recovery: consecutive failures reset from {}", self._consecutive_failures
            )
        self._consecutive_failures = 0

    def _record_failure(self, exc: Exception | None = None) -> None:
        """API 호출 실패 → 실패 카운터 증가."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
            logger.critical(
                "API CIRCUIT BREAKER: {} consecutive failures — orders will be blocked",
                self._consecutive_failures,
            )
        if exc is not None:
            try:
                from src.monitoring.metrics import errors_counter

                errors_counter.labels(component="SpotExchange", error_type=type(exc).__name__).inc()
            except Exception:  # noqa: S110
                pass

    def set_metrics_callback(self, callback: ApiMetricsCallback) -> None:
        """API 메트릭 콜백 설정."""
        self._metrics_callback = callback

    def _report_metric(self, endpoint: str, duration: float, status: str) -> None:
        """API 메트릭 콜백 호출 (콜백이 있을 때만)."""
        if self._metrics_callback is not None:
            self._metrics_callback.on_api_call(endpoint, duration, status)

    # =========================================================================
    # Trading Methods
    # =========================================================================

    async def create_market_buy(
        self,
        symbol: str,
        quote_amount: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Spot Market Buy (USDT 금액 기준).

        quoteOrderQty로 USDT 금액만큼 매수합니다.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            quote_amount: 매수 금액 (USDT)
            client_order_id: 멱등성 키

        Returns:
            CCXT 주문 응답 dict
        """
        safe_cost = self.exchange.cost_to_precision(symbol, quote_amount)  # type: ignore[no-untyped-call]

        params: dict[str, Any] = {"quoteOrderQty": safe_cost, "recvWindow": 5000}
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        async def _do_create() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.create_order(  # type: ignore[assignment,arg-type]
                symbol,
                "market",
                "buy",
                None,  # amount ignored when quoteOrderQty is set  # pyright: ignore[reportArgumentType]
                None,
                params,
            )
            return result

        return await self._execute_order(_do_create, "create_market_buy", symbol, quote_amount)

    async def create_market_sell(
        self,
        symbol: str,
        base_amount: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Spot Market Sell (base 수량 기준).

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            base_amount: 매도 수량 (base asset)
            client_order_id: 멱등성 키

        Returns:
            CCXT 주문 응답 dict
        """
        safe_amount = self.exchange.amount_to_precision(symbol, base_amount)  # type: ignore[no-untyped-call]

        params: dict[str, Any] = {"recvWindow": 5000}
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        async def _do_create() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.create_order(  # type: ignore[assignment,arg-type]
                symbol,
                "market",
                "sell",
                safe_amount,  # pyright: ignore[reportArgumentType]
                None,
                params,
            )
            return result

        return await self._execute_order(_do_create, "create_market_sell", symbol, base_amount)

    async def create_stop_limit_sell(
        self,
        symbol: str,
        base_amount: float,
        stop_price: float,
        limit_price: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Spot STOP_LOSS_LIMIT Sell 주문.

        stop_price에 도달하면 limit_price로 지정가 매도 주문이 발동됩니다.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            base_amount: 매도 수량 (base asset)
            stop_price: 트리거 가격
            limit_price: 지정가 가격
            client_order_id: 멱등성 키

        Returns:
            CCXT 주문 응답 dict
        """
        safe_amount = self.exchange.amount_to_precision(symbol, base_amount)  # type: ignore[no-untyped-call]
        safe_stop = self.exchange.price_to_precision(symbol, stop_price)  # type: ignore[no-untyped-call]
        safe_limit = self.exchange.price_to_precision(symbol, limit_price)  # type: ignore[no-untyped-call]

        params: dict[str, Any] = {"stopPrice": safe_stop, "recvWindow": 5000}
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        async def _do_create() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.create_order(  # type: ignore[assignment,arg-type]
                symbol,
                "STOP_LOSS_LIMIT",  # pyright: ignore[reportArgumentType]
                "sell",
                safe_amount,  # pyright: ignore[reportArgumentType]
                safe_limit,
                params,
            )
            return result

        return await self._execute_order(_do_create, "create_stop_limit_sell", symbol, base_amount)

    async def _execute_order(
        self,
        func: Any,
        endpoint: str,
        symbol: str,
        amount: float,
    ) -> dict[str, Any]:
        """주문 실행 공통 래퍼 (에러 핸들링 + 메트릭).

        Args:
            func: 실행할 async 함수
            endpoint: API endpoint 이름
            symbol: 거래 심볼
            amount: 주문 수량/금액

        Returns:
            CCXT 주문 응답 dict
        """
        t0 = time.monotonic()
        try:
            result = await self._retry_with_backoff(func)
        except ccxt.InsufficientFunds as e:
            self._record_failure(e)
            self._report_metric(endpoint, time.monotonic() - t0, "failure")
            raise InsufficientFundsError(
                "Insufficient funds for spot order",
                context={"symbol": symbol, "amount": amount, "error": str(e)},
            ) from e
        except ccxt.InvalidOrder as e:
            self._report_metric(endpoint, time.monotonic() - t0, "failure")
            raise OrderExecutionError(
                "Invalid spot order parameters",
                context={"symbol": symbol, "amount": amount, "error": str(e)},
            ) from e
        except ccxt.ExchangeError as e:
            self._record_failure(e)
            self._report_metric(endpoint, time.monotonic() - t0, "failure")
            raise OrderExecutionError(
                "Spot order execution failed",
                context={"symbol": symbol, "error": str(e)},
            ) from e
        except (NetworkError, RateLimitError) as e:
            self._record_failure(e)
            self._report_metric(endpoint, time.monotonic() - t0, "failure")
            raise
        else:
            self._record_success()
            self._report_metric(endpoint, time.monotonic() - t0, "success")
            return result

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def fetch_balance(self) -> dict[str, Any]:
        """Spot 잔고 조회.

        Returns:
            잔고 정보 dict (각 asset의 total, free, used)
        """

        async def _do_fetch() -> dict[str, Any]:
            balance: dict[str, Any] = await self.exchange.fetch_balance()  # type: ignore[assignment]
            return balance

        t0 = time.monotonic()
        try:
            result = await self._retry_with_backoff(_do_fetch)
        except Exception:
            self._report_metric("fetch_balance", time.monotonic() - t0, "failure")
            raise
        self._report_metric("fetch_balance", time.monotonic() - t0, "success")
        return result

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """실시간 가격 조회.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")

        Returns:
            Ticker dict (last, bid, ask 등 포함)
        """

        async def _do_fetch() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.fetch_ticker(symbol)  # type: ignore[assignment]
            return result

        t0 = time.monotonic()
        try:
            result = await self._retry_with_backoff(_do_fetch)
        except Exception:
            self._report_metric("fetch_ticker", time.monotonic() - t0, "failure")
            raise
        self._report_metric("fetch_ticker", time.monotonic() - t0, "success")
        return result

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """미체결 주문 조회.

        Args:
            symbol: 거래 심볼 (None이면 전체)

        Returns:
            미체결 주문 리스트
        """

        async def _do_fetch() -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = await self.exchange.fetch_open_orders(symbol)  # type: ignore[assignment]
            return result

        t0 = time.monotonic()
        try:
            result = await self._retry_with_backoff(_do_fetch)
        except Exception:
            self._report_metric("fetch_open_orders", time.monotonic() - t0, "failure")
            raise
        self._report_metric("fetch_open_orders", time.monotonic() - t0, "success")
        return result

    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """주문 상태 조회.

        Args:
            order_id: 주문 ID
            symbol: 거래 심볼

        Returns:
            주문 상태 dict
        """

        async def _do_fetch() -> dict[str, Any]:
            result: dict[str, Any] = await self.exchange.fetch_order(order_id, symbol)  # type: ignore[assignment]
            return result

        t0 = time.monotonic()
        try:
            result = await self._retry_with_backoff(_do_fetch)
        except Exception:
            self._report_metric("fetch_order", time.monotonic() - t0, "failure")
            raise
        self._report_metric("fetch_order", time.monotonic() - t0, "success")
        return result

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

        t0 = time.monotonic()
        try:
            result = await self._retry_with_backoff(_do_cancel)
        except Exception:
            self._report_metric("cancel_order", time.monotonic() - t0, "failure")
            raise
        self._report_metric("cancel_order", time.monotonic() - t0, "success")
        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_min_notional(self, symbol: str) -> float:
        """심볼의 MIN_NOTIONAL 값 조회.

        Args:
            symbol: 거래 심볼

        Returns:
            최소 주문 금액 (USD). 정보 없으면 10.0 (Binance Spot 기본값)
        """
        _default_min_notional = 10.0
        try:
            market: dict[str, Any] = self.exchange.market(symbol)  # type: ignore[no-untyped-call,assignment]
            limits = market.get("limits", {})
            cost_limits = limits.get("cost", {})
            min_val = cost_limits.get("min")
            if min_val is not None:
                return float(min_val)
        except Exception:
            logger.debug("Failed to get MIN_NOTIONAL for {}, using default", symbol)
        return _default_min_notional

    def validate_min_notional(self, symbol: str, notional_usd: float) -> bool:
        """주문 금액이 MIN_NOTIONAL 이상인지 검증.

        Args:
            symbol: 거래 심볼
            notional_usd: 주문 금액 (USD)

        Returns:
            True면 통과, False면 미달
        """
        min_notional = self.get_min_notional(symbol)
        return notional_usd >= min_notional

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
                last_exc = e
                is_rate_limit = True
                if attempt < max_retries:
                    wait = base_backoff * (2 ** (attempt + 1))
                    wait *= 0.5 + random.random()
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
                    wait *= 0.5 + random.random()
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

        if last_exc is None:
            msg = "_retry_with_backoff: unexpected state — no exception recorded"
            raise RuntimeError(msg)
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
