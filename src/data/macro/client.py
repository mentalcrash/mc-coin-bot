"""Rate-limited async HTTP clients for macro data APIs.

AsyncMacroClient: FRED API (httpx + rate limiter + retry)
YFinanceClient: yfinance wrapper (asyncio.to_thread)

Rules Applied:
    - #23 Exception Handling: Domain-driven hierarchy
    - #19 Git Security: No secrets in code
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

if TYPE_CHECKING:
    import pandas as pd

from src.core.exceptions import NetworkError, RateLimitError

# Per-source rate limits (requests per minute)
MACRO_RATE_LIMITS: dict[str, int] = {
    "fred": 100,  # FRED ~120 req/min, conservative 100
    "coingecko": 30,  # CoinGecko Demo plan ~30 req/min
}

# CoinGecko API base URL
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/"

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_TIMEOUT = 30.0

# HTTP status codes
HTTP_TOO_MANY_REQUESTS = 429


class _RateLimiter:
    """Interval-based rate limiter using asyncio.Lock."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / requests_per_minute
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def interval(self) -> float:
        """Minimum interval between requests (seconds)."""
        return self._interval

    async def acquire(self) -> None:
        """Wait for rate limit slot."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                wait_time = self._interval - elapsed
                await asyncio.sleep(wait_time)
            self._last_request = time.monotonic()


class AsyncMacroClient:
    """Rate-limited async HTTP client for FRED API.

    Uses httpx.AsyncClient with rate limiting and exponential backoff retry.

    Example:
        >>> async with AsyncMacroClient("fred") as client:
        ...     resp = await client.get("https://api.stlouisfed.org/fred/series/observations", params=...)
    """

    def __init__(
        self,
        source: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._source = source
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        rpm = MACRO_RATE_LIMITS.get(source, 30)
        self._rate_limiter = _RateLimiter(rpm)

    async def __aenter__(self) -> AsyncMacroClient:
        """Enter async context: create httpx client."""
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context: close httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Send GET request with rate limiting and retry.

        Args:
            url: Request URL.
            **kwargs: Additional httpx request kwargs.

        Returns:
            httpx.Response object.

        Raises:
            RuntimeError: Client not initialized.
            NetworkError: Request failed after retries.
            RateLimitError: Rate limit exceeded after retries.
        """
        if self._client is None:
            msg = "Client not initialized. Use 'async with AsyncMacroClient(...)' context manager."
            raise RuntimeError(msg)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            await self._rate_limiter.acquire()
            try:
                response = await self._client.get(url, **kwargs)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == HTTP_TOO_MANY_REQUESTS:
                    last_exc = e
                    wait = self._backoff_base ** (attempt + 1)
                    logger.warning(
                        "Rate limited by {} (429), retry {}/{} in {:.1f}s",
                        self._source,
                        attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise NetworkError(
                    f"HTTP {e.response.status_code} from {self._source}: {url}",
                    context={"url": url, "status": e.response.status_code},
                ) from e
            except httpx.TimeoutException as e:
                last_exc = e
                wait = self._backoff_base ** (attempt + 1)
                logger.warning(
                    "Timeout from {}, retry {}/{} in {:.1f}s",
                    self._source,
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_exc = e
                wait = self._backoff_base ** (attempt + 1)
                logger.warning(
                    "HTTP error from {}: {}, retry {}/{} in {:.1f}s",
                    self._source,
                    e,
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                return response

        if (
            isinstance(last_exc, httpx.HTTPStatusError)
            and last_exc.response.status_code == HTTP_TOO_MANY_REQUESTS
        ):
            raise RateLimitError(
                f"Rate limit exceeded for {self._source} after {self._max_retries} retries",
                context={"source": self._source, "url": url},
            ) from last_exc

        raise NetworkError(
            f"Request to {self._source} failed after {self._max_retries} retries: {url}",
            context={"source": self._source, "url": url, "last_error": str(last_exc)},
        ) from last_exc


class AsyncCoinGeckoClient:
    """Rate-limited async HTTP client for CoinGecko API.

    Uses httpx.AsyncClient with rate limiting and exponential backoff retry.
    Auth via ``x-cg-demo-api-key`` header (CoinGecko Free Demo plan).

    Example:
        >>> async with AsyncCoinGeckoClient(api_key="demo-key") as client:
        ...     resp = await client.get("global")
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._source = "coingecko"
        self._api_key = api_key
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        rpm = MACRO_RATE_LIMITS.get("coingecko", 30)
        self._rate_limiter = _RateLimiter(rpm)

    async def __aenter__(self) -> AsyncCoinGeckoClient:
        """Enter async context: create httpx client."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key
        self._client = httpx.AsyncClient(
            base_url=COINGECKO_BASE_URL,
            http2=True,
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=True,
            headers=headers,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context: close httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, endpoint: str, **kwargs: Any) -> httpx.Response:
        """Send GET request with rate limiting and retry.

        Args:
            endpoint: API endpoint (relative to base_url).
            **kwargs: Additional httpx request kwargs.

        Returns:
            httpx.Response object.

        Raises:
            RuntimeError: Client not initialized.
            NetworkError: Request failed after retries.
            RateLimitError: Rate limit exceeded after retries.
        """
        if self._client is None:
            msg = "Client not initialized. Use 'async with AsyncCoinGeckoClient(...)' context manager."
            raise RuntimeError(msg)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            await self._rate_limiter.acquire()
            try:
                response = await self._client.get(endpoint, **kwargs)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == HTTP_TOO_MANY_REQUESTS:
                    last_exc = e
                    wait = self._backoff_base ** (attempt + 1)
                    logger.warning(
                        "Rate limited by {} (429), retry {}/{} in {:.1f}s",
                        self._source,
                        attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise NetworkError(
                    f"HTTP {e.response.status_code} from {self._source}: {endpoint}",
                    context={"endpoint": endpoint, "status": e.response.status_code},
                ) from e
            except httpx.TimeoutException as e:
                last_exc = e
                wait = self._backoff_base ** (attempt + 1)
                logger.warning(
                    "Timeout from {}, retry {}/{} in {:.1f}s",
                    self._source,
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_exc = e
                wait = self._backoff_base ** (attempt + 1)
                logger.warning(
                    "HTTP error from {}: {}, retry {}/{} in {:.1f}s",
                    self._source,
                    e,
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                return response

        if (
            isinstance(last_exc, httpx.HTTPStatusError)
            and last_exc.response.status_code == HTTP_TOO_MANY_REQUESTS
        ):
            raise RateLimitError(
                f"Rate limit exceeded for {self._source} after {self._max_retries} retries",
                context={"source": self._source, "endpoint": endpoint},
            ) from last_exc

        raise NetworkError(
            f"Request to {self._source} failed after {self._max_retries} retries: {endpoint}",
            context={
                "source": self._source,
                "endpoint": endpoint,
                "last_error": str(last_exc),
            },
        ) from last_exc


class YFinanceClient:
    """Async wrapper for yfinance downloads.

    yfinance는 인증 불필요, sync 라이브러리이므로 asyncio.to_thread로 래핑.

    Example:
        >>> client = YFinanceClient()
        >>> df = await client.fetch_ticker("SPY", "2020-01-01", "2025-12-31")
    """

    async def fetch_ticker(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """단일 티커 OHLCV 데이터 다운로드.

        Args:
            ticker: 티커 심볼 (e.g., "SPY")
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)

        Returns:
            OHLCV DataFrame (columns: Open, High, Low, Close, Volume)
        """
        return await asyncio.to_thread(self._download_sync, ticker, start, end)

    @staticmethod
    def _download_sync(ticker: str, start: str, end: str) -> pd.DataFrame:
        """Sync yfinance download (to_thread에서 호출)."""
        import pandas as _pd
        import yfinance as yf

        result = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if result is None:
            return _pd.DataFrame()
        return _pd.DataFrame(result)
