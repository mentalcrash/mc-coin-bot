"""Rate-limited async HTTP client for on-chain data APIs.

Wraps httpx.AsyncClient with per-source rate limiting and retry logic.

Rules Applied:
    - #23 Exception Handling: Domain-driven hierarchy
    - #19 Git Security: No secrets in code
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from src.core.exceptions import NetworkError, RateLimitError

# Per-source rate limits (requests per minute)
SOURCE_RATE_LIMITS: dict[str, int] = {
    "defillama": 25,
    "coinmetrics": 90,
    "glassnode": 10,
}

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_TIMEOUT = 30.0

# HTTP status codes
HTTP_TOO_MANY_REQUESTS = 429


class RateLimiter:
    """Interval-based rate limiter using asyncio.Lock.

    Ensures minimum interval between requests to a given source.
    """

    def __init__(self, requests_per_minute: int) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute.
        """
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


class AsyncOnchainClient:
    """Rate-limited async HTTP client for on-chain APIs.

    Uses httpx.AsyncClient with HTTP/2 support, per-source rate limiting,
    and exponential backoff retry.

    Example:
        >>> async with AsyncOnchainClient("defillama") as client:
        ...     resp = await client.get("https://stablecoins.llama.fi/stablecoincharts/all")
    """

    def __init__(
        self,
        source: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize client.

        Args:
            source: Data source name (for rate limit lookup).
            max_retries: Maximum retry attempts.
            backoff_base: Exponential backoff base.
            timeout: Request timeout in seconds.
        """
        self._source = source
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        rpm = SOURCE_RATE_LIMITS.get(source, 30)
        self._rate_limiter = RateLimiter(rpm)

    async def __aenter__(self) -> AsyncOnchainClient:
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
            RuntimeError: Client not initialized (use async with).
            NetworkError: Request failed after retries.
            RateLimitError: Rate limit exceeded after retries.
        """
        if self._client is None:
            msg = (
                "Client not initialized. Use 'async with AsyncOnchainClient(...)' context manager."
            )
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
                        f"Rate limited by {self._source} (429), retry {attempt + 1}/{self._max_retries} in {wait:.1f}s",
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
                    f"Timeout from {self._source}, retry {attempt + 1}/{self._max_retries} in {wait:.1f}s",
                )
                await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_exc = e
                wait = self._backoff_base ** (attempt + 1)
                logger.warning(
                    f"HTTP error from {self._source}: {e}, retry {attempt + 1}/{self._max_retries} in {wait:.1f}s",
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
