"""Rate-limited async HTTP client for Deribit Public API v2.

AsyncOptionsClient: httpx + rate limiter + retry (exponential backoff)

Rules Applied:
    - #23 Exception Handling: Domain-driven hierarchy
    - #19 Git Security: No secrets in code (Deribit public = no auth)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

from src.core.exceptions import NetworkError, RateLimitError

# Per-source rate limits (requests per minute)
OPTIONS_RATE_LIMITS: dict[str, int] = {
    "deribit": 300,  # 보수적: 5 req/s (public endpoint pool 공유)
}

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_TIMEOUT = 30.0

# HTTP status codes
HTTP_TOO_MANY_REQUESTS = 429

# Deribit API base URL
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2/public/"


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


class AsyncOptionsClient:
    """Rate-limited async HTTP client for Deribit Public API.

    Uses httpx.AsyncClient with rate limiting and exponential backoff retry.
    No authentication required — Deribit public endpoints are free.

    Example:
        >>> async with AsyncOptionsClient("deribit") as client:
        ...     resp = await client.get("get_tradingview_chart_data", params=...)
    """

    def __init__(
        self,
        source: str = "deribit",
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

        rpm = OPTIONS_RATE_LIMITS.get(source, 60)
        self._rate_limiter = _RateLimiter(rpm)

    async def __aenter__(self) -> AsyncOptionsClient:
        """Enter async context: create httpx client."""
        self._client = httpx.AsyncClient(
            base_url=DERIBIT_BASE_URL,
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
            msg = (
                "Client not initialized. Use 'async with AsyncOptionsClient(...)' context manager."
            )
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
