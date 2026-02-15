"""Tests for src/data/onchain/client.py — rate limiter and async HTTP client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.data.onchain.client import (
    DEFAULT_MAX_RETRIES,
    SOURCE_RATE_LIMITS,
    AsyncOnchainClient,
    RateLimiter,
)


class TestRateLimiter:
    def test_interval_calculation(self) -> None:
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.interval == pytest.approx(1.0)

    def test_interval_calculation_high_rpm(self) -> None:
        limiter = RateLimiter(requests_per_minute=120)
        assert limiter.interval == pytest.approx(0.5)

    @pytest.mark.asyncio()
    async def test_first_call_no_wait(self) -> None:
        limiter = RateLimiter(requests_per_minute=60)
        with patch("src.data.onchain.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_consecutive_calls_wait(self) -> None:
        limiter = RateLimiter(requests_per_minute=60)
        with patch("src.data.onchain.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            # Second call should trigger sleep
            await limiter.acquire()
            mock_sleep.assert_awaited_once()


class TestSourceRateLimits:
    def test_defillama_rate_limit(self) -> None:
        assert SOURCE_RATE_LIMITS["defillama"] == 25

    def test_coinmetrics_rate_limit(self) -> None:
        assert SOURCE_RATE_LIMITS["coinmetrics"] == 90


class TestAsyncOnchainClient:
    @pytest.mark.asyncio()
    async def test_context_manager(self) -> None:
        async with AsyncOnchainClient("defillama") as client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio()
    async def test_get_without_init_raises(self) -> None:
        client = AsyncOnchainClient("defillama")
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.get("https://example.com")

    @pytest.mark.asyncio()
    async def test_get_success(self) -> None:
        async with AsyncOnchainClient("defillama") as client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_httpx = AsyncMock()
            mock_httpx.get = AsyncMock(return_value=mock_response)
            client._client = mock_httpx

            with patch("src.data.onchain.client.asyncio.sleep", new_callable=AsyncMock):
                resp = await client.get("https://stablecoins.llama.fi/stablecoincharts/all")

            assert resp == mock_response
            mock_httpx.get.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_retry_on_timeout(self) -> None:
        async with AsyncOnchainClient("defillama", max_retries=2) as client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_httpx = AsyncMock()
            mock_httpx.get = AsyncMock(
                side_effect=[
                    httpx.ReadTimeout("timeout"),
                    mock_response,
                ]
            )
            client._client = mock_httpx

            with patch("src.data.onchain.client.asyncio.sleep", new_callable=AsyncMock):
                resp = await client.get("https://example.com/api")

            assert resp == mock_response
            assert mock_httpx.get.await_count == 2

    @pytest.mark.asyncio()
    async def test_retry_exhausted_raises_network_error(self) -> None:
        async with AsyncOnchainClient("defillama", max_retries=2) as client:
            mock_httpx = AsyncMock()
            mock_httpx.get = AsyncMock(
                side_effect=httpx.ReadTimeout("timeout"),
            )
            client._client = mock_httpx

            with (
                patch("src.data.onchain.client.asyncio.sleep", new_callable=AsyncMock),
                pytest.raises(Exception, match="failed after"),
            ):
                await client.get("https://example.com/api")

    @pytest.mark.asyncio()
    async def test_rate_limit_429_raises(self) -> None:
        async with AsyncOnchainClient("defillama", max_retries=2) as client:
            mock_resp_429 = MagicMock(spec=httpx.Response)
            mock_resp_429.status_code = 429

            mock_httpx = AsyncMock()
            mock_httpx.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "429 Too Many Requests",
                    request=MagicMock(),
                    response=mock_resp_429,
                ),
            )
            client._client = mock_httpx

            with (
                patch("src.data.onchain.client.asyncio.sleep", new_callable=AsyncMock),
                pytest.raises(Exception, match="Rate limit"),
            ):
                await client.get("https://example.com/api")

    def test_default_max_retries(self) -> None:
        assert DEFAULT_MAX_RETRIES == 3
