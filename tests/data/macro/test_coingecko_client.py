"""Tests for AsyncCoinGeckoClient."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.macro.client import AsyncCoinGeckoClient


class TestAsyncCoinGeckoClient:
    """AsyncCoinGeckoClient 테스트."""

    async def test_context_manager(self) -> None:
        """async with 컨텍스트 매니저."""
        async with AsyncCoinGeckoClient(api_key="demo") as client:
            assert client._client is not None
        assert client._client is None

    async def test_get_without_context(self) -> None:
        """컨텍스트 매니저 없이 호출 시 RuntimeError."""
        client = AsyncCoinGeckoClient(api_key="demo")
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.get("global")

    async def test_get_success(self) -> None:
        """성공적인 GET 요청."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async with AsyncCoinGeckoClient(api_key="demo") as client:
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)
            client._client.aclose = AsyncMock()

            result = await client.get("global")
            assert result == mock_response

    async def test_rate_limiter_interval(self) -> None:
        """Rate limiter 간격 확인 (30 RPM → 2.0초 간격)."""
        client = AsyncCoinGeckoClient(api_key="demo")
        assert abs(client._rate_limiter.interval - 2.0) < 0.01

    async def test_api_key_header(self) -> None:
        """x-cg-demo-api-key 헤더 설정 확인."""
        async with AsyncCoinGeckoClient(api_key="test-key-123") as client:
            assert client._client is not None
            assert client._client.headers.get("x-cg-demo-api-key") == "test-key-123"

    async def test_base_url(self) -> None:
        """Base URL 확인."""
        async with AsyncCoinGeckoClient(api_key="demo") as client:
            assert client._client is not None
            assert str(client._client.base_url) == "https://api.coingecko.com/api/v3/"
