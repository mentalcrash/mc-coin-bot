"""Tests for AsyncHyperliquidClient."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.deriv_ext.client import AsyncHyperliquidClient


class TestAsyncHyperliquidClient:
    """AsyncHyperliquidClient 테스트."""

    async def test_context_manager(self) -> None:
        """async with 컨텍스트 매니저."""
        async with AsyncHyperliquidClient() as client:
            assert client._client is not None
        assert client._client is None

    async def test_post_without_context(self) -> None:
        """컨텍스트 매니저 없이 호출 시 RuntimeError."""
        client = AsyncHyperliquidClient()
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.post("info", json={"type": "metaAndAssetCtxs"})

    async def test_post_success(self) -> None:
        """성공적인 POST 요청."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async with AsyncHyperliquidClient() as client:
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.post = AsyncMock(return_value=mock_response)
            client._client.aclose = AsyncMock()

            result = await client.post("info", json={"type": "metaAndAssetCtxs"})
            assert result == mock_response

    async def test_rate_limiter_interval(self) -> None:
        """Rate limiter 간격 확인 (60 RPM → 1.0초 간격)."""
        client = AsyncHyperliquidClient()
        assert abs(client._rate_limiter.interval - 1.0) < 0.01

    async def test_no_auth_headers(self) -> None:
        """인증 헤더 없음 확인."""
        async with AsyncHyperliquidClient() as client:
            assert client._client is not None
            # No auth headers should be set
            assert "api_key" not in client._client.headers
            assert "Authorization" not in client._client.headers

    async def test_base_url(self) -> None:
        """Base URL 확인."""
        async with AsyncHyperliquidClient() as client:
            assert client._client is not None
            assert str(client._client.base_url) == "https://api.hyperliquid.xyz/"
