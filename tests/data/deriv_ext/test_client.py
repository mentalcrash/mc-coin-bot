"""Tests for Coinalyze client (AsyncCoinalyzeClient)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.deriv_ext.client import AsyncCoinalyzeClient


class TestAsyncCoinalyzeClient:
    """AsyncCoinalyzeClient 테스트."""

    async def test_context_manager(self) -> None:
        """async with 컨텍스트 매니저."""
        async with AsyncCoinalyzeClient("coinalyze", api_key="test_key") as client:
            assert client._client is not None
        assert client._client is None

    async def test_get_without_context(self) -> None:
        """컨텍스트 매니저 없이 호출 시 RuntimeError."""
        client = AsyncCoinalyzeClient("coinalyze", api_key="test_key")
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.get("open-interest-history")

    async def test_get_success(self) -> None:
        """성공적인 GET 요청."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async with AsyncCoinalyzeClient("coinalyze", api_key="test_key") as client:
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)
            client._client.aclose = AsyncMock()

            result = await client.get("open-interest-history")
            assert result == mock_response

    async def test_rate_limiter_interval(self) -> None:
        """Rate limiter 간격 확인."""
        client = AsyncCoinalyzeClient("coinalyze", api_key="test_key")
        # 40 RPM → 1.5초 간격
        assert abs(client._rate_limiter.interval - 1.5) < 0.01

    async def test_default_source(self) -> None:
        """기본 source는 coinalyze."""
        client = AsyncCoinalyzeClient()
        assert client._source == "coinalyze"

    async def test_api_key_in_headers(self) -> None:
        """API key가 헤더에 포함되는지 확인."""
        async with AsyncCoinalyzeClient("coinalyze", api_key="my_api_key") as client:
            assert client._client is not None
            assert client._client.headers.get("api_key") == "my_api_key"
