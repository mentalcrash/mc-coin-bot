"""Tests for Deribit Options client (AsyncOptionsClient)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.options.client import AsyncOptionsClient


class TestAsyncOptionsClient:
    """AsyncOptionsClient 테스트."""

    async def test_context_manager(self) -> None:
        """async with 컨텍스트 매니저."""
        async with AsyncOptionsClient("deribit") as client:
            assert client._client is not None
        assert client._client is None

    async def test_get_without_context(self) -> None:
        """컨텍스트 매니저 없이 호출 시 RuntimeError."""
        client = AsyncOptionsClient("deribit")
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.get("get_tradingview_chart_data")

    async def test_get_success(self) -> None:
        """성공적인 GET 요청."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async with AsyncOptionsClient("deribit") as client:
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)
            client._client.aclose = AsyncMock()

            result = await client.get("get_tradingview_chart_data")
            assert result == mock_response

    async def test_rate_limiter_interval(self) -> None:
        """Rate limiter 간격 확인."""
        client = AsyncOptionsClient("deribit")
        # 300 RPM → 0.2초 간격
        assert abs(client._rate_limiter.interval - 0.2) < 0.01

    async def test_default_source(self) -> None:
        """기본 source는 deribit."""
        client = AsyncOptionsClient()
        assert client._source == "deribit"
