"""Tests for macro data clients (AsyncMacroClient, YFinanceClient)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pandas as pd
import pytest

from src.data.macro.client import AsyncMacroClient, YFinanceClient


class TestAsyncMacroClient:
    """AsyncMacroClient 테스트."""

    async def test_context_manager(self) -> None:
        """async with 컨텍스트 매니저."""
        async with AsyncMacroClient("fred") as client:
            assert client._client is not None
        assert client._client is None

    async def test_get_without_context(self) -> None:
        """컨텍스트 매니저 없이 호출 시 RuntimeError."""
        client = AsyncMacroClient("fred")
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.get("https://example.com")

    async def test_get_success(self) -> None:
        """성공적인 GET 요청."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async with AsyncMacroClient("fred") as client:
            client._client = AsyncMock(spec=httpx.AsyncClient)
            client._client.get = AsyncMock(return_value=mock_response)
            client._client.aclose = AsyncMock()

            result = await client.get("https://api.stlouisfed.org/fred/series/observations")
            assert result == mock_response

    async def test_rate_limiter_interval(self) -> None:
        """Rate limiter 간격 확인."""
        client = AsyncMacroClient("fred")
        # 100 RPM → 0.6초 간격
        assert abs(client._rate_limiter.interval - 0.6) < 0.01


class TestYFinanceClient:
    """YFinanceClient 테스트."""

    async def test_fetch_ticker(self) -> None:
        """단일 티커 다운로드 (mock)."""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [103.0, 104.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.to_datetime(["2024-01-15", "2024-01-16"]),
        )

        client = YFinanceClient()
        with patch.object(client, "_download_sync", return_value=mock_df):
            result = await client.fetch_ticker("SPY", "2024-01-01", "2024-12-31")
            assert len(result) == 2
            assert "Close" in result.columns

    def test_download_sync_empty(self) -> None:
        """yfinance 빈 결과 처리."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = YFinanceClient._download_sync("INVALID", "2024-01-01", "2024-12-31")
            assert result.empty
