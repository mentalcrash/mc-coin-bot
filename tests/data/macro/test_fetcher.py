"""Tests for macro data fetcher (MacroFetcher + route_fetch)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pandas as pd
import pytest

from src.data.macro.client import AsyncMacroClient
from src.data.macro.fetcher import FRED_SERIES, YFINANCE_TICKERS, MacroFetcher, route_fetch


@pytest.fixture
def mock_fred_response() -> MagicMock:
    """FRED API JSON 응답 fixture."""
    data = {
        "observations": [
            {"date": "2024-01-15", "value": "123.45"},
            {"date": "2024-01-16", "value": "."},
            {"date": "2024-01-17", "value": "124.00"},
        ]
    }
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = data
    return resp


@pytest.fixture
def mock_client() -> AsyncMock:
    """Mock AsyncMacroClient."""
    client = AsyncMock(spec=AsyncMacroClient)
    return client


class TestMacroFetcher:
    """MacroFetcher 테스트."""

    async def test_fetch_fred_series_dxy(
        self, mock_client: AsyncMock, mock_fred_response: MagicMock
    ) -> None:
        """FRED DXY 데이터 가져오기."""
        mock_client.get.return_value = mock_fred_response
        fetcher = MacroFetcher(mock_client, api_key="test_key")

        df = await fetcher.fetch_fred_series("dxy")

        assert len(df) == 3
        assert "date" in df.columns
        assert "value" in df.columns
        assert "series_id" in df.columns
        assert df.iloc[0]["series_id"] == FRED_SERIES["dxy"]

    async def test_fred_dot_value_becomes_none(
        self, mock_client: AsyncMock, mock_fred_response: MagicMock
    ) -> None:
        """FRED "." 값이 None으로 변환."""
        mock_client.get.return_value = mock_fred_response
        fetcher = MacroFetcher(mock_client, api_key="test_key")

        df = await fetcher.fetch_fred_series("dxy")

        # "." → None → NaN in DataFrame
        assert pd.isna(df.iloc[1]["value"])

    async def test_fetch_fred_unknown_series(self, mock_client: AsyncMock) -> None:
        """알 수 없는 시리즈 이름."""
        fetcher = MacroFetcher(mock_client, api_key="test_key")
        with pytest.raises(ValueError, match="Unknown FRED series"):
            await fetcher.fetch_fred_series("unknown_series")

    async def test_fetch_fred_empty_response(self, mock_client: AsyncMock) -> None:
        """빈 FRED 응답."""
        resp = MagicMock(spec=httpx.Response)
        resp.json.return_value = {"observations": []}
        mock_client.get.return_value = resp

        fetcher = MacroFetcher(mock_client, api_key="test_key")
        df = await fetcher.fetch_fred_series("vix")
        assert df.empty

    async def test_fetch_yfinance_ticker(self, mock_client: AsyncMock) -> None:
        """yfinance 티커 가져오기."""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [103.0],
                "Volume": [1000000],
            },
            index=pd.to_datetime(["2024-01-15"]),
        )

        fetcher = MacroFetcher(mock_client, api_key="")
        with patch.object(fetcher._yf_client, "fetch_ticker", return_value=mock_df):
            df = await fetcher.fetch_yfinance_ticker("spy")

        assert len(df) == 1
        assert "close" in df.columns
        assert "ticker" in df.columns
        assert df.iloc[0]["ticker"] == "SPY"

    async def test_fetch_yfinance_unknown_ticker(self, mock_client: AsyncMock) -> None:
        """알 수 없는 yfinance 티커."""
        fetcher = MacroFetcher(mock_client, api_key="")
        with pytest.raises(ValueError, match="Unknown yfinance ticker"):
            await fetcher.fetch_yfinance_ticker("unknown_ticker")

    async def test_all_fred_series_defined(self) -> None:
        """모든 FRED 시리즈가 정의되어 있는지."""
        assert len(FRED_SERIES) == 6  # gold removed (FRED series discontinued 2022-01)
        assert "dxy" in FRED_SERIES
        assert "vix" in FRED_SERIES
        assert "m2" in FRED_SERIES

    async def test_all_yfinance_tickers_defined(self) -> None:
        """모든 yfinance 티커가 정의되어 있는지."""
        assert len(YFINANCE_TICKERS) == 6
        assert "spy" in YFINANCE_TICKERS
        assert "hyg" in YFINANCE_TICKERS


class TestRouteFetch:
    """route_fetch 함수 테스트."""

    async def test_route_fred(self, mock_client: AsyncMock, mock_fred_response: MagicMock) -> None:
        """FRED 라우팅."""
        mock_client.get.return_value = mock_fred_response
        fetcher = MacroFetcher(mock_client, api_key="test_key")

        df = await route_fetch(fetcher, "fred", "dxy")
        assert not df.empty

    async def test_route_yfinance(self, mock_client: AsyncMock) -> None:
        """yfinance 라우팅."""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [103.0],
                "Volume": [1000000],
            },
            index=pd.to_datetime(["2024-01-15"]),
        )
        fetcher = MacroFetcher(mock_client, api_key="")
        with patch.object(fetcher._yf_client, "fetch_ticker", return_value=mock_df):
            df = await route_fetch(fetcher, "yfinance", "spy")
        assert not df.empty

    async def test_route_unknown_source(self, mock_client: AsyncMock) -> None:
        """알 수 없는 source."""
        fetcher = MacroFetcher(mock_client, api_key="")
        with pytest.raises(ValueError, match="Unknown macro source"):
            await route_fetch(fetcher, "unknown", "test")
