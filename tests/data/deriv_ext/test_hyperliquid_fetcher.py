"""Tests for HyperliquidFetcher and route_fetch extension."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.deriv_ext.client import AsyncHyperliquidClient
from src.data.deriv_ext.fetcher import HyperliquidFetcher, route_fetch


class TestHyperliquidFetcher:
    """HyperliquidFetcher 테스트."""

    async def test_fetch_asset_contexts(self) -> None:
        """metaAndAssetCtxs 스냅샷."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {
                "universe": [
                    {"name": "BTC", "szDecimals": 5},
                    {"name": "ETH", "szDecimals": 4},
                    {"name": "SOL", "szDecimals": 2},
                ],
            },
            [
                {
                    "markPx": "95000.5",
                    "openInterest": "500000000",
                    "funding": "0.0001",
                    "premium": "0.002",
                    "dayNtlVlm": "5000000000",
                },
                {
                    "markPx": "3200.0",
                    "openInterest": "100000000",
                    "funding": "-0.0005",
                    "premium": "0.001",
                    "dayNtlVlm": "2000000000",
                },
                {
                    "markPx": "180.0",
                    "openInterest": "50000000",
                    "funding": "0.0003",
                    "premium": None,
                    "dayNtlVlm": "500000000",
                },
            ],
        ]

        client = AsyncMock(spec=AsyncHyperliquidClient)
        client.post = AsyncMock(return_value=mock_response)

        fetcher = HyperliquidFetcher(client)
        df = await fetcher.fetch_asset_contexts()

        # Only BTC and ETH (SOL filtered out)
        assert len(df) == 2
        assert set(df["coin"]) == {"BTC", "ETH"}
        assert "mark_price" in df.columns
        assert "open_interest" in df.columns
        assert "funding" in df.columns

    async def test_fetch_asset_contexts_empty(self) -> None:
        """빈 응답."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"unexpected": "format"}

        client = AsyncMock(spec=AsyncHyperliquidClient)
        client.post = AsyncMock(return_value=mock_response)

        fetcher = HyperliquidFetcher(client)
        df = await fetcher.fetch_asset_contexts()
        assert df.empty

    async def test_fetch_predicted_fundings(self) -> None:
        """predictedFundings 스냅샷."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {
                "coin": "BTC",
                "venues": [
                    ["Binance", "0.0002"],
                    ["Bybit", "0.00015"],
                    ["Hyperliquid", "0.0001"],
                ],
            },
            {
                "coin": "ETH",
                "venues": [
                    ["Binance", "-0.0001"],
                    ["Hyperliquid", "-0.00005"],
                ],
            },
            {
                "coin": "SOL",
                "venues": [
                    ["Binance", "0.0005"],
                ],
            },
        ]

        client = AsyncMock(spec=AsyncHyperliquidClient)
        client.post = AsyncMock(return_value=mock_response)

        fetcher = HyperliquidFetcher(client)
        df = await fetcher.fetch_predicted_fundings()

        # BTC: 3 venues + ETH: 2 venues = 5 rows (SOL filtered out)
        assert len(df) == 5
        assert set(df["coin"]) == {"BTC", "ETH"}
        assert "venue" in df.columns
        assert "predicted_funding" in df.columns

    async def test_fetch_predicted_fundings_empty(self) -> None:
        """빈 응답."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = "not a list"

        client = AsyncMock(spec=AsyncHyperliquidClient)
        client.post = AsyncMock(return_value=mock_response)

        fetcher = HyperliquidFetcher(client)
        df = await fetcher.fetch_predicted_fundings()
        assert df.empty


class TestRouteFetchHyperliquid:
    """route_fetch Hyperliquid 확장 테스트."""

    async def test_route_asset_contexts(self) -> None:
        """route_fetch hyperliquid/hl_asset_contexts."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"universe": [{"name": "BTC"}]},
            [
                {
                    "markPx": "95000.5",
                    "openInterest": "500000000",
                    "funding": "0.0001",
                    "premium": None,
                    "dayNtlVlm": "5000000000",
                }
            ],
        ]

        client = AsyncMock(spec=AsyncHyperliquidClient)
        client.post = AsyncMock(return_value=mock_response)

        fetcher = HyperliquidFetcher(client)
        df = await route_fetch(fetcher, "hyperliquid", "hl_asset_contexts")
        assert len(df) == 1

    async def test_route_predicted_fundings(self) -> None:
        """route_fetch hyperliquid/hl_predicted_fundings."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"coin": "BTC", "venues": [["Binance", "0.0002"]]},
        ]

        client = AsyncMock(spec=AsyncHyperliquidClient)
        client.post = AsyncMock(return_value=mock_response)

        fetcher = HyperliquidFetcher(client)
        df = await route_fetch(fetcher, "hyperliquid", "hl_predicted_fundings")
        assert len(df) == 1

    async def test_route_unknown_hyperliquid_name(self) -> None:
        """route_fetch hyperliquid/unknown → ValueError."""
        client = AsyncMock(spec=AsyncHyperliquidClient)
        fetcher = HyperliquidFetcher(client)

        with pytest.raises(ValueError, match="Unknown Hyperliquid dataset"):
            await route_fetch(fetcher, "hyperliquid", "unknown_dataset")

    async def test_route_unknown_source(self) -> None:
        """route_fetch unknown_source → ValueError."""
        client = AsyncMock(spec=AsyncHyperliquidClient)
        fetcher = HyperliquidFetcher(client)

        with pytest.raises(ValueError, match="Unknown deriv_ext source"):
            await route_fetch(fetcher, "unknown_source", "some_name")
