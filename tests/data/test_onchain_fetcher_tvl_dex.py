"""Tests for DeFiLlama TVL + DEX volume fetchers."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.onchain.fetcher import DEFILLAMA_API_URL, OnchainFetcher


@pytest.fixture()
def mock_client() -> AsyncMock:
    """Mock AsyncOnchainClient."""
    return AsyncMock()


@pytest.fixture()
def fetcher(mock_client: AsyncMock) -> OnchainFetcher:
    """OnchainFetcher with mock client."""
    return OnchainFetcher(client=mock_client)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestDefillamaApiUrl:
    def test_defillama_api_url(self) -> None:
        assert "api.llama.fi" in DEFILLAMA_API_URL


# ---------------------------------------------------------------------------
# TVL
# ---------------------------------------------------------------------------


class TestFetchTvlTotal:
    @pytest.mark.asyncio()
    async def test_fetch_tvl_total_success(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"date": 1704067200, "tvl": 50000000000},
            {"date": 1704153600, "tvl": 51000000000},
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_tvl()

        assert len(df) == 2
        assert list(df.columns) == ["date", "chain", "tvl_usd"]
        assert df["chain"].iloc[0] == "all"
        assert df["chain"].iloc[1] == "all"

    @pytest.mark.asyncio()
    async def test_fetch_tvl_chain_success(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"date": 1704067200, "tvl": 30000000000},
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_tvl(chain="Ethereum")

        assert len(df) == 1
        assert df["chain"].iloc[0] == "Ethereum"

    @pytest.mark.asyncio()
    async def test_fetch_tvl_empty_response(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_tvl()

        assert df.empty
        assert list(df.columns) == ["date", "chain", "tvl_usd"]

    @pytest.mark.asyncio()
    async def test_fetch_tvl_url_format(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        await fetcher.fetch_tvl()

        called_url = mock_client.get.call_args[0][0]
        assert "/v2/historicalChainTvl" in called_url

    @pytest.mark.asyncio()
    async def test_fetch_tvl_url_with_chain(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        await fetcher.fetch_tvl(chain="Ethereum")

        called_url = mock_client.get.call_args[0][0]
        assert "/v2/historicalChainTvl/Ethereum" in called_url

    @pytest.mark.asyncio()
    async def test_fetch_tvl_decimal_precision(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"date": 1704067200, "tvl": 50123456789.12},
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_tvl()
        val = df["tvl_usd"].iloc[0]
        assert isinstance(val, Decimal)


# ---------------------------------------------------------------------------
# DEX Volume
# ---------------------------------------------------------------------------


class TestFetchDexVolume:
    @pytest.mark.asyncio()
    async def test_fetch_dex_volume_success(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "totalDataChart": [
                [1704067200, 5000000000],
                [1704153600, 5500000000],
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_dex_volume()

        assert len(df) == 2
        assert list(df.columns) == ["date", "volume_usd", "source"]
        assert df["source"].iloc[0] == "defillama"

    @pytest.mark.asyncio()
    async def test_fetch_dex_volume_empty_chart(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"totalDataChart": []}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_dex_volume()

        assert df.empty
        assert list(df.columns) == ["date", "volume_usd", "source"]

    @pytest.mark.asyncio()
    async def test_fetch_dex_volume_url(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"totalDataChart": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_dex_volume()

        called_url = mock_client.get.call_args[0][0]
        assert "/overview/dexs" in called_url

    @pytest.mark.asyncio()
    async def test_fetch_dex_volume_decimal_precision(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "totalDataChart": [
                [1704067200, 5123456789.99],
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_dex_volume()
        val = df["volume_usd"].iloc[0]
        assert isinstance(val, Decimal)

    @pytest.mark.asyncio()
    async def test_fetch_dex_volume_missing_key(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """Response without totalDataChart key → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"protocols": []}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_dex_volume()

        assert df.empty
        assert list(df.columns) == ["date", "volume_usd", "source"]
