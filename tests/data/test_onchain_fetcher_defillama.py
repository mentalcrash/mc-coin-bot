"""Tests for src/data/onchain/fetcher.py — DeFiLlama stablecoin fetcher."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.onchain.fetcher import (
    DEFI_CHAINS,
    DEFILLAMA_BASE_URL,
    STABLECOIN_IDS,
    OnchainFetcher,
)


@pytest.fixture()
def mock_client() -> AsyncMock:
    """Mock AsyncOnchainClient."""
    return AsyncMock()


@pytest.fixture()
def fetcher(mock_client: AsyncMock) -> OnchainFetcher:
    """OnchainFetcher with mock client."""
    return OnchainFetcher(client=mock_client)


class TestConstants:
    def test_defi_chains(self) -> None:
        assert "Ethereum" in DEFI_CHAINS
        assert "Tron" in DEFI_CHAINS
        assert "BSC" in DEFI_CHAINS
        assert len(DEFI_CHAINS) == 5

    def test_stablecoin_ids(self) -> None:
        assert STABLECOIN_IDS["USDT"] == 1
        assert STABLECOIN_IDS["USDC"] == 2

    def test_base_url(self) -> None:
        assert "llama.fi" in DEFILLAMA_BASE_URL


class TestFetchStablecoinTotal:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "date": 1704067200,  # 2024-01-01
                "totalCirculating": {"peggedUSD": 130000000000},
            },
            {
                "date": 1704153600,  # 2024-01-02
                "totalCirculating": {"peggedUSD": 130500000000},
            },
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_total()

        assert len(df) == 2
        assert "date" in df.columns
        assert "total_circulating_usd" in df.columns
        assert "source" in df.columns
        assert df["source"].iloc[0] == "defillama"

    @pytest.mark.asyncio()
    async def test_empty_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_total()

        assert df.empty
        assert list(df.columns) == ["date", "total_circulating_usd", "source"]

    @pytest.mark.asyncio()
    async def test_url(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        await fetcher.fetch_stablecoin_total()

        called_url = mock_client.get.call_args[0][0]
        assert "/stablecoincharts/all" in called_url

    @pytest.mark.asyncio()
    async def test_decimal_precision(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "date": 1704067200,
                "totalCirculating": {"peggedUSD": 130123456789.12},
            },
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_total()
        val = df["total_circulating_usd"].iloc[0]
        assert isinstance(val, Decimal)


class TestFetchStablecoinByChain:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "date": 1704067200,
                "totalCirculating": {"peggedUSD": 50000000000},
                "totalMinted": {"peggedUSD": 51000000000},
            },
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_by_chain("Ethereum")

        assert len(df) == 1
        assert df["chain"].iloc[0] == "Ethereum"
        assert "total_circulating_usd" in df.columns
        assert "total_minted_usd" in df.columns

    @pytest.mark.asyncio()
    async def test_empty_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_by_chain("Ethereum")

        assert df.empty
        assert "chain" in df.columns

    @pytest.mark.asyncio()
    async def test_url_contains_chain(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        await fetcher.fetch_stablecoin_by_chain("Tron")

        called_url = mock_client.get.call_args[0][0]
        assert "/stablecoincharts/Tron" in called_url

    @pytest.mark.asyncio()
    async def test_no_minted_data(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "date": 1704067200,
                "totalCirculating": {"peggedUSD": 50000000000},
                "totalMinted": {},
            },
        ]
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_by_chain("Ethereum")
        assert df["total_minted_usd"].iloc[0] is None


class TestFetchStablecoinIndividual:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "chainBalances": {
                "Ethereum": {
                    "tokens": [
                        {
                            "date": 1704067200,
                            "circulating": {"peggedUSD": 40000000000},
                        },
                    ]
                },
                "Tron": {
                    "tokens": [
                        {
                            "date": 1704067200,
                            "circulating": {"peggedUSD": 30000000000},
                        },
                    ]
                },
            }
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_individual(1, "USDT")

        assert len(df) == 1  # Aggregated across chains
        assert df["stablecoin_id"].iloc[0] == 1
        assert df["name"].iloc[0] == "USDT"
        # 40B + 30B = 70B
        assert df["circulating_usd"].iloc[0] == Decimal(70000000000)

    @pytest.mark.asyncio()
    async def test_empty_chain_balances(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"chainBalances": {}}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_individual(1, "USDT")

        assert df.empty
        assert "circulating_usd" in df.columns

    @pytest.mark.asyncio()
    async def test_url_contains_id(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"chainBalances": {}}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_stablecoin_individual(2, "USDC")

        called_url = mock_client.get.call_args[0][0]
        assert "/stablecoin/2" in called_url

    @pytest.mark.asyncio()
    async def test_cross_chain_aggregation_multiple_dates(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "chainBalances": {
                "Ethereum": {
                    "tokens": [
                        {"date": 1704067200, "circulating": {"peggedUSD": 10000}},
                        {"date": 1704153600, "circulating": {"peggedUSD": 11000}},
                    ]
                },
                "Tron": {
                    "tokens": [
                        {"date": 1704067200, "circulating": {"peggedUSD": 5000}},
                        {"date": 1704153600, "circulating": {"peggedUSD": 6000}},
                    ]
                },
            }
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_stablecoin_individual(1, "USDT")

        assert len(df) == 2
        # Date 1: 10000 + 5000 = 15000
        assert df["circulating_usd"].iloc[0] == Decimal(15000)
        # Date 2: 11000 + 6000 = 17000
        assert df["circulating_usd"].iloc[1] == Decimal(17000)
