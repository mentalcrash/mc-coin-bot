"""Tests for CoinGecko fetch methods and route_fetch extension."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.macro.client import AsyncCoinGeckoClient, AsyncMacroClient
from src.data.macro.fetcher import MacroFetcher, route_fetch


class TestCoinGeckoFetch:
    """CoinGecko fetch 메서드 테스트."""

    def _make_fetcher(self, cg_client: AsyncCoinGeckoClient) -> MacroFetcher:
        """테스트용 MacroFetcher 생성."""
        fred_client = MagicMock(spec=AsyncMacroClient)
        return MacroFetcher(fred_client, api_key="", coingecko_client=cg_client)

    async def test_fetch_coingecko_global(self) -> None:
        """CoinGecko global 스냅샷."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": {
                "active_cryptocurrencies": 15000,
                "market_cap_percentage": {"btc": 58.5, "eth": 12.3},
                "total_market_cap": {"usd": 2500000000000},
                "total_volume": {"usd": 150000000000},
            }
        }

        cg_client = AsyncMock(spec=AsyncCoinGeckoClient)
        cg_client.get = AsyncMock(return_value=mock_response)

        fetcher = self._make_fetcher(cg_client)
        df = await fetcher.fetch_coingecko_global()

        assert len(df) == 1
        assert "btc_dominance" in df.columns
        assert "total_market_cap_usd" in df.columns
        assert "source" in df.columns
        assert df["source"].iloc[0] == "coingecko"

    async def test_fetch_coingecko_global_empty(self) -> None:
        """CoinGecko global 빈 응답."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {}}

        cg_client = AsyncMock(spec=AsyncCoinGeckoClient)
        cg_client.get = AsyncMock(return_value=mock_response)

        fetcher = self._make_fetcher(cg_client)
        df = await fetcher.fetch_coingecko_global()
        assert df.empty

    async def test_fetch_coingecko_defi(self) -> None:
        """CoinGecko DeFi 스냅샷."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": {
                "defi_market_cap": "150000000000",
                "defi_to_eth_ratio": "2.5",
                "defi_dominance": "4.2",
            }
        }

        cg_client = AsyncMock(spec=AsyncCoinGeckoClient)
        cg_client.get = AsyncMock(return_value=mock_response)

        fetcher = self._make_fetcher(cg_client)
        df = await fetcher.fetch_coingecko_defi()

        assert len(df) == 1
        assert "defi_dominance" in df.columns

    async def test_fetch_coingecko_without_client(self) -> None:
        """CoinGecko client 미설정 시 RuntimeError."""
        fred_client = MagicMock(spec=AsyncMacroClient)
        fetcher = MacroFetcher(fred_client, api_key="")
        with pytest.raises(RuntimeError, match="CoinGecko client not configured"):
            await fetcher.fetch_coingecko_global()


class TestRouteFetchCoinGecko:
    """route_fetch CoinGecko 확장 테스트."""

    async def test_route_global_metrics(self) -> None:
        """route_fetch coingecko/global_metrics."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": {
                "active_cryptocurrencies": 15000,
                "market_cap_percentage": {"btc": 58.5, "eth": 12.3},
                "total_market_cap": {"usd": 2500000000000},
                "total_volume": {"usd": 150000000000},
            }
        }

        cg_client = AsyncMock(spec=AsyncCoinGeckoClient)
        cg_client.get = AsyncMock(return_value=mock_response)

        fred_client = MagicMock(spec=AsyncMacroClient)
        fetcher = MacroFetcher(fred_client, api_key="", coingecko_client=cg_client)

        df = await route_fetch(fetcher, "coingecko", "global_metrics")
        assert len(df) == 1
        assert "btc_dominance" in df.columns

    async def test_route_defi_global(self) -> None:
        """route_fetch coingecko/defi_global."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": {
                "defi_market_cap": "150000000000",
                "defi_to_eth_ratio": "2.5",
                "defi_dominance": "4.2",
            }
        }

        cg_client = AsyncMock(spec=AsyncCoinGeckoClient)
        cg_client.get = AsyncMock(return_value=mock_response)

        fred_client = MagicMock(spec=AsyncMacroClient)
        fetcher = MacroFetcher(fred_client, api_key="", coingecko_client=cg_client)

        df = await route_fetch(fetcher, "coingecko", "defi_global")
        assert len(df) == 1

    async def test_route_unknown_coingecko_name(self) -> None:
        """route_fetch coingecko/unknown → ValueError."""
        cg_client = AsyncMock(spec=AsyncCoinGeckoClient)
        fred_client = MagicMock(spec=AsyncMacroClient)
        fetcher = MacroFetcher(fred_client, api_key="", coingecko_client=cg_client)

        with pytest.raises(ValueError, match="Unknown CoinGecko dataset"):
            await route_fetch(fetcher, "coingecko", "unknown_dataset")
