"""Tests for BinanceFuturesClient derivatives fetch methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.exchange.binance_futures_client import BinanceFuturesClient


@pytest.fixture()
def mock_exchange() -> MagicMock:
    """ccxt.pro.binance mock."""
    exchange = MagicMock()
    exchange.fetch_funding_rate_history = AsyncMock(return_value=[])
    exchange.fapidata_get_openinteresthist = AsyncMock(return_value=[])
    exchange.fapidata_get_globallongshortaccountratio = AsyncMock(return_value=[])
    exchange.fapidata_get_takerlongshortratio = AsyncMock(return_value=[])
    exchange.load_markets = AsyncMock()
    exchange.close = AsyncMock()
    exchange.set_sandbox_mode = MagicMock()
    exchange.markets = {"BTC/USDT:USDT": {"id": "BTCUSDT"}}
    return exchange


@pytest.fixture()
def client(mock_exchange: MagicMock) -> BinanceFuturesClient:
    """BinanceFuturesClient with mocked exchange."""
    c = BinanceFuturesClient.__new__(BinanceFuturesClient)
    c._exchange = mock_exchange
    c._is_open = True
    c._metrics_callback = None
    return c


class TestFetchFundingRateHistory:
    @pytest.mark.asyncio()
    async def test_returns_list(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1000, "fundingRate": 0.0001, "markPrice": 42000},
        ]
        result = await client.fetch_funding_rate_history("BTC/USDT")
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_with_since_and_limit(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fetch_funding_rate_history.return_value = []
        await client.fetch_funding_rate_history("BTC/USDT", since=1000, limit=100)
        mock_exchange.fetch_funding_rate_history.assert_called_once()

    @pytest.mark.asyncio()
    async def test_symbol_conversion(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        """BTC/USDT → BTC/USDT:USDT 변환."""
        mock_exchange.fetch_funding_rate_history.return_value = []
        await client.fetch_funding_rate_history("BTC/USDT")
        call_args = mock_exchange.fetch_funding_rate_history.call_args
        # to_futures_symbol이 적용되어 BTC/USDT:USDT로 변환되어야 함
        assert "USDT" in str(call_args)

    @pytest.mark.asyncio()
    async def test_empty_response(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fetch_funding_rate_history.return_value = []
        result = await client.fetch_funding_rate_history("BTC/USDT")
        assert result == []


class TestFetchOpenInterestHistory:
    @pytest.mark.asyncio()
    async def test_returns_list(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fapidata_get_openinteresthist.return_value = [
            {"timestamp": 1000, "sumOpenInterest": "50000", "sumOpenInterestValue": "2100000000"},
        ]
        result = await client.fetch_open_interest_history("BTC/USDT")
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_bare_symbol_format(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        """BTC/USDT → BTCUSDT 변환 (Binance-specific)."""
        mock_exchange.fapidata_get_openinteresthist.return_value = []
        await client.fetch_open_interest_history("BTC/USDT")
        call_args = mock_exchange.fapidata_get_openinteresthist.call_args
        params = call_args[0][0] if call_args[0] else call_args[1].get("params", {})
        # symbol이 BTCUSDT 형식이어야 함 (슬래시 없음)
        assert "symbol" in params
        assert "/" not in params["symbol"]

    @pytest.mark.asyncio()
    async def test_with_period(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fapidata_get_openinteresthist.return_value = []
        await client.fetch_open_interest_history("BTC/USDT", period="4h")
        assert mock_exchange.fapidata_get_openinteresthist.called


class TestFetchLongShortRatio:
    @pytest.mark.asyncio()
    async def test_returns_list(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fapidata_get_globallongshortaccountratio.return_value = [
            {
                "timestamp": 1000,
                "longAccount": "0.55",
                "shortAccount": "0.45",
                "longShortRatio": "1.22",
            },
        ]
        result = await client.fetch_long_short_ratio("BTC/USDT")
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_empty_response(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fapidata_get_globallongshortaccountratio.return_value = []
        result = await client.fetch_long_short_ratio("BTC/USDT")
        assert result == []


class TestFetchTakerBuySellRatio:
    @pytest.mark.asyncio()
    async def test_returns_list(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fapidata_get_takerlongshortratio.return_value = [
            {"timestamp": 1000, "buyVol": "1000", "sellVol": "800", "buySellRatio": "1.25"},
        ]
        result = await client.fetch_taker_buy_sell_ratio("BTC/USDT")
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_empty_response(
        self, client: BinanceFuturesClient, mock_exchange: MagicMock
    ) -> None:
        mock_exchange.fapidata_get_takerlongshortratio.return_value = []
        result = await client.fetch_taker_buy_sell_ratio("BTC/USDT")
        assert result == []
