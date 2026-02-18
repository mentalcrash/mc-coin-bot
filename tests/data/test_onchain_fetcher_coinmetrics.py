"""Tests for Coin Metrics fetcher in src/data/onchain/fetcher.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.onchain.fetcher import (
    CM_ASSETS,
    CM_METRICS,
    CM_PAGE_SIZE,
    CM_RENAME_MAP,
    COINMETRICS_BASE_URL,
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


class TestCoinMetricsConstants:
    def test_base_url(self) -> None:
        assert COINMETRICS_BASE_URL == "https://community-api.coinmetrics.io/v4"

    def test_metrics_list(self) -> None:
        assert "CapMVRVCur" in CM_METRICS
        assert "CapMrktCurUSD" in CM_METRICS
        assert "AdrActCnt" in CM_METRICS
        assert len(CM_METRICS) == 12

    def test_assets(self) -> None:
        assert "btc" in CM_ASSETS
        assert "eth" in CM_ASSETS
        assert len(CM_ASSETS) == 2

    def test_page_size(self) -> None:
        assert CM_PAGE_SIZE == 10000


class TestFetchCoinMetrics:
    @pytest.mark.asyncio()
    async def test_success_single_page(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "time": "2024-01-01T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "2.3",
                    "CapMrktCurUSD": "400000000000",
                },
                {
                    "time": "2024-01-02T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "2.4",
                    "CapMrktCurUSD": "410000000000",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur", "CapMrktCurUSD"])

        assert len(df) == 2
        assert "time" in df.columns
        assert "asset" in df.columns
        assert "CapMVRVCur" in df.columns
        assert "CapMrktCurUSD" in df.columns
        assert df["asset"].iloc[0] == "btc"
        assert df["CapMVRVCur"].iloc[0] == Decimal("2.3")

    @pytest.mark.asyncio()
    async def test_decimal_precision(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "time": "2024-01-01T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "2.345678901234",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur"])

        val = df["CapMVRVCur"].iloc[0]
        assert isinstance(val, Decimal)
        assert val == Decimal("2.345678901234")

    @pytest.mark.asyncio()
    async def test_empty_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur", "CapMrktCurUSD"])

        assert df.empty
        assert list(df.columns) == ["time", "asset", "CapMVRVCur", "CapMrktCurUSD"]

    @pytest.mark.asyncio()
    async def test_missing_metric_value(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "time": "2024-01-01T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "2.3",
                    # CapMrktCurUSD missing entirely
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur", "CapMrktCurUSD"])

        assert df["CapMVRVCur"].iloc[0] == Decimal("2.3")
        assert df["CapMrktCurUSD"].iloc[0] is None

    @pytest.mark.asyncio()
    async def test_empty_string_metric(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "time": "2024-01-01T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur"])

        assert df["CapMVRVCur"].iloc[0] is None

    @pytest.mark.asyncio()
    async def test_pagination(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        page1_response = MagicMock()
        page1_response.json.return_value = {
            "data": [
                {
                    "time": "2024-01-01T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "2.3",
                },
            ],
            "next_page_url": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?next_page_token=abc",
        }

        page2_response = MagicMock()
        page2_response.json.return_value = {
            "data": [
                {
                    "time": "2024-01-02T00:00:00.000000000Z",
                    "asset": "btc",
                    "CapMVRVCur": "2.4",
                },
            ],
        }

        mock_client.get.side_effect = [page1_response, page2_response]

        df = await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur"])

        assert len(df) == 2
        assert mock_client.get.call_count == 2
        # Second call uses next_page_url directly
        second_call_url = mock_client.get.call_args_list[1][0][0]
        assert "next_page_token=abc" in second_call_url

    @pytest.mark.asyncio()
    async def test_url_construction(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_coinmetrics(
            "btc", metrics=["CapMVRVCur"], start="2024-01-01", end="2024-12-31"
        )

        called_url = mock_client.get.call_args[0][0]
        assert "/timeseries/asset-metrics" in called_url

        called_params = mock_client.get.call_args[1]["params"]
        assert called_params["assets"] == "btc"
        assert called_params["metrics"] == "CapMVRVCur"
        assert called_params["frequency"] == "1d"
        assert called_params["start_time"] == "2024-01-01"
        assert called_params["end_time"] == "2024-12-31"
        assert called_params["page_size"] == CM_PAGE_SIZE

    @pytest.mark.asyncio()
    async def test_default_metrics(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_coinmetrics("btc")

        called_params = mock_client.get.call_args[1]["params"]
        assert called_params["metrics"] == ",".join(CM_METRICS)

    @pytest.mark.asyncio()
    async def test_no_end_param_omitted(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_coinmetrics("btc", metrics=["CapMVRVCur"])

        called_params = mock_client.get.call_args[1]["params"]
        assert "end_time" not in called_params

    def test_rename_map(self) -> None:
        """CM_RENAME_MAP contains all metrics that need renaming."""
        assert CM_RENAME_MAP["CapMVRVCur"] == "oc_mvrv"
        assert CM_RENAME_MAP["CapMrktCurUSD"] == "oc_mktcap_usd"
        assert CM_RENAME_MAP["FlowInExUSD"] == "oc_flow_in_ex_usd"
        assert CM_RENAME_MAP["FlowOutExUSD"] == "oc_flow_out_ex_usd"
        assert len(CM_RENAME_MAP) == 10
        # All keys should be in CM_METRICS
        for key in CM_RENAME_MAP:
            assert key in CM_METRICS, f"{key} in CM_RENAME_MAP but not in CM_METRICS"
