"""Tests for beaconcha.in ETH staking APR fetcher."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.data.onchain.fetcher import BEACONCHA_API_URL, OnchainFetcher


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


class TestBeaconchaConstants:
    def test_api_url(self) -> None:
        assert "beaconcha.in" in BEACONCHA_API_URL
        assert "/api/v1" in BEACONCHA_API_URL


# ---------------------------------------------------------------------------
# fetch_eth_staking_apr
# ---------------------------------------------------------------------------


class TestFetchEthStakingApr:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """정상 응답 — avg_apr, cl_apr, el_apr 파싱."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "data": {
                "apr": 0.0278,
                "cl_apr": 0.0269,
                "el_apr": 0.0009,
                "day": 1908,
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        assert len(df) == 1
        expected_cols = ["timestamp", "avg_apr", "cl_apr", "el_apr", "source"]
        assert list(df.columns) == expected_cols

    @pytest.mark.asyncio()
    async def test_empty_non_ok_status(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """비-OK 상태 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ERROR", "data": None}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        assert df.empty
        assert "avg_apr" in df.columns

    @pytest.mark.asyncio()
    async def test_non_dict_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """dict가 아닌 응답 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = "unexpected"
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        assert df.empty

    @pytest.mark.asyncio()
    async def test_non_dict_data_field(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """data 필드가 dict가 아닌 경우 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OK", "data": "not_a_dict"}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        assert df.empty

    @pytest.mark.asyncio()
    async def test_apr_is_decimal(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """APR 값은 Decimal 타입."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "data": {"apr": 0.0278, "cl_apr": 0.0269, "el_apr": 0.0009},
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        assert isinstance(df["avg_apr"].iloc[0], Decimal)
        assert isinstance(df["cl_apr"].iloc[0], Decimal)
        assert isinstance(df["el_apr"].iloc[0], Decimal)

    @pytest.mark.asyncio()
    async def test_source_column(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """source 컬럼은 'beaconcha_in' 고정."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "data": {"apr": 0.03, "cl_apr": 0.028, "el_apr": 0.002},
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        assert df["source"].iloc[0] == "beaconcha_in"

    @pytest.mark.asyncio()
    async def test_timestamp_utc(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """timestamp는 UTC timezone-aware."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "data": {"apr": 0.03, "cl_apr": 0.028, "el_apr": 0.002},
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_staking_apr()

        ts = df["timestamp"].iloc[0]
        assert isinstance(ts, pd.Timestamp)
        assert str(ts.tz) == "UTC"
