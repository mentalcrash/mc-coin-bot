"""Tests for Blockchain.com Charts fetcher."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.data.onchain.fetcher import BC_CHARTS, BLOCKCHAIN_API_URL, OnchainFetcher


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


class TestBlockchainConstants:
    def test_api_url(self) -> None:
        assert "blockchain.info" in BLOCKCHAIN_API_URL

    def test_charts_list(self) -> None:
        assert "hash-rate" in BC_CHARTS
        assert "miners-revenue" in BC_CHARTS
        assert "transaction-fees-usd" in BC_CHARTS
        assert len(BC_CHARTS) == 3


# ---------------------------------------------------------------------------
# fetch_blockchain_chart
# ---------------------------------------------------------------------------


class TestFetchBlockchainChart:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """정상 응답 — 2개 data point 파싱, 컬럼 구조."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [
                {"x": 1704067200, "y": 523000000000.0},
                {"x": 1704153600, "y": 530000000000.0},
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "chart_name", "value", "source"]

    @pytest.mark.asyncio()
    async def test_empty_values(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """빈 values → empty DataFrame with columns."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "values": []}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        assert df.empty
        assert list(df.columns) == ["timestamp", "chart_name", "value", "source"]

    @pytest.mark.asyncio()
    async def test_missing_values_key(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """values 키 없음 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        assert df.empty

    @pytest.mark.asyncio()
    async def test_value_is_decimal(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """value는 Decimal 타입."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [{"x": 1704067200, "y": 42.5}],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("miners-revenue")

        assert isinstance(df["value"].iloc[0], Decimal)

    @pytest.mark.asyncio()
    async def test_chart_name_column(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """chart_name 컬럼에 전달한 이름 기록."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [{"x": 1704067200, "y": 100}],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("transaction-fees-usd")

        assert df["chart_name"].iloc[0] == "transaction-fees-usd"

    @pytest.mark.asyncio()
    async def test_source_column(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """source 컬럼은 'blockchain_com' 고정."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [{"x": 1704067200, "y": 100}],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        assert df["source"].iloc[0] == "blockchain_com"

    @pytest.mark.asyncio()
    async def test_timestamp_utc(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """timestamp는 UTC timezone-aware."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [{"x": 1704067200, "y": 100}],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        ts = df["timestamp"].iloc[0]
        assert isinstance(ts, pd.Timestamp)
        assert str(ts.tz) == "UTC"

    @pytest.mark.asyncio()
    async def test_url_and_params(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """올바른 URL과 파라미터로 호출."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "values": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_blockchain_chart("hash-rate", timespan="2years")

        called_url = mock_client.get.call_args[0][0]
        called_kwargs = mock_client.get.call_args[1]
        assert called_url == f"{BLOCKCHAIN_API_URL}/hash-rate"
        assert called_kwargs["params"]["timespan"] == "2years"
        assert called_kwargs["params"]["format"] == "json"
        assert called_kwargs["params"]["sampled"] == "false"

    @pytest.mark.asyncio()
    async def test_non_dict_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """dict가 아닌 응답 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = "unexpected"
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        assert df.empty

    @pytest.mark.asyncio()
    async def test_skip_null_values(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """x 또는 y가 None인 entry skip."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [
                {"x": 1704067200, "y": 100},
                {"x": None, "y": 200},
                {"x": 1704153600, "y": None},
                {"x": 1704240000, "y": 300},
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        assert len(df) == 2
        assert df["value"].iloc[0] == Decimal(100)
        assert df["value"].iloc[1] == Decimal(300)

    @pytest.mark.asyncio()
    async def test_scientific_notation(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """과학적 표기법 (5.234e11) → Decimal 변환."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "values": [{"x": 1704067200, "y": 5.234e11}],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_blockchain_chart("hash-rate")

        val = df["value"].iloc[0]
        assert isinstance(val, Decimal)
        assert val == Decimal("523400000000.0")
