"""Tests for Fear & Greed Index fetcher (Alternative.me)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.data.onchain.fetcher import FEAR_GREED_URL, OnchainFetcher


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


class TestFearGreedUrl:
    def test_fear_greed_url(self) -> None:
        assert "alternative.me" in FEAR_GREED_URL


# ---------------------------------------------------------------------------
# fetch_fear_greed
# ---------------------------------------------------------------------------


class TestFetchFearGreed:
    @pytest.mark.asyncio()
    async def test_success(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """정상 응답 — 2개 레코드 파싱."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "73",
                    "value_classification": "Greed",
                    "timestamp": "1704067200",
                },
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1704153600",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "value", "classification", "source"]

    @pytest.mark.asyncio()
    async def test_empty_response(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """빈 응답 → empty DataFrame with columns."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        assert df.empty
        assert list(df.columns) == ["timestamp", "value", "classification", "source"]

    @pytest.mark.asyncio()
    async def test_missing_data_key(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """data 키 없음 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "Fear and Greed Index"}
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        assert df.empty

    @pytest.mark.asyncio()
    async def test_value_types(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """value는 int, classification은 str."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "value": "50",
                    "value_classification": "Neutral",
                    "timestamp": "1704067200",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        assert df["value"].iloc[0] == 50
        assert df["value"].dtype.kind == "i"  # integer type (int64 등)
        assert df["classification"].iloc[0] == "Neutral"

    @pytest.mark.asyncio()
    async def test_timestamp_utc(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """timestamp는 UTC timezone-aware."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "value": "60",
                    "value_classification": "Greed",
                    "timestamp": "1704067200",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        ts = df["timestamp"].iloc[0]
        assert isinstance(ts, pd.Timestamp)
        assert str(ts.tz) == "UTC"

    @pytest.mark.asyncio()
    async def test_source_column(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """source 컬럼은 'alternative_me'."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "value": "40",
                    "value_classification": "Fear",
                    "timestamp": "1704067200",
                },
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        assert df["source"].iloc[0] == "alternative_me"

    @pytest.mark.asyncio()
    async def test_url_and_params(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """올바른 URL과 파라미터로 호출."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_fear_greed()

        called_url = mock_client.get.call_args[0][0]
        called_kwargs = mock_client.get.call_args[1]
        assert "alternative.me/fng" in called_url
        assert called_kwargs["params"]["limit"] == "0"

    @pytest.mark.asyncio()
    async def test_non_dict_response(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """dict가 아닌 응답 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = "unexpected"
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_fear_greed()

        assert df.empty
