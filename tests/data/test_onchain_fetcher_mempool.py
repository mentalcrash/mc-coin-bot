"""Tests for mempool.space mining data fetcher."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.data.onchain.fetcher import MEMPOOL_API_URL, OnchainFetcher


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


class TestMempoolConstants:
    def test_api_url(self) -> None:
        assert "mempool.space" in MEMPOOL_API_URL
        assert "/api/v1" in MEMPOOL_API_URL


# ---------------------------------------------------------------------------
# fetch_mempool_mining
# ---------------------------------------------------------------------------


class TestFetchMempoolMining:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """정상 응답 — hashrates + difficulty 파싱."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [
                {"timestamp": 1704067200, "avgHashrate": 5.23e20},
                {"timestamp": 1704153600, "avgHashrate": 5.30e20},
            ],
            "difficulty": [
                {"time": 1704067200, "height": 800000, "difficulty": 7.2e13, "adjustment": 0.05},
            ],
            "currentHashrate": 5.8e20,
            "currentDifficulty": 7.5e13,
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert len(df) == 2
        expected_cols = ["timestamp", "avg_hashrate", "difficulty", "block_height", "adjustment", "source"]
        assert list(df.columns) == expected_cols

    @pytest.mark.asyncio()
    async def test_empty_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """빈 hashrates/difficulty → empty DataFrame with columns."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [],
            "difficulty": [],
            "currentHashrate": 0,
            "currentDifficulty": 0,
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert df.empty
        assert "avg_hashrate" in df.columns

    @pytest.mark.asyncio()
    async def test_non_dict_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """dict가 아닌 응답 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = "unexpected"
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert df.empty

    @pytest.mark.asyncio()
    async def test_hashrate_is_decimal(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """avg_hashrate는 Decimal 타입."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [{"timestamp": 1704067200, "avgHashrate": 5.23e20}],
            "difficulty": [],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert isinstance(df["avg_hashrate"].iloc[0], Decimal)

    @pytest.mark.asyncio()
    async def test_difficulty_is_decimal(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """difficulty는 Decimal 타입."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [{"timestamp": 1704067200, "avgHashrate": 5.23e20}],
            "difficulty": [
                {"time": 1704067200, "height": 800000, "difficulty": 7.2e13, "adjustment": 0.05},
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert isinstance(df["difficulty"].iloc[0], Decimal)
        assert df["difficulty"].iloc[0] == Decimal(str(7.2e13))

    @pytest.mark.asyncio()
    async def test_timestamp_utc(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """timestamp는 UTC timezone-aware."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [{"timestamp": 1704067200, "avgHashrate": 100}],
            "difficulty": [],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        ts = df["timestamp"].iloc[0]
        assert isinstance(ts, pd.Timestamp)
        assert str(ts.tz) == "UTC"

    @pytest.mark.asyncio()
    async def test_source_column(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """source 컬럼은 'mempool_space' 고정."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [{"timestamp": 1704067200, "avgHashrate": 100}],
            "difficulty": [],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert df["source"].iloc[0] == "mempool_space"

    @pytest.mark.asyncio()
    async def test_url_with_interval(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """올바른 URL과 interval 파라미터로 호출."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"hashrates": [], "difficulty": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_mempool_mining(interval="1y")

        called_url = mock_client.get.call_args[0][0]
        assert called_url == f"{MEMPOOL_API_URL}/mining/hashrate/1y"

    @pytest.mark.asyncio()
    async def test_default_interval_all(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """기본 interval은 'all'."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"hashrates": [], "difficulty": []}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_mempool_mining()

        called_url = mock_client.get.call_args[0][0]
        assert called_url.endswith("/all")

    @pytest.mark.asyncio()
    async def test_difficulty_merge(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """difficulty 매칭 — 같은 timestamp 시 block_height/adjustment 포함."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [
                {"timestamp": 1704067200, "avgHashrate": 5.0e20},
                {"timestamp": 1704153600, "avgHashrate": 5.1e20},
            ],
            "difficulty": [
                {"time": 1704067200, "height": 800000, "difficulty": 7.2e13, "adjustment": 0.05},
            ],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        # 첫 번째 row: difficulty 매칭됨
        assert df["block_height"].iloc[0] == 800000
        assert df["adjustment"].iloc[0] == Decimal("0.05")

        # 두 번째 row: difficulty 매칭 없음 (pandas converts None → NaN)
        assert pd.isna(df["block_height"].iloc[1])
        assert pd.isna(df["adjustment"].iloc[1])

    @pytest.mark.asyncio()
    async def test_skip_null_timestamp(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """timestamp가 None인 hashrate entry skip."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [
                {"timestamp": None, "avgHashrate": 100},
                {"timestamp": 1704067200, "avgHashrate": 200},
            ],
            "difficulty": [],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert len(df) == 1
        assert df["avg_hashrate"].iloc[0] == Decimal(200)

    @pytest.mark.asyncio()
    async def test_hashrates_only(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """difficulty 없이 hashrates만 있는 경우."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashrates": [{"timestamp": 1704067200, "avgHashrate": 5.0e20}],
            "difficulty": [],
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_mempool_mining()

        assert len(df) == 1
        assert df["difficulty"].iloc[0] == Decimal(0)
        assert df["block_height"].iloc[0] is None
