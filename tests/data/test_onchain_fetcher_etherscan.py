"""Tests for Etherscan ETH Supply fetcher."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.data.onchain.fetcher import ETHERSCAN_API_URL, WEI_PER_ETH, OnchainFetcher


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


class TestEtherscanConstants:
    def test_api_url(self) -> None:
        assert "etherscan.io" in ETHERSCAN_API_URL

    def test_wei_per_eth(self) -> None:
        assert Decimal(1000000000000000000) == WEI_PER_ETH


# ---------------------------------------------------------------------------
# fetch_eth_supply
# ---------------------------------------------------------------------------


class TestFetchEthSupply:
    @pytest.mark.asyncio()
    async def test_success(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """정상 응답 — 1행 스냅샷, 컬럼 구조."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": "120000000000000000000000000",
                "Eth2Staking": "30000000000000000000000000",
                "BurntFees": "4000000000000000000000000",
                "WithdrawnTotal": "1000000000000000000000000",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-api-key")

        assert len(df) == 1
        assert list(df.columns) == [
            "timestamp",
            "eth_supply",
            "eth2_staking",
            "burnt_fees",
            "withdrawn_total",
            "source",
        ]

    @pytest.mark.asyncio()
    async def test_wei_to_eth_conversion(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """Wei → ETH 변환 정확도."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": "120000000000000000000000000",
                "Eth2Staking": "30000000000000000000000000",
                "BurntFees": "4000000000000000000000000",
                "WithdrawnTotal": "1000000000000000000000000",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-api-key")

        assert df["eth_supply"].iloc[0] == Decimal(120000000)
        assert df["eth2_staking"].iloc[0] == Decimal(30000000)
        assert df["burnt_fees"].iloc[0] == Decimal(4000000)
        assert df["withdrawn_total"].iloc[0] == Decimal(1000000)

    @pytest.mark.asyncio()
    async def test_empty_api_key(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """빈 API key → 빈 DataFrame, API 호출 없음."""
        df = await fetcher.fetch_eth_supply("")

        assert df.empty
        assert list(df.columns) == [
            "timestamp",
            "eth_supply",
            "eth2_staking",
            "burnt_fees",
            "withdrawn_total",
            "source",
        ]
        mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_api_error_status(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """status != '1' → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "0",
            "message": "NOTOK",
            "result": "Invalid API Key",
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("bad-key")

        assert df.empty

    @pytest.mark.asyncio()
    async def test_decimal_type(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """모든 금융 값은 Decimal 타입."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": "120000000000000000000000000",
                "Eth2Staking": "30000000000000000000000000",
                "BurntFees": "4000000000000000000000000",
                "WithdrawnTotal": "1000000000000000000000000",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        for col in ["eth_supply", "eth2_staking", "burnt_fees", "withdrawn_total"]:
            assert isinstance(df[col].iloc[0], Decimal), f"{col} should be Decimal"

    @pytest.mark.asyncio()
    async def test_source_column(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """source 컬럼은 'etherscan' 고정."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": "1000000000000000000",
                "Eth2Staking": "0",
                "BurntFees": "0",
                "WithdrawnTotal": "0",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        assert df["source"].iloc[0] == "etherscan"

    @pytest.mark.asyncio()
    async def test_timestamp_utc(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """timestamp는 UTC timezone-aware, 초 단위 floor."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": "1000000000000000000",
                "Eth2Staking": "0",
                "BurntFees": "0",
                "WithdrawnTotal": "0",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        ts = df["timestamp"].iloc[0]
        assert isinstance(ts, pd.Timestamp)
        assert str(ts.tz) == "UTC"
        # floor("s") 적용 확인 — 나노초 부분이 0
        assert ts.nanosecond == 0

    @pytest.mark.asyncio()
    async def test_url_and_params(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """올바른 URL과 파라미터로 호출."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "0", "message": "NOTOK", "result": "Error"}
        mock_client.get.return_value = mock_response

        await fetcher.fetch_eth_supply("my-api-key")

        called_url = mock_client.get.call_args[0][0]
        called_kwargs = mock_client.get.call_args[1]
        assert called_url == ETHERSCAN_API_URL
        assert called_kwargs["params"]["module"] == "stats"
        assert called_kwargs["params"]["action"] == "ethsupply2"
        assert called_kwargs["params"]["apikey"] == "my-api-key"

    @pytest.mark.asyncio()
    async def test_non_dict_response(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """dict가 아닌 응답 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = "unexpected string"
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        assert df.empty

    @pytest.mark.asyncio()
    async def test_non_dict_result(self, fetcher: OnchainFetcher, mock_client: AsyncMock) -> None:
        """result가 dict가 아닌 경우 (에러 메시지 등) → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": "Max rate limit reached",
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        assert df.empty

    @pytest.mark.asyncio()
    async def test_missing_eth_supply_key(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """EthSupply 키가 없으면 → empty DataFrame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "Eth2Staking": "30000000000000000000000000",
                "BurntFees": "4000000000000000000000000",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        assert df.empty

    @pytest.mark.asyncio()
    async def test_partial_fields_default_zero(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """일부 필드만 존재 시 나머지는 Decimal(0)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": "120000000000000000000000000",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        assert len(df) == 1
        assert df["eth_supply"].iloc[0] == Decimal(120000000)
        assert df["eth2_staking"].iloc[0] == Decimal(0)
        assert df["burnt_fees"].iloc[0] == Decimal(0)
        assert df["withdrawn_total"].iloc[0] == Decimal(0)

    @pytest.mark.asyncio()
    async def test_large_wei_precision(
        self, fetcher: OnchainFetcher, mock_client: AsyncMock
    ) -> None:
        """대량 Wei 값에서 Decimal 정밀도 유지."""
        # 120,220,572.123456789012345678 ETH in Wei
        wei_val = "120220572123456789012345678"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": {
                "EthSupply": wei_val,
                "Eth2Staking": "0",
                "BurntFees": "0",
                "WithdrawnTotal": "0",
            },
        }
        mock_client.get.return_value = mock_response

        df = await fetcher.fetch_eth_supply("test-key")

        expected = Decimal(wei_val) / WEI_PER_ETH
        assert df["eth_supply"].iloc[0] == expected
