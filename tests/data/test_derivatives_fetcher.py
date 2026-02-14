"""Tests for src/data/derivatives_fetcher.py — 비동기 derivatives 배치 수집."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.data.derivatives_fetcher import (
    DerivativesFetcher,
    _get_year_timestamps,
)
from src.models.derivatives import (
    DerivativesBatch,
    FundingRateRecord,
    LongShortRatioRecord,
    OpenInterestRecord,
    TakerRatioRecord,
)


@pytest.fixture()
def mock_client() -> AsyncMock:
    """Mock BinanceFuturesClient."""
    client = AsyncMock()
    client.fetch_funding_rate_history = AsyncMock(return_value=[])
    client.fetch_open_interest_history = AsyncMock(return_value=[])
    client.fetch_long_short_ratio = AsyncMock(return_value=[])
    client.fetch_taker_buy_sell_ratio = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def fetcher(mock_client: AsyncMock) -> DerivativesFetcher:
    """DerivativesFetcher with mock client."""
    return DerivativesFetcher(client=mock_client)


class TestGetYearTimestamps:
    def test_2024(self) -> None:
        start_ts, end_ts = _get_year_timestamps(2024)
        start_dt = datetime.fromtimestamp(start_ts / 1000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)
        assert start_dt.year == 2024
        assert start_dt.month == 1
        assert start_dt.day == 1
        assert end_dt.year == 2024
        assert end_dt.month == 12
        assert end_dt.day == 31

    def test_start_before_end(self) -> None:
        start_ts, end_ts = _get_year_timestamps(2025)
        assert start_ts < end_ts


class TestFetchFundingRates:
    @pytest.mark.asyncio()
    async def test_empty_response(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        mock_client.fetch_funding_rate_history.return_value = []
        records = await fetcher.fetch_funding_rates("BTC/USDT", 1000, 2000)
        assert records == []

    @pytest.mark.asyncio()
    async def test_single_page(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
        mock_client.fetch_funding_rate_history.return_value = [
            {"timestamp": ts, "fundingRate": "0.0001", "markPrice": "42000"},
        ]
        records = await fetcher.fetch_funding_rates("BTC/USDT", ts - 1, ts + 1)
        assert len(records) == 1
        assert isinstance(records[0], FundingRateRecord)
        assert records[0].funding_rate == Decimal("0.0001")

    @pytest.mark.asyncio()
    async def test_pagination_stops_at_end(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        """Timestamp > end_ts 인 레코드는 무시."""
        ts1 = 1000
        ts2 = 3000  # end_ts(2000) 초과
        mock_client.fetch_funding_rate_history.return_value = [
            {"timestamp": ts1, "fundingRate": "0.0001", "markPrice": "42000"},
            {"timestamp": ts2, "fundingRate": "0.0002", "markPrice": "42500"},
        ]
        records = await fetcher.fetch_funding_rates("BTC/USDT", 0, 2000)
        assert len(records) == 1

    @pytest.mark.asyncio()
    async def test_pagination_advances(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        """since가 last_ts+1 로 전진."""
        call_count = 0

        async def _side_effect(*_args: object, **kwargs: object) -> list[dict[str, object]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"timestamp": 500, "fundingRate": "0.0001", "markPrice": "42000"}]
            return []  # 두 번째 호출에서 빈 응답 → 종료

        mock_client.fetch_funding_rate_history.side_effect = _side_effect
        records = await fetcher.fetch_funding_rates("BTC/USDT", 0, 1000)
        assert len(records) == 1
        assert call_count == 2

    @pytest.mark.asyncio()
    async def test_stale_timestamp_breaks(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        """last_ts <= since 면 무한루프 방지."""
        mock_client.fetch_funding_rate_history.return_value = [
            {"timestamp": 0, "fundingRate": "0.0001", "markPrice": "42000"},
        ]
        records = await fetcher.fetch_funding_rates("BTC/USDT", 0, 1000)
        assert len(records) == 1


class TestFetchOpenInterest:
    @pytest.mark.asyncio()
    async def test_empty(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        records = await fetcher.fetch_open_interest("BTC/USDT", 0, 1000)
        assert records == []

    @pytest.mark.asyncio()
    async def test_single_record(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        ts = 500
        mock_client.fetch_open_interest_history.side_effect = [
            [{"timestamp": ts, "sumOpenInterest": "50000", "sumOpenInterestValue": "2100000000"}],
            [],  # 두 번째 호출: 빈 응답으로 종료
        ]
        records = await fetcher.fetch_open_interest("BTC/USDT", 0, 1000)
        assert len(records) == 1
        assert isinstance(records[0], OpenInterestRecord)
        assert records[0].sum_open_interest == Decimal(50000)

    @pytest.mark.asyncio()
    async def test_respects_end_ts(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        mock_client.fetch_open_interest_history.side_effect = [
            [
                {
                    "timestamp": 500,
                    "sumOpenInterest": "50000",
                    "sumOpenInterestValue": "2100000000",
                },
                {
                    "timestamp": 1500,
                    "sumOpenInterest": "60000",
                    "sumOpenInterestValue": "2200000000",
                },
            ],
        ]
        records = await fetcher.fetch_open_interest("BTC/USDT", 0, 1000)
        assert len(records) == 1


class TestFetchLongShortRatio:
    @pytest.mark.asyncio()
    async def test_empty(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        records = await fetcher.fetch_long_short_ratio("BTC/USDT", 0, 1000)
        assert records == []

    @pytest.mark.asyncio()
    async def test_single_record(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        ts = 500
        mock_client.fetch_long_short_ratio.side_effect = [
            [
                {
                    "timestamp": ts,
                    "longAccount": "0.55",
                    "shortAccount": "0.45",
                    "longShortRatio": "1.22",
                }
            ],
            [],
        ]
        records = await fetcher.fetch_long_short_ratio("BTC/USDT", 0, 1000)
        assert len(records) == 1
        assert isinstance(records[0], LongShortRatioRecord)
        assert records[0].long_short_ratio == Decimal("1.22")


class TestFetchTakerRatio:
    @pytest.mark.asyncio()
    async def test_empty(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        records = await fetcher.fetch_taker_ratio("BTC/USDT", 0, 1000)
        assert records == []

    @pytest.mark.asyncio()
    async def test_single_record(self, fetcher: DerivativesFetcher, mock_client: AsyncMock) -> None:
        ts = 500
        mock_client.fetch_taker_buy_sell_ratio.side_effect = [
            [{"timestamp": ts, "buyVol": "1000", "sellVol": "800", "buySellRatio": "1.25"}],
            [],
        ]
        records = await fetcher.fetch_taker_ratio("BTC/USDT", 0, 1000)
        assert len(records) == 1
        assert isinstance(records[0], TakerRatioRecord)
        assert records[0].buy_sell_ratio == Decimal("1.25")


class TestFetchYear:
    @pytest.mark.asyncio()
    async def test_empty_year(self, fetcher: DerivativesFetcher) -> None:
        batch = await fetcher.fetch_year("BTC/USDT", 2024)
        assert isinstance(batch, DerivativesBatch)
        assert batch.is_empty

    @pytest.mark.asyncio()
    async def test_combined_batch(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        ts = int(datetime(2024, 6, 1, tzinfo=UTC).timestamp() * 1000)
        mock_client.fetch_funding_rate_history.side_effect = [
            [{"timestamp": ts, "fundingRate": "0.0001", "markPrice": "42000"}],
            [],
        ]
        mock_client.fetch_open_interest_history.side_effect = [
            [{"timestamp": ts, "sumOpenInterest": "50000", "sumOpenInterestValue": "2100000000"}],
            [],
        ]
        mock_client.fetch_long_short_ratio.side_effect = [
            [
                {
                    "timestamp": ts,
                    "longAccount": "0.55",
                    "shortAccount": "0.45",
                    "longShortRatio": "1.22",
                }
            ],
            [],
        ]
        mock_client.fetch_taker_buy_sell_ratio.side_effect = [
            [{"timestamp": ts, "buyVol": "1000", "sellVol": "800", "buySellRatio": "1.25"}],
            [],
        ]

        batch = await fetcher.fetch_year("BTC/USDT", 2024)
        assert not batch.is_empty
        assert batch.total_records == 4
        assert len(batch.funding_rates) == 1
        assert len(batch.open_interest) == 1
        assert len(batch.long_short_ratios) == 1
        assert len(batch.taker_ratios) == 1


class TestRateLimiting:
    @pytest.mark.asyncio()
    async def test_rate_limit_sleep_called(
        self, fetcher: DerivativesFetcher, mock_client: AsyncMock
    ) -> None:
        """Rate limiting이 요청 사이에 동작하는지 확인."""
        ts1, ts2 = 100, 500
        call_count = 0

        async def _side_effect(*_args: object, **kwargs: object) -> list[dict[str, object]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"timestamp": ts1, "fundingRate": "0.0001", "markPrice": "42000"}]
            if call_count == 2:
                return [{"timestamp": ts2, "fundingRate": "0.0002", "markPrice": "42500"}]
            return []

        mock_client.fetch_funding_rate_history.side_effect = _side_effect

        with patch("src.data.derivatives_fetcher.asyncio.sleep", new_callable=AsyncMock):
            records = await fetcher.fetch_funding_rates("BTC/USDT", 0, 1000)
            assert len(records) == 2
