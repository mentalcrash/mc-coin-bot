"""Tests for Coinalyze Extended Derivatives fetcher (CoinalyzeFetcher + route_fetch)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.deriv_ext.client import AsyncCoinalyzeClient
from src.data.deriv_ext.fetcher import COINALYZE_DATASETS, CoinalyzeFetcher, route_fetch


@pytest.fixture
def mock_client() -> AsyncMock:
    """Mock AsyncCoinalyzeClient."""
    return AsyncMock(spec=AsyncCoinalyzeClient)


def _make_response(data: dict | list) -> MagicMock:  # type: ignore[type-arg]
    """Mock httpx.Response with JSON data."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = data
    return resp


class TestCoinalyzeFetcher:
    """CoinalyzeFetcher 테스트."""

    async def test_fetch_agg_oi(self, mock_client: AsyncMock) -> None:
        """Aggregated OI 데이터 가져오기."""
        oi_data = [
            {
                "symbol": "BTCUSDT.6",
                "history": [
                    {"t": 1705276800, "o": 50000, "h": 52000, "l": 49000, "c": 51000},
                    {"t": 1705363200, "o": 51000, "h": 53000, "l": 50000, "c": 52000},
                ],
            }
        ]
        mock_client.get.return_value = _make_response(oi_data)
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_agg_oi("BTC", start="2024-01-15", end="2024-01-16")

        assert len(df) == 2
        assert "date" in df.columns
        assert "close" in df.columns
        assert "symbol" in df.columns
        assert df.iloc[0]["symbol"] == "BTCUSDT.6"

    async def test_fetch_agg_oi_empty(self, mock_client: AsyncMock) -> None:
        """빈 OI 응답."""
        mock_client.get.return_value = _make_response([])
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_agg_oi("BTC")
        assert df.empty

    async def test_fetch_agg_funding(self, mock_client: AsyncMock) -> None:
        """Aggregated Funding Rate 가져오기."""
        funding_data = [
            {
                "symbol": "ETHUSDT.6",
                "history": [
                    {"t": 1705276800, "o": 0.0001, "h": 0.0003, "l": -0.0001, "c": 0.0002},
                ],
            }
        ]
        mock_client.get.return_value = _make_response(funding_data)
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_agg_funding("ETH", start="2024-01-15", end="2024-01-16")

        assert len(df) == 1
        assert "close" in df.columns
        assert df.iloc[0]["symbol"] == "ETHUSDT.6"

    async def test_fetch_agg_funding_empty(self, mock_client: AsyncMock) -> None:
        """빈 Funding 응답."""
        mock_client.get.return_value = _make_response([])
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_agg_funding("BTC")
        assert df.empty

    async def test_fetch_liquidations(self, mock_client: AsyncMock) -> None:
        """Liquidation 데이터 가져오기."""
        liq_data = [
            {
                "symbol": "BTCUSDT.6",
                "history": [
                    {"t": 1705276800, "l": 1500000, "s": 800000},
                    {"t": 1705280400, "l": 200000, "s": 300000},
                ],
            }
        ]
        mock_client.get.return_value = _make_response(liq_data)
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_liquidations("BTC", start="2024-01-15", end="2024-01-16")

        assert len(df) == 2
        assert "long_volume" in df.columns
        assert "short_volume" in df.columns

    async def test_fetch_liquidations_empty(self, mock_client: AsyncMock) -> None:
        """빈 Liquidation 응답."""
        mock_client.get.return_value = _make_response([])
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_liquidations("BTC")
        assert df.empty

    async def test_fetch_cvd(self, mock_client: AsyncMock) -> None:
        """CVD 데이터 가져오기."""
        cvd_data = [
            {
                "symbol": "BTCUSDT.6",
                "history": [
                    {
                        "t": 1705276800,
                        "o": 100,
                        "h": 200,
                        "l": 50,
                        "c": 180,
                        "v": 50000,
                        "bv": 28000,
                    },
                ],
            }
        ]
        mock_client.get.return_value = _make_response(cvd_data)
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_cvd("BTC", start="2024-01-15", end="2024-01-16")

        assert len(df) == 1
        assert "buy_volume" in df.columns
        assert "volume" in df.columns

    async def test_fetch_cvd_empty(self, mock_client: AsyncMock) -> None:
        """빈 CVD 응답."""
        mock_client.get.return_value = _make_response([])
        fetcher = CoinalyzeFetcher(mock_client)

        df = await fetcher.fetch_cvd("BTC")
        assert df.empty

    async def test_all_datasets_defined(self) -> None:
        """모든 Coinalyze 데이터셋이 정의되어 있는지."""
        assert len(COINALYZE_DATASETS) == 8
        assert "btc_agg_oi" in COINALYZE_DATASETS
        assert "eth_agg_oi" in COINALYZE_DATASETS
        assert "btc_agg_funding" in COINALYZE_DATASETS
        assert "eth_agg_funding" in COINALYZE_DATASETS
        assert "btc_liquidations" in COINALYZE_DATASETS
        assert "eth_liquidations" in COINALYZE_DATASETS
        assert "btc_cvd" in COINALYZE_DATASETS
        assert "eth_cvd" in COINALYZE_DATASETS


class TestRouteFetch:
    """route_fetch 함수 테스트."""

    async def test_route_agg_oi(self, mock_client: AsyncMock) -> None:
        """OI 라우팅."""
        oi_data = [
            {
                "symbol": "BTCUSDT.6",
                "history": [{"t": 1705276800, "o": 1, "h": 1, "l": 1, "c": 1}],
            }
        ]
        mock_client.get.return_value = _make_response(oi_data)
        fetcher = CoinalyzeFetcher(mock_client)

        df = await route_fetch(fetcher, "coinalyze", "btc_agg_oi")
        assert not df.empty

    async def test_route_unknown_source(self, mock_client: AsyncMock) -> None:
        """알 수 없는 source."""
        fetcher = CoinalyzeFetcher(mock_client)
        with pytest.raises(ValueError, match="Unknown deriv_ext source"):
            await route_fetch(fetcher, "unknown", "btc_agg_oi")

    async def test_route_unknown_dataset(self, mock_client: AsyncMock) -> None:
        """알 수 없는 dataset."""
        fetcher = CoinalyzeFetcher(mock_client)
        with pytest.raises(ValueError, match="Unknown deriv_ext dataset"):
            await route_fetch(fetcher, "coinalyze", "unknown_dataset")
