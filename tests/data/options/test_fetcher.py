"""Tests for Deribit Options data fetcher (OptionsFetcher + route_fetch)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.options.client import AsyncOptionsClient
from src.data.options.fetcher import DERIBIT_DATASETS, OptionsFetcher, route_fetch


@pytest.fixture
def mock_client() -> AsyncMock:
    """Mock AsyncOptionsClient."""
    return AsyncMock(spec=AsyncOptionsClient)


def _make_response(data: dict) -> MagicMock:  # type: ignore[type-arg]
    """Mock httpx.Response with JSON data."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = data
    return resp


class TestOptionsFetcher:
    """OptionsFetcher 테스트."""

    async def test_fetch_dvol(self, mock_client: AsyncMock) -> None:
        """DVOL 데이터 가져오기."""
        dvol_data = {
            "result": {
                "ticks": [1705276800000, 1705363200000],
                "open": [60.0, 62.0],
                "high": [65.0, 67.0],
                "low": [58.0, 60.0],
                "close": [63.0, 65.0],
                "volume": [100.0, 110.0],
            }
        }
        # First call returns data, subsequent calls return empty (pagination)
        empty_data = {"result": {}}
        mock_client.get.side_effect = [
            _make_response(dvol_data),
            _make_response(empty_data),
            _make_response(empty_data),
        ]
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_dvol("BTC", start="2024-01-15", end="2024-01-16")

        assert len(df) == 2
        assert "date" in df.columns
        assert "close" in df.columns
        assert "currency" in df.columns
        assert df.iloc[0]["currency"] == "BTC"

    async def test_fetch_dvol_empty(self, mock_client: AsyncMock) -> None:
        """빈 DVOL 응답."""
        mock_client.get.return_value = _make_response({"result": {}})
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_dvol("BTC")
        assert df.empty

    async def test_fetch_pc_ratio(self, mock_client: AsyncMock) -> None:
        """Put/Call Ratio 가져오기."""
        book_data = {
            "result": [
                {"instrument_name": "BTC-26JAN24-40000-C", "open_interest": 100},
                {"instrument_name": "BTC-26JAN24-40000-P", "open_interest": 60},
                {"instrument_name": "BTC-26JAN24-42000-C", "open_interest": 150},
                {"instrument_name": "BTC-26JAN24-42000-P", "open_interest": 90},
            ]
        }
        mock_client.get.return_value = _make_response(book_data)
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_pc_ratio("BTC")

        assert len(df) == 1
        assert "pc_ratio" in df.columns
        assert "put_oi" in df.columns
        assert "call_oi" in df.columns
        # put=150, call=250, ratio=0.6
        assert float(df.iloc[0]["put_oi"]) == 150.0
        assert float(df.iloc[0]["call_oi"]) == 250.0

    async def test_fetch_hist_vol(self, mock_client: AsyncMock) -> None:
        """Historical Volatility 가져오기."""
        hvol_data = {
            "result": [
                [1705276800000, 40.0, 50.0, 55.0, 58.0, 60.0, 62.0, 65.0],
                [1705363200000, 42.0, 52.0, 57.0, 60.0, 62.0, 64.0, 67.0],
            ]
        }
        mock_client.get.return_value = _make_response(hvol_data)
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_hist_vol("BTC")

        assert len(df) == 2
        assert "vol_7d" in df.columns
        assert "vol_30d" in df.columns
        assert "vol_365d" in df.columns

    async def test_fetch_hist_vol_empty(self, mock_client: AsyncMock) -> None:
        """빈 Historical Volatility 응답."""
        mock_client.get.return_value = _make_response({"result": []})
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_hist_vol("BTC")
        assert df.empty

    async def test_fetch_term_structure(self, mock_client: AsyncMock) -> None:
        """Term Structure 가져오기."""
        # Step 1: instruments
        instruments_data = {
            "result": [
                {
                    "instrument_name": "BTC-26JAN24",
                    "expiration_timestamp": 1706227200000,
                    "settlement_period": "month",
                },
                {
                    "instrument_name": "BTC-29MAR24",
                    "expiration_timestamp": 1711670400000,
                    "settlement_period": "month",
                },
                {
                    "instrument_name": "BTC-PERPETUAL",
                    "expiration_timestamp": 0,
                    "settlement_period": "perpetual",
                },
            ]
        }
        # Step 2: tickers
        near_ticker = {"result": {"mark_price": 43500}}
        far_ticker = {"result": {"mark_price": 44200}}
        # Step 3: index
        index_data = {"result": {"index_price": 43000}}

        mock_client.get.side_effect = [
            _make_response(instruments_data),
            _make_response(near_ticker),
            _make_response(far_ticker),
            _make_response(index_data),
        ]
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_term_structure("BTC")

        assert len(df) == 1
        assert "slope" in df.columns
        assert "near_expiry" in df.columns
        assert "far_expiry" in df.columns
        assert df.iloc[0]["near_expiry"] == "BTC-26JAN24"

    async def test_fetch_term_structure_insufficient_futures(self, mock_client: AsyncMock) -> None:
        """Futures가 2개 미만일 때 빈 결과."""
        instruments_data = {
            "result": [
                {
                    "instrument_name": "BTC-PERPETUAL",
                    "expiration_timestamp": 0,
                    "settlement_period": "perpetual",
                },
            ]
        }
        mock_client.get.return_value = _make_response(instruments_data)
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_term_structure("BTC")
        assert df.empty

    async def test_fetch_max_pain(self, mock_client: AsyncMock) -> None:
        """Max Pain 계산."""
        book_data = {
            "result": [
                {"instrument_name": "BTC-26JAN24-40000-C", "open_interest": 100},
                {"instrument_name": "BTC-26JAN24-40000-P", "open_interest": 50},
                {"instrument_name": "BTC-26JAN24-42000-C", "open_interest": 200},
                {"instrument_name": "BTC-26JAN24-42000-P", "open_interest": 80},
                {"instrument_name": "BTC-26JAN24-44000-C", "open_interest": 50},
                {"instrument_name": "BTC-26JAN24-44000-P", "open_interest": 150},
            ]
        }
        mock_client.get.return_value = _make_response(book_data)
        fetcher = OptionsFetcher(mock_client)

        df = await fetcher.fetch_max_pain("BTC")

        assert len(df) == 1
        assert "max_pain_strike" in df.columns
        assert "total_oi" in df.columns
        assert "expiry" in df.columns

    async def test_all_datasets_defined(self) -> None:
        """모든 Deribit 데이터셋이 정의되어 있는지."""
        assert len(DERIBIT_DATASETS) == 6
        assert "btc_dvol" in DERIBIT_DATASETS
        assert "eth_dvol" in DERIBIT_DATASETS
        assert "btc_pc_ratio" in DERIBIT_DATASETS
        assert "btc_hist_vol" in DERIBIT_DATASETS
        assert "btc_term_structure" in DERIBIT_DATASETS
        assert "btc_max_pain" in DERIBIT_DATASETS


class TestRouteFetch:
    """route_fetch 함수 테스트."""

    async def test_route_dvol(self, mock_client: AsyncMock) -> None:
        """DVOL 라우팅."""
        dvol_data = {
            "result": {
                "ticks": [1705276800000],
                "open": [60.0],
                "high": [65.0],
                "low": [58.0],
                "close": [63.0],
                "volume": [100.0],
            }
        }
        mock_client.get.return_value = _make_response(dvol_data)
        fetcher = OptionsFetcher(mock_client)

        df = await route_fetch(fetcher, "deribit", "btc_dvol")
        assert not df.empty

    async def test_route_unknown_source(self, mock_client: AsyncMock) -> None:
        """알 수 없는 source."""
        fetcher = OptionsFetcher(mock_client)
        with pytest.raises(ValueError, match="Unknown options source"):
            await route_fetch(fetcher, "unknown", "btc_dvol")

    async def test_route_unknown_dataset(self, mock_client: AsyncMock) -> None:
        """알 수 없는 dataset."""
        fetcher = OptionsFetcher(mock_client)
        with pytest.raises(ValueError, match="Unknown options dataset"):
            await route_fetch(fetcher, "deribit", "unknown_dataset")
