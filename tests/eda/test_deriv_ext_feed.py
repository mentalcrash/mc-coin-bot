"""Tests for src/eda/deriv_ext_feed.py — BacktestDerivExtProvider / LiveDerivExtFeed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.eda.deriv_ext_feed import BacktestDerivExtProvider, LiveDerivExtFeed

# ---------------------------------------------------------------------------
# BacktestDerivExtProvider
# ---------------------------------------------------------------------------


class TestBacktestDerivExtProvider:
    def test_enrich_merge_asof(self) -> None:
        """Precomputed PER-ASSET 데이터가 merge_asof로 병합되는지 검증."""
        ohlcv_dates = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0] * 5}, index=ohlcv_dates)

        dext_dates = pd.date_range("2024-01-01", periods=3, freq="2D", tz="UTC")
        dext = pd.DataFrame(
            {"dext_agg_oi_close": [1e9, 1.1e9, 1.2e9]},
            index=dext_dates,
        )

        provider = BacktestDerivExtProvider({"BTC/USDT": dext})
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")

        assert "dext_agg_oi_close" in result.columns
        assert len(result) == 5
        assert result.iloc[0]["dext_agg_oi_close"] == 1e9
        assert result.iloc[2]["dext_agg_oi_close"] == 1.1e9

    def test_per_asset_independence(self) -> None:
        """PER-ASSET: BTC와 ETH에 다른 데이터 제공."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        btc_dext = pd.DataFrame({"dext_agg_oi_close": [1e9, 1.1e9, 1.2e9]}, index=dates)
        eth_dext = pd.DataFrame({"dext_agg_oi_close": [5e8, 5.1e8, 5.2e8]}, index=dates)
        ohlcv = pd.DataFrame({"close": [100.0] * 3}, index=dates)

        provider = BacktestDerivExtProvider({"BTC/USDT": btc_dext, "ETH/USDT": eth_dext})

        btc = provider.enrich_dataframe(ohlcv.copy(), "BTC/USDT")
        eth = provider.enrich_dataframe(ohlcv.copy(), "ETH/USDT")

        assert btc["dext_agg_oi_close"].iloc[0] == 1e9
        assert eth["dext_agg_oi_close"].iloc[0] == 5e8

    def test_missing_symbol_returns_original(self) -> None:
        """미등록 symbol → 원본 df 반환."""
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )

        provider = BacktestDerivExtProvider({})
        result = provider.enrich_dataframe(ohlcv, "DOGE/USDT")

        pd.testing.assert_frame_equal(result, ohlcv)

    def test_empty_precomputed_returns_original(self) -> None:
        """빈 precomputed DataFrame → 원본 반환."""
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )

        provider = BacktestDerivExtProvider({"BTC/USDT": pd.DataFrame()})
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    def test_get_columns_none(self) -> None:
        """Backtest 모드는 항상 None 반환."""
        provider = BacktestDerivExtProvider({"BTC/USDT": pd.DataFrame()})
        assert provider.get_deriv_ext_columns("BTC/USDT") is None

    def test_multiple_columns(self) -> None:
        """여러 dext_* 컬럼 동시 병합."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        dext = pd.DataFrame(
            {
                "dext_agg_oi_close": [1e9, 1.1e9, 1.2e9],
                "dext_liq_long_vol": [5e6, 6e6, 7e6],
                "dext_hl_oi": [2e9, 2.1e9, 2.2e9],
            },
            index=dates,
        )
        ohlcv = pd.DataFrame({"close": [100.0] * 3}, index=dates)

        provider = BacktestDerivExtProvider({"BTC/USDT": dext})
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")

        assert "dext_agg_oi_close" in result.columns
        assert "dext_liq_long_vol" in result.columns
        assert "dext_hl_oi" in result.columns


# ---------------------------------------------------------------------------
# LiveDerivExtFeed
# ---------------------------------------------------------------------------


class TestLiveDerivExtFeed:
    @pytest.mark.asyncio
    async def test_start_creates_clients_and_tasks(self) -> None:
        """start() — clients 생성 + polling tasks 시작."""
        feed = LiveDerivExtFeed(["BTC/USDT"])

        with (
            patch.object(feed, "_load_cache"),
            patch.dict("os.environ", {"COINALYZE_API_KEY": "test-key"}, clear=False),
        ):
            await feed.start()
            assert feed._ca_client is not None
            assert feed._hl_client is not None
            assert feed._ca_fetcher is not None
            assert feed._hl_fetcher is not None
            # coinalyze + hyperliquid = 2 tasks
            assert len(feed._tasks) == 2

            await feed.stop()
            assert feed._ca_client is None
            assert feed._hl_client is None
            assert len(feed._tasks) == 0

    @pytest.mark.asyncio
    async def test_missing_coinalyze_key_skips_task(self) -> None:
        """COINALYZE_API_KEY 없으면 Coinalyze polling 건너뜀."""
        feed = LiveDerivExtFeed(["BTC/USDT"])

        with (
            patch.object(feed, "_load_cache"),
            patch.dict("os.environ", {"COINALYZE_API_KEY": ""}, clear=False),
        ):
            await feed.start()
            assert feed._ca_client is None
            assert feed._ca_fetcher is None
            # hyperliquid only = 1 task
            assert len(feed._tasks) == 1

            await feed.stop()

    def test_enrich_broadcasts_cached(self) -> None:
        """캐시된 PER-ASSET 값이 전체 행에 broadcast."""
        feed = LiveDerivExtFeed(["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"dext_agg_oi_close": 1e9, "dext_hl_oi": 2e9}}

        ohlcv = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "BTC/USDT")

        assert result["dext_agg_oi_close"].iloc[0] == 1e9
        assert result["dext_hl_oi"].iloc[2] == 2e9

    def test_get_columns(self) -> None:
        """캐시 반환 검증."""
        feed = LiveDerivExtFeed(["BTC/USDT", "ETH/USDT"])
        feed._cache = {
            "BTC/USDT": {"dext_agg_oi_close": 1e9},
            "ETH/USDT": {"dext_agg_oi_close": 5e8},
        }

        assert feed.get_deriv_ext_columns("BTC/USDT") == {"dext_agg_oi_close": 1e9}
        assert feed.get_deriv_ext_columns("ETH/USDT") == {"dext_agg_oi_close": 5e8}
        assert feed.get_deriv_ext_columns("DOGE/USDT") is None

    def test_enrich_no_cache_returns_original(self) -> None:
        """캐시 없는 symbol → 원본 반환."""
        feed = LiveDerivExtFeed(["BTC/USDT"])
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "UNKNOWN/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    def test_symbol_for_asset(self) -> None:
        """asset → symbol 매핑 검증."""
        feed = LiveDerivExtFeed(["BTC/USDT", "ETH/USDT", "DOGE/USDT"])
        assert feed._symbol_for_asset("BTC") == "BTC/USDT"
        assert feed._symbol_for_asset("ETH") == "ETH/USDT"
        assert feed._symbol_for_asset("SOL") is None

    @pytest.mark.asyncio
    async def test_poll_coinalyze_updates_cache(self) -> None:
        """_fetch_coinalyze — API 응답 → PER-ASSET 캐시 업데이트."""
        feed = LiveDerivExtFeed(["BTC/USDT", "ETH/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_agg_oi = AsyncMock(
            return_value=pd.DataFrame(
                {"close": [1.5e9]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )
        mock_fetcher.fetch_agg_funding = AsyncMock(
            return_value=pd.DataFrame(
                {"close": [0.0005]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )
        mock_fetcher.fetch_liquidations = AsyncMock(
            return_value=pd.DataFrame(
                {"long_volume": [5e6], "short_volume": [3e6]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )
        mock_fetcher.fetch_cvd = AsyncMock(
            return_value=pd.DataFrame(
                {"buy_volume": [2e7]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )

        feed._ca_fetcher = mock_fetcher
        await feed._fetch_coinalyze()

        assert feed._cache["BTC/USDT"]["dext_agg_oi_close"] == 1.5e9
        assert feed._cache["BTC/USDT"]["dext_agg_funding_close"] == 0.0005
        assert feed._cache["BTC/USDT"]["dext_liq_long_vol"] == 5e6
        assert feed._cache["BTC/USDT"]["dext_cvd_buy_vol"] == 2e7

    @pytest.mark.asyncio
    async def test_poll_hyperliquid_updates_cache(self) -> None:
        """_fetch_hyperliquid — asset contexts → PER-ASSET 캐시 업데이트."""
        feed = LiveDerivExtFeed(["BTC/USDT", "ETH/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_asset_contexts = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "coin": ["BTC", "ETH"],
                    "open_interest": [5e9, 2e9],
                    "funding": [0.0001, 0.0002],
                },
                index=pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True),
            )
        )

        feed._hl_fetcher = mock_fetcher
        await feed._fetch_hyperliquid()

        assert feed._cache["BTC/USDT"]["dext_hl_oi"] == 5e9
        assert feed._cache["BTC/USDT"]["dext_hl_funding"] == 0.0001
        assert feed._cache["ETH/USDT"]["dext_hl_oi"] == 2e9

    @pytest.mark.asyncio
    async def test_poll_error_keeps_cache(self) -> None:
        """API 실패 시 기존 캐시 값 유지."""
        feed = LiveDerivExtFeed(["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"dext_hl_oi": 5e9}}

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_asset_contexts = AsyncMock(side_effect=RuntimeError("API error"))

        feed._hl_fetcher = mock_fetcher
        await feed._fetch_hyperliquid()

        assert feed._cache["BTC/USDT"]["dext_hl_oi"] == 5e9
