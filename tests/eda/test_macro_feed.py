"""Tests for src/eda/macro_feed.py — BacktestMacroProvider / LiveMacroFeed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.eda.macro_feed import BacktestMacroProvider, LiveMacroFeed

# ---------------------------------------------------------------------------
# BacktestMacroProvider
# ---------------------------------------------------------------------------


class TestBacktestMacroProvider:
    def test_enrich_merge_asof(self) -> None:
        """Precomputed GLOBAL 데이터가 merge_asof로 병합되는지 검증."""
        ohlcv_dates = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0] * 5}, index=ohlcv_dates)

        macro_dates = pd.date_range("2024-01-01", periods=3, freq="2D", tz="UTC")
        macro = pd.DataFrame(
            {"macro_dxy": [103.5, 104.0, 103.8]},
            index=macro_dates,
        )

        provider = BacktestMacroProvider(macro)
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")

        assert "macro_dxy" in result.columns
        assert len(result) == 5
        # merge_asof backward: Jan 1 → 103.5, Jan 2 → 103.5, Jan 3 → 104.0
        assert result.iloc[0]["macro_dxy"] == 103.5
        assert result.iloc[2]["macro_dxy"] == 104.0

    def test_symbol_agnostic(self) -> None:
        """GLOBAL scope: 다른 symbol에도 동일 데이터 제공."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        macro = pd.DataFrame({"macro_vix": [20.0, 21.0, 22.0]}, index=dates)
        ohlcv = pd.DataFrame({"close": [100.0] * 3}, index=dates)

        provider = BacktestMacroProvider(macro)

        btc = provider.enrich_dataframe(ohlcv.copy(), "BTC/USDT")
        eth = provider.enrich_dataframe(ohlcv.copy(), "ETH/USDT")

        pd.testing.assert_series_equal(btc["macro_vix"], eth["macro_vix"])

    def test_empty_precomputed_returns_original(self) -> None:
        """빈 precomputed DataFrame → 원본 반환."""
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )

        provider = BacktestMacroProvider(pd.DataFrame())
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    def test_get_columns_none(self) -> None:
        """Backtest 모드는 항상 None 반환."""
        provider = BacktestMacroProvider(pd.DataFrame())
        assert provider.get_macro_columns("BTC/USDT") is None


# ---------------------------------------------------------------------------
# LiveMacroFeed
# ---------------------------------------------------------------------------


class TestLiveMacroFeed:
    @pytest.mark.asyncio
    async def test_start_creates_clients_and_tasks(self) -> None:
        """start() — clients 생성 + polling tasks 시작."""
        feed = LiveMacroFeed()

        with (
            patch.object(feed, "_load_cache"),
            patch.dict("os.environ", {"FRED_API_KEY": "test-key"}, clear=False),
        ):
            await feed.start()
            assert feed._fred_client is not None
            assert feed._cg_client is not None
            assert feed._fetcher is not None
            # FRED + yfinance + coingecko = 3 tasks
            assert len(feed._tasks) == 3

            await feed.stop()
            assert feed._fred_client is None
            assert feed._cg_client is None
            assert feed._fetcher is None
            assert len(feed._tasks) == 0

    @pytest.mark.asyncio
    async def test_missing_fred_key_skips_fred_task(self) -> None:
        """FRED_API_KEY 없으면 FRED polling task 건너뜀."""
        feed = LiveMacroFeed()

        with (
            patch.object(feed, "_load_cache"),
            patch.dict("os.environ", {"FRED_API_KEY": ""}, clear=False),
        ):
            await feed.start()
            # yfinance + coingecko only = 2 tasks
            assert len(feed._tasks) == 2

            await feed.stop()

    def test_enrich_broadcasts_cached(self) -> None:
        """캐시된 GLOBAL 값이 전체 행에 broadcast."""
        feed = LiveMacroFeed()
        feed._cache = {"macro_dxy": 103.5, "macro_vix": 20.0}

        ohlcv = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "BTC/USDT")

        assert result["macro_dxy"].iloc[0] == 103.5
        assert result["macro_vix"].iloc[2] == 20.0

    def test_enrich_symbol_agnostic(self) -> None:
        """GLOBAL: 어떤 symbol이든 동일 값 broadcast."""
        feed = LiveMacroFeed()
        feed._cache = {"macro_dxy": 103.5}

        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )
        btc = feed.enrich_dataframe(ohlcv.copy(), "BTC/USDT")
        eth = feed.enrich_dataframe(ohlcv.copy(), "ETH/USDT")

        assert btc["macro_dxy"].iloc[0] == eth["macro_dxy"].iloc[0]

    def test_get_columns(self) -> None:
        """캐시 반환 검증."""
        feed = LiveMacroFeed()
        feed._cache = {"macro_dxy": 103.5}

        assert feed.get_macro_columns("BTC/USDT") == {"macro_dxy": 103.5}
        assert feed.get_macro_columns("ETH/USDT") == {"macro_dxy": 103.5}

    def test_get_columns_empty(self) -> None:
        """빈 캐시 → None."""
        feed = LiveMacroFeed()
        assert feed.get_macro_columns("BTC/USDT") is None

    def test_enrich_no_cache_returns_original(self) -> None:
        """캐시 없으면 원본 반환."""
        feed = LiveMacroFeed()
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "BTC/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    @pytest.mark.asyncio
    async def test_poll_fred_updates_cache(self) -> None:
        """_fetch_fred — API 응답 → 캐시 업데이트."""
        feed = LiveMacroFeed()

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_fred_series = AsyncMock(
            return_value=pd.DataFrame(
                {"value": [103.5]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_fred()

        # All 6 FRED series should be called
        assert mock_fetcher.fetch_fred_series.call_count == 6
        assert feed._cache["macro_dxy"] == 103.5

    @pytest.mark.asyncio
    async def test_poll_coingecko_updates_cache(self) -> None:
        """_fetch_coingecko — CoinGecko global + defi → 캐시 업데이트."""
        feed = LiveMacroFeed()

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_coingecko_global = AsyncMock(
            return_value=pd.DataFrame(
                {"btc_dominance": [55.0], "total_market_cap_usd": [2.5e12]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )
        mock_fetcher.fetch_coingecko_defi = AsyncMock(
            return_value=pd.DataFrame(
                {"defi_dominance": [3.5]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_coingecko()

        assert feed._cache["macro_btc_dom"] == 55.0
        assert feed._cache["macro_total_mcap"] == 2.5e12
        assert feed._cache["macro_defi_dom"] == 3.5

    @pytest.mark.asyncio
    async def test_poll_error_keeps_cache(self) -> None:
        """API 실패 시 기존 캐시 값 유지."""
        feed = LiveMacroFeed()
        feed._cache = {"macro_dxy": 103.5}

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_fred_series = AsyncMock(side_effect=RuntimeError("API error"))

        feed._fetcher = mock_fetcher
        await feed._fetch_fred()

        assert feed._cache["macro_dxy"] == 103.5

    @pytest.mark.asyncio
    async def test_poll_yfinance_updates_cache(self) -> None:
        """_fetch_yfinance — yfinance 응답 → 캐시 업데이트."""
        feed = LiveMacroFeed()

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_yfinance_ticker = AsyncMock(
            return_value=pd.DataFrame(
                {"close": [450.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_yfinance()

        assert mock_fetcher.fetch_yfinance_ticker.call_count == 6
        assert feed._cache["macro_spy_close"] == 450.0
