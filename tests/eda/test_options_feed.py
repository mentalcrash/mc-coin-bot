"""Tests for src/eda/options_feed.py — BacktestOptionsProvider / LiveOptionsFeed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.eda.options_feed import BacktestOptionsProvider, LiveOptionsFeed

# ---------------------------------------------------------------------------
# BacktestOptionsProvider
# ---------------------------------------------------------------------------


class TestBacktestOptionsProvider:
    def test_enrich_merge_asof(self) -> None:
        """Precomputed GLOBAL 데이터가 merge_asof로 병합되는지 검증."""
        ohlcv_dates = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0] * 5}, index=ohlcv_dates)

        opt_dates = pd.date_range("2024-01-01", periods=3, freq="2D", tz="UTC")
        options = pd.DataFrame(
            {"opt_btc_dvol": [55.0, 60.0, 58.0]},
            index=opt_dates,
        )

        provider = BacktestOptionsProvider(options)
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")

        assert "opt_btc_dvol" in result.columns
        assert len(result) == 5
        # merge_asof backward: Jan 1 → 55, Jan 2 → 55, Jan 3 → 60
        assert result.iloc[0]["opt_btc_dvol"] == 55.0
        assert result.iloc[2]["opt_btc_dvol"] == 60.0

    def test_symbol_agnostic(self) -> None:
        """GLOBAL scope: 다른 symbol에도 동일 데이터 제공."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        options = pd.DataFrame({"opt_btc_dvol": [55.0, 60.0, 58.0]}, index=dates)
        ohlcv = pd.DataFrame({"close": [100.0] * 3}, index=dates)

        provider = BacktestOptionsProvider(options)

        btc = provider.enrich_dataframe(ohlcv.copy(), "BTC/USDT")
        eth = provider.enrich_dataframe(ohlcv.copy(), "ETH/USDT")

        pd.testing.assert_series_equal(btc["opt_btc_dvol"], eth["opt_btc_dvol"])

    def test_empty_precomputed_returns_original(self) -> None:
        """빈 precomputed DataFrame → 원본 반환."""
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )

        provider = BacktestOptionsProvider(pd.DataFrame())
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    def test_get_columns_none(self) -> None:
        """Backtest 모드는 항상 None 반환."""
        provider = BacktestOptionsProvider(pd.DataFrame())
        assert provider.get_options_columns("BTC/USDT") is None

    def test_multiple_columns(self) -> None:
        """여러 opt_* 컬럼 동시 병합."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        options = pd.DataFrame(
            {
                "opt_btc_dvol": [55.0, 60.0, 58.0],
                "opt_btc_pc_ratio": [0.8, 0.9, 0.85],
                "opt_btc_rv30d": [0.5, 0.52, 0.48],
            },
            index=dates,
        )
        ohlcv = pd.DataFrame({"close": [100.0] * 3}, index=dates)

        provider = BacktestOptionsProvider(options)
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")

        assert "opt_btc_dvol" in result.columns
        assert "opt_btc_pc_ratio" in result.columns
        assert "opt_btc_rv30d" in result.columns


# ---------------------------------------------------------------------------
# LiveOptionsFeed
# ---------------------------------------------------------------------------


class TestLiveOptionsFeed:
    @pytest.mark.asyncio
    async def test_start_creates_clients_and_tasks(self) -> None:
        """start() — client 생성 + 2개 polling task 시작."""
        feed = LiveOptionsFeed()

        with patch.object(feed, "_load_cache"):
            await feed.start()
            assert feed._client is not None
            assert feed._fetcher is not None
            assert len(feed._tasks) == 2

            await feed.stop()
            assert feed._client is None
            assert feed._fetcher is None
            assert len(feed._tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_closes_client(self) -> None:
        """stop() — client close + task cancel."""
        feed = LiveOptionsFeed()

        with patch.object(feed, "_load_cache"):
            await feed.start()
            client = feed._client
            assert client is not None

            await feed.stop()
            # Client should be cleaned up
            assert feed._client is None
            assert feed._tasks == []

    def test_enrich_broadcasts_cached(self) -> None:
        """캐시된 GLOBAL 값이 전체 행에 broadcast."""
        feed = LiveOptionsFeed()
        feed._cache = {"opt_btc_dvol": 55.0, "opt_btc_pc_ratio": 0.85}

        ohlcv = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "BTC/USDT")

        assert result["opt_btc_dvol"].iloc[0] == 55.0
        assert result["opt_btc_pc_ratio"].iloc[2] == 0.85

    def test_get_columns(self) -> None:
        """캐시 반환 검증."""
        feed = LiveOptionsFeed()
        feed._cache = {"opt_btc_dvol": 55.0}

        assert feed.get_options_columns("BTC/USDT") == {"opt_btc_dvol": 55.0}

    def test_get_columns_empty(self) -> None:
        """빈 캐시 → None."""
        feed = LiveOptionsFeed()
        assert feed.get_options_columns("BTC/USDT") is None

    def test_enrich_no_cache_returns_original(self) -> None:
        """캐시 없으면 원본 반환."""
        feed = LiveOptionsFeed()
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "BTC/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    @pytest.mark.asyncio
    async def test_poll_dvol_pcr_updates_cache(self) -> None:
        """_fetch_dvol_pcr — API 응답 → 캐시 업데이트."""
        feed = LiveOptionsFeed()

        mock_fetcher = AsyncMock()
        # BTC DVOL
        mock_fetcher.fetch_dvol = AsyncMock(
            side_effect=[
                pd.DataFrame({"close": [55.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]),
                pd.DataFrame({"close": [45.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]),
            ]
        )
        # PC Ratio
        mock_fetcher.fetch_pc_ratio = AsyncMock(
            return_value=pd.DataFrame(
                {"pc_ratio": [0.85]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_dvol_pcr()

        assert feed._cache["opt_btc_dvol"] == 55.0
        assert feed._cache["opt_eth_dvol"] == 45.0
        assert feed._cache["opt_btc_pc_ratio"] == 0.85

    @pytest.mark.asyncio
    async def test_poll_error_keeps_cache(self) -> None:
        """API 실패 시 기존 캐시 값 유지."""
        feed = LiveOptionsFeed()
        feed._cache = {"opt_btc_dvol": 55.0}

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_dvol = AsyncMock(side_effect=RuntimeError("API error"))
        mock_fetcher.fetch_pc_ratio = AsyncMock(side_effect=RuntimeError("API error"))

        feed._fetcher = mock_fetcher
        await feed._fetch_dvol_pcr()

        # 기존 값 유지
        assert feed._cache["opt_btc_dvol"] == 55.0

    @pytest.mark.asyncio
    async def test_poll_vol_term_updates_cache(self) -> None:
        """_fetch_vol_term — Hist Vol + Term Structure 캐시 업데이트."""
        feed = LiveOptionsFeed()

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_hist_vol = AsyncMock(
            return_value=pd.DataFrame(
                {"vol_30d": [0.52]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )
        mock_fetcher.fetch_term_structure = AsyncMock(
            return_value=pd.DataFrame(
                {"slope": [1.25]}, index=[pd.Timestamp("2024-01-01", tz="UTC")]
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_vol_term()

        assert feed._cache["opt_btc_rv30d"] == 0.52
        assert feed._cache["opt_btc_term_slope"] == 1.25

    @pytest.mark.asyncio
    async def test_silver_fallback_on_start(self) -> None:
        """시작 시 Silver에서 초기 로드."""
        feed = LiveOptionsFeed()
        load_called = False

        def mock_load() -> None:
            nonlocal load_called
            load_called = True
            feed._cache = {"opt_btc_dvol": 50.0}

        with patch.object(feed, "_load_cache", side_effect=mock_load):
            await feed.start()

        assert load_called
        assert feed._cache["opt_btc_dvol"] == 50.0
        await feed.stop()
