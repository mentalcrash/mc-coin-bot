"""Tests for src/eda/options_feed.py — BacktestOptionsProvider / LiveOptionsFeed."""

from __future__ import annotations

from unittest.mock import patch

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
    async def test_start_stop(self) -> None:
        """라이프사이클 검증 — start/stop이 오류 없이 완료."""
        feed = LiveOptionsFeed(refresh_interval=86400)

        with patch.object(feed, "_load_cache"):
            await feed.start()
            assert feed._task is not None

            await feed.stop()
            assert feed._task is None

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
    async def test_periodic_refresh(self) -> None:
        """_periodic_refresh에서 _load_cache 성공 시 정상 동작."""
        feed = LiveOptionsFeed(refresh_interval=1)
        call_count = 0

        def mock_load() -> None:
            nonlocal call_count
            call_count += 1
            feed._shutdown.set()

        with patch.object(feed, "_load_cache", side_effect=mock_load):
            feed._shutdown.clear()
            await feed._periodic_refresh()

        assert call_count == 1
