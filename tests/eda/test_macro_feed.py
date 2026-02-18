"""Tests for src/eda/macro_feed.py — BacktestMacroProvider / LiveMacroFeed."""

from __future__ import annotations

from unittest.mock import patch

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
    async def test_start_stop(self) -> None:
        """라이프사이클 검증 — start/stop이 오류 없이 완료."""
        feed = LiveMacroFeed(refresh_interval=86400)

        with patch.object(feed, "_load_cache"):
            await feed.start()
            assert feed._task is not None

            await feed.stop()
            assert feed._task is None

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
    async def test_periodic_refresh(self) -> None:
        """_periodic_refresh에서 _load_cache 성공 시 정상 동작."""
        feed = LiveMacroFeed(refresh_interval=1)
        call_count = 0

        def mock_load() -> None:
            nonlocal call_count
            call_count += 1
            feed._shutdown.set()

        with patch.object(feed, "_load_cache", side_effect=mock_load):
            feed._shutdown.clear()
            await feed._periodic_refresh()

        assert call_count == 1
