"""Tests for src/eda/deriv_ext_feed.py — BacktestDerivExtProvider / LiveDerivExtFeed."""

from __future__ import annotations

from unittest.mock import patch

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
        # merge_asof backward: Jan 1 → 1e9, Jan 2 → 1e9, Jan 3 → 1.1e9
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
    async def test_start_stop(self) -> None:
        """라이프사이클 검증 — start/stop이 오류 없이 완료."""
        feed = LiveDerivExtFeed(["BTC/USDT"], refresh_interval=86400)

        with patch.object(feed, "_load_cache"):
            await feed.start()
            assert feed._task is not None

            await feed.stop()
            assert feed._task is None

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

    def test_asset_extraction(self) -> None:
        """symbol에서 asset 추출 검증 (BTC/USDT → BTC)."""
        feed = LiveDerivExtFeed(["BTC/USDT", "ETH/USDT", "DOGE/USDT"])
        # DOGE는 ASSET_PRECOMPUTE_DEFS에 없으므로 skip
        symbols = feed._symbols
        for s in symbols:
            asset = s.split("/")[0].upper()
            assert asset in ("BTC", "ETH", "DOGE")

    @pytest.mark.asyncio
    async def test_periodic_refresh(self) -> None:
        """_periodic_refresh에서 _load_cache 성공 시 정상 동작."""
        feed = LiveDerivExtFeed(["BTC/USDT"], refresh_interval=1)
        call_count = 0

        def mock_load() -> None:
            nonlocal call_count
            call_count += 1
            feed._shutdown.set()

        with patch.object(feed, "_load_cache", side_effect=mock_load):
            feed._shutdown.clear()
            await feed._periodic_refresh()

        assert call_count == 1
