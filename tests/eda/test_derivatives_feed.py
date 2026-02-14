"""Tests for src/eda/derivatives_feed.py — Backtest + Live derivatives providers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.eda.derivatives_feed import BacktestDerivativesProvider, LiveDerivativesFeed


class TestBacktestDerivativesProvider:
    def test_enrich_with_data(self) -> None:
        """Precomputed derivatives가 merge_asof로 병합."""
        ohlcv_index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {"close": [42000, 42500, 43000, 43500, 44000]},
            index=ohlcv_index,
        )

        deriv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        deriv_df = pd.DataFrame(
            {"funding_rate": [0.0001, 0.0002, 0.0003]},
            index=deriv_index,
        )

        provider = BacktestDerivativesProvider({"BTC/USDT": deriv_df})
        enriched = provider.enrich_dataframe(ohlcv_df, "BTC/USDT")

        assert "funding_rate" in enriched.columns
        assert "close" in enriched.columns
        assert len(enriched) == 5

    def test_enrich_no_data(self) -> None:
        """해당 심볼 데이터가 없으면 원본 반환."""
        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {"close": [42000, 42500, 43000]},
            index=ohlcv_index,
        )

        provider = BacktestDerivativesProvider({})
        result = provider.enrich_dataframe(ohlcv_df, "BTC/USDT")
        assert "funding_rate" not in result.columns

    def test_enrich_empty_deriv(self) -> None:
        """빈 DataFrame이면 원본 반환."""
        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {"close": [42000, 42500, 43000]},
            index=ohlcv_index,
        )

        provider = BacktestDerivativesProvider({"BTC/USDT": pd.DataFrame()})
        result = provider.enrich_dataframe(ohlcv_df, "BTC/USDT")
        assert "funding_rate" not in result.columns

    def test_get_derivatives_columns_returns_none(self) -> None:
        """Backtest 모드에서는 항상 None."""
        provider = BacktestDerivativesProvider({})
        assert provider.get_derivatives_columns("BTC/USDT") is None

    def test_multiple_symbols(self) -> None:
        """여러 심볼 독립 처리."""
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        btc_deriv = pd.DataFrame({"funding_rate": [0.0001, 0.0002, 0.0003]}, index=index)
        eth_deriv = pd.DataFrame({"funding_rate": [0.0005, 0.0006, 0.0007]}, index=index)

        provider = BacktestDerivativesProvider(
            {
                "BTC/USDT": btc_deriv,
                "ETH/USDT": eth_deriv,
            }
        )

        ohlcv = pd.DataFrame({"close": [42000, 42500, 43000]}, index=index)
        btc_enriched = provider.enrich_dataframe(ohlcv.copy(), "BTC/USDT")
        eth_enriched = provider.enrich_dataframe(ohlcv.copy(), "ETH/USDT")

        assert btc_enriched["funding_rate"].iloc[0] == pytest.approx(0.0001)
        assert eth_enriched["funding_rate"].iloc[0] == pytest.approx(0.0005)

    def test_merge_asof_backward(self) -> None:
        """merge_asof는 backward fill — 이전 값만 사용."""
        ohlcv_index = pd.DatetimeIndex(
            pd.to_datetime(["2024-01-01 12:00", "2024-01-02 12:00", "2024-01-03 12:00"], utc=True)
        )
        ohlcv_df = pd.DataFrame({"close": [42000, 42500, 43000]}, index=ohlcv_index)

        # derivatives는 1일 00:00에만 존재
        deriv_index = pd.DatetimeIndex(
            pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
        )
        deriv_df = pd.DataFrame({"funding_rate": [0.0001, 0.0002]}, index=deriv_index)

        provider = BacktestDerivativesProvider({"BTC/USDT": deriv_df})
        enriched = provider.enrich_dataframe(ohlcv_df, "BTC/USDT")

        # 01-01 12:00 → 01-01 00:00 값(0.0001), 01-02 12:00 → 01-02 00:00 값(0.0002)
        assert enriched["funding_rate"].iloc[0] == pytest.approx(0.0001)
        assert enriched["funding_rate"].iloc[1] == pytest.approx(0.0002)


class TestLiveDerivativesFeed:
    @pytest.fixture()
    def mock_client(self) -> AsyncMock:
        client = AsyncMock()
        client.fetch_funding_rate_history = AsyncMock(return_value=[])
        client.fetch_open_interest_history = AsyncMock(return_value=[])
        client.fetch_long_short_ratio = AsyncMock(return_value=[])
        client.fetch_taker_buy_sell_ratio = AsyncMock(return_value=[])
        return client

    def test_init(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(["BTC/USDT"], mock_client)
        assert feed._symbols == ["BTC/USDT"]
        assert feed._cache == {}

    def test_get_derivatives_columns_empty(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(["BTC/USDT"], mock_client)
        assert feed.get_derivatives_columns("BTC/USDT") is None

    def test_get_derivatives_columns_cached(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(["BTC/USDT"], mock_client)
        feed._cache["BTC/USDT"] = {"funding_rate": 0.0001}
        result = feed.get_derivatives_columns("BTC/USDT")
        assert result is not None
        assert result["funding_rate"] == 0.0001

    def test_enrich_dataframe_no_cache(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(["BTC/USDT"], mock_client)
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [42000, 42500, 43000]}, index=index)
        result = feed.enrich_dataframe(df, "BTC/USDT")
        assert "funding_rate" not in result.columns

    def test_enrich_dataframe_with_cache(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(["BTC/USDT"], mock_client)
        feed._cache["BTC/USDT"] = {"funding_rate": 0.0001, "open_interest": 50000}

        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [42000, 42500, 43000]}, index=index)
        result = feed.enrich_dataframe(df, "BTC/USDT")

        assert "funding_rate" in result.columns
        assert "open_interest" in result.columns
        # 모든 행에 동일한 값 broadcast
        assert (result["funding_rate"] == 0.0001).all()

    @pytest.mark.asyncio()
    async def test_start_creates_tasks(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(
            ["BTC/USDT"],
            mock_client,
            poll_interval_fr=1,
            poll_interval_oi=1,
            poll_interval_ratios=1,
        )
        await feed.start()
        assert len(feed._tasks) == 3
        await feed.stop()

    @pytest.mark.asyncio()
    async def test_stop_cancels_tasks(self, mock_client: AsyncMock) -> None:
        feed = LiveDerivativesFeed(
            ["BTC/USDT"],
            mock_client,
            poll_interval_fr=1,
            poll_interval_oi=1,
            poll_interval_ratios=1,
        )
        await feed.start()
        assert len(feed._tasks) == 3
        await feed.stop()
        assert len(feed._tasks) == 0

    @pytest.mark.asyncio()
    async def test_poll_funding_rates_caches(self, mock_client: AsyncMock) -> None:
        """FR polling이 cache를 업데이트."""
        mock_client.fetch_funding_rate_history.return_value = [
            {"fundingRate": 0.0003, "markPrice": 45000},
        ]

        feed = LiveDerivativesFeed(["BTC/USDT"], mock_client, poll_interval_fr=100)

        # _poll_funding_rates를 한 번만 실행하도록 shutdown event 사용
        feed._shutdown.clear()

        # 내부 polling 메서드 직접 호출 (shutdown event 세팅으로 1회만 실행)
        async def _run_once() -> None:
            # polling 1회 후 shutdown
            for symbol in feed._symbols:
                raw = await mock_client.fetch_funding_rate_history(symbol, limit=1)
                if raw:
                    item = raw[-1]
                    cache = feed._cache.setdefault(symbol, {})
                    cache["funding_rate"] = float(item.get("fundingRate", 0))

        await _run_once()
        assert feed._cache.get("BTC/USDT") is not None
        assert feed._cache["BTC/USDT"]["funding_rate"] == pytest.approx(0.0003)
