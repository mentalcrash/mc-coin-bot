"""Tests for src/eda/onchain_feed.py — BacktestOnchainProvider / LiveOnchainFeed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.exceptions import StorageError
from src.eda.onchain_feed import BacktestOnchainProvider, LiveOnchainFeed

# ---------------------------------------------------------------------------
# BacktestOnchainProvider
# ---------------------------------------------------------------------------


class TestBacktestOnchainProvider:
    def test_enrich_merge_asof(self) -> None:
        """Precomputed 데이터가 merge_asof로 병합되는지 검증."""
        ohlcv_dates = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0] * 5}, index=ohlcv_dates)

        onchain_dates = pd.date_range("2024-01-01", periods=3, freq="2D", tz="UTC")
        onchain = pd.DataFrame(
            {"oc_fear_greed": [72.0, 68.0, 75.0]},
            index=onchain_dates,
        )

        provider = BacktestOnchainProvider({"BTC/USDT": onchain})
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")

        assert "oc_fear_greed" in result.columns
        assert len(result) == 5
        # merge_asof backward: Jan 1 → 72, Jan 2 → 72, Jan 3 → 68
        assert result.iloc[0]["oc_fear_greed"] == 72.0
        assert result.iloc[2]["oc_fear_greed"] == 68.0

    def test_missing_symbol_returns_original(self) -> None:
        """미등록 symbol → 원본 df 반환."""
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )

        provider = BacktestOnchainProvider({})
        result = provider.enrich_dataframe(ohlcv, "ETH/USDT")

        pd.testing.assert_frame_equal(result, ohlcv)

    def test_get_columns_none(self) -> None:
        """Backtest 모드는 항상 None 반환."""
        provider = BacktestOnchainProvider({"BTC/USDT": pd.DataFrame()})
        assert provider.get_onchain_columns("BTC/USDT") is None

    def test_empty_precomputed_returns_original(self) -> None:
        """빈 precomputed DataFrame → 원본 반환."""
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )

        provider = BacktestOnchainProvider({"BTC/USDT": pd.DataFrame()})
        result = provider.enrich_dataframe(ohlcv, "BTC/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)


# ---------------------------------------------------------------------------
# LiveOnchainFeed
# ---------------------------------------------------------------------------


class TestLiveOnchainFeed:
    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """라이프사이클 검증 — start/stop이 오류 없이 완료."""
        feed = LiveOnchainFeed(["BTC/USDT"], refresh_interval=86400)

        # _load_cache를 빈 캐시로 patch
        with patch.object(feed, "_load_cache"):
            await feed.start()
            assert feed._task is not None

            await feed.stop()
            assert feed._task is None

    def test_enrich_broadcasts_cached(self) -> None:
        """캐시된 값이 전체 행에 broadcast."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"oc_fear_greed": 72.0, "oc_tvl_usd": 150e9}}

        ohlcv = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "BTC/USDT")

        assert result["oc_fear_greed"].iloc[0] == 72.0
        assert result["oc_tvl_usd"].iloc[2] == 150e9

    def test_get_columns(self) -> None:
        """캐시 반환 검증."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"oc_fear_greed": 72.0}}

        assert feed.get_onchain_columns("BTC/USDT") == {"oc_fear_greed": 72.0}
        assert feed.get_onchain_columns("ETH/USDT") is None

    def test_no_data_graceful(self) -> None:
        """Silver 없으면 빈 캐시."""
        feed = LiveOnchainFeed(["DOGE/USDT"])

        # _load_cache가 service.load에서 예외 → 빈 캐시
        with patch(
            "src.data.onchain.service.OnchainDataService.load",
            side_effect=StorageError("not found"),
        ):
            feed._load_cache()

        assert feed._cache == {}

    def test_enrich_no_cache_returns_original(self) -> None:
        """캐시 없는 symbol → 원본 반환."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        ohlcv = pd.DataFrame(
            {"close": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC"),
        )
        result = feed.enrich_dataframe(ohlcv, "UNKNOWN/USDT")
        pd.testing.assert_frame_equal(result, ohlcv)

    def test_get_health_status(self) -> None:
        """get_health_status()가 캐시 상태를 반환."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])
        feed._cache = {
            "BTC/USDT": {"oc_fear_greed": 72.0, "oc_tvl_usd": 150e9},
            "ETH/USDT": {"oc_fear_greed": 72.0},
        }
        health = feed.get_health_status()
        assert health["symbols_cached"] == 2
        assert health["total_columns"] == 3

    def test_get_health_status_empty(self) -> None:
        """빈 캐시 → 0."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        health = feed.get_health_status()
        assert health["symbols_cached"] == 0
        assert health["total_columns"] == 0

    def test_update_cache_metrics(self) -> None:
        """update_cache_metrics() 호출 시 오류 없이 완료."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"oc_fear_greed": 72.0}}
        # Should not raise even if prometheus_client imported
        feed.update_cache_metrics()

    def test_notification_queue_default_none(self) -> None:
        """기본 _notification_queue는 None."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        assert feed._notification_queue is None

    @pytest.mark.asyncio
    async def test_periodic_refresh_success_increments(self) -> None:
        """_periodic_refresh에서 _load_cache 성공 시 success counter."""
        feed = LiveOnchainFeed(["BTC/USDT"], refresh_interval=1)
        call_count = 0

        def mock_load() -> None:
            nonlocal call_count
            call_count += 1
            feed._shutdown.set()  # 1회 refresh 후 종료

        with patch.object(feed, "_load_cache", side_effect=mock_load):
            feed._shutdown.clear()
            await feed._periodic_refresh()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_periodic_refresh_failure_does_not_crash(self) -> None:
        """_periodic_refresh에서 _load_cache 예외 시 계속 실행."""
        feed = LiveOnchainFeed(["BTC/USDT"], refresh_interval=1)
        call_count = 0

        def mock_load() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "test error"
                raise RuntimeError(msg)
            feed._shutdown.set()

        with patch.object(feed, "_load_cache", side_effect=mock_load):
            feed._shutdown.clear()
            await feed._periodic_refresh()

        assert call_count == 2  # 1 failure + 1 success then shutdown


# ---------------------------------------------------------------------------
# Helper: mock silver side_load
# ---------------------------------------------------------------------------

_NOT_FOUND = StorageError("not found")


def _make_mock_silver(
    registry: dict[tuple[str, str], pd.DataFrame],
) -> MagicMock:
    """(source, name) → DataFrame 매핑에서 mock silver 생성."""
    mock = MagicMock()

    def _load(source: str, name: str) -> pd.DataFrame:
        key = (source, name)
        if key in registry:
            return registry[key]
        raise _NOT_FOUND

    mock.load = MagicMock(side_effect=_load)
    return mock


# ---------------------------------------------------------------------------
# precompute() via OnchainDataService
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_precompute_btc_global_and_asset(self) -> None:
        """BTC/USDT가 global + BTC asset sources 모두 받는지."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._catalog = None

        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")

        def _df(cols: dict[str, list[float]], dc: str = "date") -> pd.DataFrame:
            return pd.DataFrame({dc: dates, **cols})

        service._silver = _make_mock_silver(
            {
                ("defillama", "stablecoin_total"): _df(
                    {"total_circulating_usd": [1e9, 1.1e9, 1.2e9]}
                ),
                ("defillama", "tvl_total"): _df({"tvl_usd": [50e9, 51e9, 52e9]}),
                ("defillama", "dex_volume"): _df({"volume_usd": [1e9, 1.1e9, 1.2e9]}),
                ("alternative_me", "fear_greed"): _df({"value": [72, 68, 75]}, dc="timestamp"),
                ("coinmetrics", "btc_metrics"): _df(
                    {
                        "CapMVRVCur": [1.5, 1.6, 1.7],
                        "CapMrktCurUSD": [400e9, 410e9, 420e9],
                        "FlowInExUSD": [10e6, 11e6, 12e6],
                        "FlowOutExUSD": [8e6, 9e6, 10e6],
                    },
                    dc="time",
                ),
                ("blockchain_com", "bc_hash-rate"): _df(
                    {"value": [500e18, 510e18, 520e18]},
                    dc="timestamp",
                ),
                ("mempool_space", "mining"): _df(
                    {"avg_hashrate": [500e18, 510e18, 520e18], "difficulty": [70e12, 71e12, 72e12]},
                    dc="timestamp",
                ),
            }
        )

        ohlcv_index = pd.date_range("2024-01-02", periods=3, freq="1D", tz="UTC")
        result = service.precompute("BTC/USDT", ohlcv_index)

        # Global columns
        assert "oc_stablecoin_total_usd" in result.columns
        assert "oc_tvl_usd" in result.columns
        assert "oc_fear_greed" in result.columns
        # BTC asset columns
        assert "oc_mvrv" in result.columns
        assert "oc_hash_rate" in result.columns

    def test_precompute_doge_global_only(self) -> None:
        """DOGE/USDT는 global sources만 받음."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._catalog = None

        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        service._silver = _make_mock_silver(
            {
                ("defillama", "stablecoin_total"): pd.DataFrame(
                    {"date": dates, "total_circulating_usd": [1e9] * 3}
                ),
                ("defillama", "tvl_total"): pd.DataFrame({"date": dates, "tvl_usd": [50e9] * 3}),
                ("defillama", "dex_volume"): pd.DataFrame({"date": dates, "volume_usd": [1e9] * 3}),
                ("alternative_me", "fear_greed"): pd.DataFrame(
                    {"timestamp": dates, "value": [72] * 3}
                ),
            }
        )

        ohlcv_index = pd.date_range("2024-01-02", periods=3, freq="1D", tz="UTC")
        result = service.precompute("DOGE/USDT", ohlcv_index)

        assert "oc_stablecoin_total_usd" in result.columns
        assert "oc_fear_greed" in result.columns
        # No BTC/ETH specific columns
        assert "oc_mvrv" not in result.columns
        assert "oc_hash_rate" not in result.columns

    def test_precompute_oc_prefix(self) -> None:
        """모든 컬럼이 oc_로 시작."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._catalog = None

        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        service._silver = _make_mock_silver(
            {
                ("defillama", "stablecoin_total"): pd.DataFrame(
                    {"date": dates, "total_circulating_usd": [1e9] * 3}
                ),
            }
        )

        ohlcv_index = pd.date_range("2024-01-02", periods=3, freq="1D", tz="UTC")
        result = service.precompute("SOL/USDT", ohlcv_index)

        for col in result.columns:
            assert col.startswith("oc_"), f"Column {col} does not start with oc_"

    def test_precompute_lag_applied(self) -> None:
        """T+1 shift 적용 확인 (defillama lag=1)."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._catalog = None

        dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        service._silver = _make_mock_silver(
            {
                ("defillama", "stablecoin_total"): pd.DataFrame(
                    {"date": dates, "total_circulating_usd": [1e9, 1.1e9, 1.2e9]}
                ),
            }
        )

        ohlcv_index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
        result = service.precompute("SOL/USDT", ohlcv_index)

        # lag=1 → Jan 1 data available at Jan 2
        assert pd.isna(result.iloc[0]["oc_stablecoin_total_usd"])  # Jan 1: no data yet
        assert result.iloc[1]["oc_stablecoin_total_usd"] == 1e9  # Jan 2: Jan 1 data

    def test_precompute_no_data_empty_df(self) -> None:
        """Silver 데이터 없으면 빈 DataFrame."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._catalog = None
        service._silver = _make_mock_silver({})  # all loads will raise

        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        result = service.precompute("BTC/USDT", ohlcv_index)

        assert result.columns.empty
        assert len(result) == 3
