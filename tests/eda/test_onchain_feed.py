"""Tests for src/eda/onchain_feed.py — BacktestOnchainProvider / LiveOnchainFeed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
    async def test_start_creates_clients_and_tasks(self) -> None:
        """start() — client 생성 + polling tasks 시작."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])

        with (
            patch.object(feed, "_load_cache"),
            patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test-key"}, clear=False),
        ):
            await feed.start()
            assert feed._fetcher is not None
            assert len(feed._clients) == 1
            # defillama + sentiment + coinmetrics + btc_mining + eth_supply = 5
            assert len(feed._tasks) == 5

            await feed.stop()
            assert feed._fetcher is None
            assert len(feed._clients) == 0
            assert len(feed._tasks) == 0

    @pytest.mark.asyncio
    async def test_missing_etherscan_key_skips_eth_supply(self) -> None:
        """ETHERSCAN_API_KEY 없으면 ETH supply polling 건너뜀."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])

        with (
            patch.object(feed, "_load_cache"),
            patch.dict("os.environ", {"ETHERSCAN_API_KEY": ""}, clear=False),
        ):
            await feed.start()
            # defillama + sentiment + coinmetrics + btc_mining = 4
            assert len(feed._tasks) == 4

            await feed.stop()

    @pytest.mark.asyncio
    async def test_no_btc_symbol_skips_mining(self) -> None:
        """BTC 심볼 없으면 mining polling 건너뜀."""
        feed = LiveOnchainFeed(["ETH/USDT"])

        with patch.object(feed, "_load_cache"):
            await feed.start()
            # defillama + sentiment + coinmetrics = 3 (no btc_mining, no eth_supply w/o key)
            assert len(feed._tasks) == 3

            await feed.stop()

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

        with (
            patch(
                "src.data.onchain.service.OnchainDataService.load",
                side_effect=StorageError("not found"),
            ),
            patch(
                "src.data.macro.storage.MacroSilverProcessor.load",
                side_effect=StorageError("not found"),
            ),
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
        feed.update_cache_metrics()

    def test_notification_queue_default_none(self) -> None:
        """기본 _notification_queue는 None."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        assert feed._notification_queue is None

    def test_symbol_for_asset(self) -> None:
        """_symbol_for_asset 매핑 검증."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])
        assert feed._symbol_for_asset("BTC") == "BTC/USDT"
        assert feed._symbol_for_asset("ETH") == "ETH/USDT"
        assert feed._symbol_for_asset("SOL") is None

    def test_set_global_cache(self) -> None:
        """_set_global_cache — 모든 심볼에 값 설정."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])
        feed._set_global_cache("oc_fear_greed", 72.0)

        assert feed._cache["BTC/USDT"]["oc_fear_greed"] == 72.0
        assert feed._cache["ETH/USDT"]["oc_fear_greed"] == 72.0

    @pytest.mark.asyncio
    async def test_poll_defillama_updates_cache(self) -> None:
        """_fetch_defillama — DeFiLlama → GLOBAL 캐시 업데이트."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_stablecoin_total = AsyncMock(
            return_value=pd.DataFrame(
                {"total_circulating_usd": [150e9]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )
        mock_fetcher.fetch_tvl = AsyncMock(
            return_value=pd.DataFrame(
                {"tvl_usd": [50e9]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )
        mock_fetcher.fetch_dex_volume = AsyncMock(
            return_value=pd.DataFrame(
                {"volume_usd": [5e9]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_defillama()

        # GLOBAL → 모든 심볼에 설정
        assert feed._cache["BTC/USDT"]["oc_stablecoin_total_usd"] == 150e9
        assert feed._cache["ETH/USDT"]["oc_tvl_usd"] == 50e9
        assert feed._cache["BTC/USDT"]["oc_dex_volume_usd"] == 5e9

    @pytest.mark.asyncio
    async def test_poll_sentiment_updates_cache(self) -> None:
        """_fetch_sentiment — Fear & Greed → GLOBAL 캐시 업데이트."""
        feed = LiveOnchainFeed(["BTC/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_fear_greed = AsyncMock(
            return_value=pd.DataFrame(
                {"value": [72]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_sentiment()

        assert feed._cache["BTC/USDT"]["oc_fear_greed"] == 72.0

    @pytest.mark.asyncio
    async def test_poll_coinmetrics_updates_cache(self) -> None:
        """_fetch_coinmetrics — CM BTC/ETH → PER-ASSET 캐시 업데이트."""
        feed = LiveOnchainFeed(["BTC/USDT", "ETH/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_coinmetrics = AsyncMock(
            return_value=pd.DataFrame(
                {"CapMVRVCur": [1.5], "CapMrktCurUSD": [400e9]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_coinmetrics()

        assert feed._cache["BTC/USDT"]["oc_mvrv"] == 1.5
        assert feed._cache["BTC/USDT"]["oc_mktcap_usd"] == 400e9

    @pytest.mark.asyncio
    async def test_poll_btc_mining_updates_cache(self) -> None:
        """_fetch_btc_mining — mempool.space → BTC 캐시 업데이트."""
        feed = LiveOnchainFeed(["BTC/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_mempool_mining = AsyncMock(
            return_value=pd.DataFrame(
                {"avg_hashrate": [500e18], "difficulty": [70e12]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )

        feed._fetcher = mock_fetcher
        await feed._fetch_btc_mining()

        assert feed._cache["BTC/USDT"]["oc_avg_hashrate"] == 500e18
        assert feed._cache["BTC/USDT"]["oc_difficulty"] == 70e12

    @pytest.mark.asyncio
    async def test_poll_eth_supply_updates_cache(self) -> None:
        """_fetch_eth_supply — Etherscan → ETH 캐시 업데이트."""
        feed = LiveOnchainFeed(["ETH/USDT"])

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_eth_supply = AsyncMock(
            return_value=pd.DataFrame(
                {"eth_supply": [120e6], "eth2_staking": [30e6]},
                index=[pd.Timestamp("2024-01-01", tz="UTC")],
            )
        )

        feed._fetcher = mock_fetcher
        with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "test-key"}, clear=False):
            await feed._fetch_eth_supply()

        assert feed._cache["ETH/USDT"]["oc_eth_supply"] == 120e6
        assert feed._cache["ETH/USDT"]["oc_eth2_staking"] == 30e6

    @pytest.mark.asyncio
    async def test_poll_error_keeps_cache(self) -> None:
        """API 실패 시 기존 캐시 값 유지."""
        feed = LiveOnchainFeed(["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"oc_fear_greed": 72.0}}

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_stablecoin_total = AsyncMock(side_effect=RuntimeError("API error"))
        mock_fetcher.fetch_tvl = AsyncMock(side_effect=RuntimeError("API error"))
        mock_fetcher.fetch_dex_volume = AsyncMock(side_effect=RuntimeError("API error"))

        feed._fetcher = mock_fetcher
        await feed._fetch_defillama()

        assert feed._cache["BTC/USDT"]["oc_fear_greed"] == 72.0


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

        assert pd.isna(result.iloc[0]["oc_stablecoin_total_usd"])
        assert result.iloc[1]["oc_stablecoin_total_usd"] == 1e9

    def test_precompute_no_data_empty_df(self) -> None:
        """Silver 데이터 없으면 빈 DataFrame."""
        from src.data.onchain.service import OnchainDataService

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._catalog = None
        service._silver = _make_mock_silver({})

        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        result = service.precompute("BTC/USDT", ohlcv_index)

        assert result.columns.empty
        assert len(result) == 3
