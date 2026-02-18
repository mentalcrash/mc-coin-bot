"""Tests for src/data/onchain/service.py — OnchainDataService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.data.onchain.fetcher import BC_CHARTS, CM_ASSETS, DEFI_CHAINS, STABLECOIN_IDS
from src.data.onchain.service import (
    ONCHAIN_BATCH_DEFINITIONS,
    SOURCE_LAG_DAYS,
    OnchainDataService,
    get_date_col,
    route_fetch,
)

# ---------------------------------------------------------------------------
# Batch Definitions
# ---------------------------------------------------------------------------


class TestGetBatchDefinitions:
    def test_all_returns_all_definitions(self) -> None:
        """'all' 타입은 모든 카테고리의 definitions 합산."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("all")

        expected_count = sum(len(defs) for defs in ONCHAIN_BATCH_DEFINITIONS.values())
        assert len(result) == expected_count
        # 모든 항목이 (source, name) 튜플
        for source, name in result:
            assert isinstance(source, str)
            assert isinstance(name, str)

    def test_stablecoin_returns_stablecoin_only(self) -> None:
        """stablecoin 타입은 stablecoin definitions만 반환."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("stablecoin")

        # 1 total + chains + individual stablecoins
        expected = 1 + len(DEFI_CHAINS) + len(STABLECOIN_IDS)
        assert len(result) == expected
        for source, _name in result:
            assert source == "defillama"

    def test_tvl_returns_tvl_only(self) -> None:
        """tvl 타입은 TVL definitions만 반환."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("tvl")

        # 1 total + chains
        expected = 1 + len(DEFI_CHAINS)
        assert len(result) == expected

    def test_coinmetrics_returns_cm_only(self) -> None:
        """coinmetrics 타입은 CM definitions만."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("coinmetrics")

        assert len(result) == len(CM_ASSETS)
        for source, name in result:
            assert source == "coinmetrics"
            assert name.endswith("_metrics")

    def test_sentiment_returns_fear_greed(self) -> None:
        """sentiment 타입은 fear_greed definition만 반환."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("sentiment")

        assert len(result) == 1
        assert result[0] == ("alternative_me", "fear_greed")

    def test_blockchain_returns_bc_charts(self) -> None:
        """blockchain 타입은 3개 BC chart definitions 반환."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("blockchain")

        assert len(result) == len(BC_CHARTS)
        for source, name in result:
            assert source == "blockchain_com"
            assert name.startswith("bc_")

    def test_etherscan_returns_eth_supply(self) -> None:
        """etherscan 타입은 eth_supply definition 1개 반환."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        result = service.get_batch_definitions("etherscan")

        assert len(result) == 1
        assert result[0] == ("etherscan", "eth_supply")

    def test_invalid_type_raises(self) -> None:
        """존재하지 않는 batch type은 ValueError."""
        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = MagicMock()
        service._catalog = None

        with pytest.raises(ValueError, match="Unknown batch type"):
            service.get_batch_definitions("nonexistent")


# ---------------------------------------------------------------------------
# Route Fetch
# ---------------------------------------------------------------------------


class TestRouteFetchMetricsCallback:
    """route_fetch()에 metrics_callback 전달 시 동작 검증."""

    @pytest.mark.asyncio
    async def test_callback_success(self) -> None:
        """성공 fetch → callback.on_fetch(status="success")."""
        fetcher = MagicMock()
        fetcher.fetch_fear_greed = AsyncMock(return_value=pd.DataFrame({"value": [72]}))
        callback = MagicMock()

        await route_fetch(fetcher, "alternative_me", "fear_greed", metrics_callback=callback)

        callback.on_fetch.assert_called_once()
        args = callback.on_fetch.call_args
        assert args[0][0] == "alternative_me"  # source
        assert args[0][1] == "fear_greed"  # name
        assert args[0][3] == "success"  # status
        assert args[0][4] == 1  # row_count

    @pytest.mark.asyncio
    async def test_callback_empty(self) -> None:
        """빈 결과 → callback.on_fetch(status="empty")."""
        fetcher = MagicMock()
        fetcher.fetch_fear_greed = AsyncMock(return_value=pd.DataFrame())
        callback = MagicMock()

        await route_fetch(fetcher, "alternative_me", "fear_greed", metrics_callback=callback)

        callback.on_fetch.assert_called_once()
        assert callback.on_fetch.call_args[0][3] == "empty"
        assert callback.on_fetch.call_args[0][4] == 0

    @pytest.mark.asyncio
    async def test_callback_failure(self) -> None:
        """예외 발생 → callback.on_fetch(status="failure") 후 re-raise."""
        fetcher = MagicMock()
        callback = MagicMock()

        with pytest.raises(ValueError, match="Unknown route"):
            await route_fetch(fetcher, "unknown", "unknown", metrics_callback=callback)

        callback.on_fetch.assert_called_once()
        assert callback.on_fetch.call_args[0][3] == "failure"
        assert callback.on_fetch.call_args[0][4] == 0

    @pytest.mark.asyncio
    async def test_no_callback_works(self) -> None:
        """callback=None일 때 정상 동작."""
        fetcher = MagicMock()
        fetcher.fetch_fear_greed = AsyncMock(return_value=pd.DataFrame({"value": [72]}))

        result = await route_fetch(fetcher, "alternative_me", "fear_greed")
        assert len(result) == 1


class TestRouteFetch:
    @pytest.mark.asyncio
    async def test_stablecoin_total(self) -> None:
        """stablecoin_total → fetch_stablecoin_total()."""
        fetcher = MagicMock()
        fetcher.fetch_stablecoin_total = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        result = await route_fetch(fetcher, "defillama", "stablecoin_total")

        fetcher.fetch_stablecoin_total.assert_awaited_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_stablecoin_chain(self) -> None:
        """stablecoin_chain_Ethereum → fetch_stablecoin_by_chain('Ethereum')."""
        fetcher = MagicMock()
        fetcher.fetch_stablecoin_by_chain = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "defillama", "stablecoin_chain_Ethereum")

        fetcher.fetch_stablecoin_by_chain.assert_awaited_once_with("Ethereum")

    @pytest.mark.asyncio
    async def test_stablecoin_individual(self) -> None:
        """stablecoin_usdt → fetch_stablecoin_individual(1, 'USDT')."""
        fetcher = MagicMock()
        fetcher.fetch_stablecoin_individual = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "defillama", "stablecoin_usdt")

        fetcher.fetch_stablecoin_individual.assert_awaited_once_with(1, "USDT")

    @pytest.mark.asyncio
    async def test_tvl_total(self) -> None:
        """tvl_total → fetch_tvl()."""
        fetcher = MagicMock()
        fetcher.fetch_tvl = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "defillama", "tvl_total")

        fetcher.fetch_tvl.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_tvl_chain(self) -> None:
        """tvl_chain_Ethereum → fetch_tvl('Ethereum')."""
        fetcher = MagicMock()
        fetcher.fetch_tvl = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "defillama", "tvl_chain_Ethereum")

        fetcher.fetch_tvl.assert_awaited_once_with("Ethereum")

    @pytest.mark.asyncio
    async def test_dex_volume(self) -> None:
        """dex_volume → fetch_dex_volume()."""
        fetcher = MagicMock()
        fetcher.fetch_dex_volume = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "defillama", "dex_volume")

        fetcher.fetch_dex_volume.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_coinmetrics(self) -> None:
        """btc_metrics → fetch_coinmetrics('btc')."""
        fetcher = MagicMock()
        fetcher.fetch_coinmetrics = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "coinmetrics", "btc_metrics")

        fetcher.fetch_coinmetrics.assert_awaited_once_with("btc")

    @pytest.mark.asyncio
    async def test_fear_greed(self) -> None:
        """fear_greed → fetch_fear_greed()."""
        fetcher = MagicMock()
        fetcher.fetch_fear_greed = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "alternative_me", "fear_greed")

        fetcher.fetch_fear_greed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_alternative_me_name_raises(self) -> None:
        """alternative_me에서 알 수 없는 name은 ValueError."""
        fetcher = MagicMock()

        with pytest.raises(ValueError, match="Unknown route"):
            await route_fetch(fetcher, "alternative_me", "unknown_name")

    @pytest.mark.asyncio
    async def test_blockchain_chart(self) -> None:
        """bc_hash-rate → fetch_blockchain_chart('hash-rate')."""
        fetcher = MagicMock()
        fetcher.fetch_blockchain_chart = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        await route_fetch(fetcher, "blockchain_com", "bc_hash-rate")

        fetcher.fetch_blockchain_chart.assert_awaited_once_with("hash-rate")

    @pytest.mark.asyncio
    async def test_unknown_blockchain_com_name_raises(self) -> None:
        """blockchain_com에서 bc_ 접두사 없으면 ValueError."""
        fetcher = MagicMock()

        with pytest.raises(ValueError, match="Unknown route"):
            await route_fetch(fetcher, "blockchain_com", "unknown_name")

    @pytest.mark.asyncio
    async def test_etherscan_eth_supply(self) -> None:
        """etherscan/eth_supply → fetch_eth_supply(api_key)."""
        fetcher = MagicMock()
        fetcher.fetch_eth_supply = AsyncMock(return_value=pd.DataFrame({"a": [1]}))

        mock_settings = MagicMock()
        mock_settings.etherscan_api_key.get_secret_value.return_value = "test-key"

        with patch("src.data.onchain.service.get_settings", return_value=mock_settings):
            result = await route_fetch(fetcher, "etherscan", "eth_supply")

        fetcher.fetch_eth_supply.assert_awaited_once_with("test-key")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_unknown_etherscan_name_raises(self) -> None:
        """etherscan에서 알 수 없는 name은 ValueError."""
        fetcher = MagicMock()

        with pytest.raises(ValueError, match="Unknown route"):
            await route_fetch(fetcher, "etherscan", "unknown_name")

    @pytest.mark.asyncio
    async def test_unknown_source_raises(self) -> None:
        """알 수 없는 source/name 조합은 ValueError."""
        fetcher = MagicMock()

        with pytest.raises(ValueError, match="Unknown route"):
            await route_fetch(fetcher, "unknown_source", "unknown_name")

    @pytest.mark.asyncio
    async def test_unknown_defillama_name_raises(self) -> None:
        """defillama에서 알 수 없는 name은 ValueError."""
        fetcher = MagicMock()

        with pytest.raises(ValueError, match="Unknown route"):
            await route_fetch(fetcher, "defillama", "unknown_name")


# ---------------------------------------------------------------------------
# get_date_col
# ---------------------------------------------------------------------------


class TestGetDateCol:
    def test_defillama(self) -> None:
        assert get_date_col("defillama") == "date"

    def test_coinmetrics(self) -> None:
        assert get_date_col("coinmetrics") == "time"

    def test_alternative_me(self) -> None:
        assert get_date_col("alternative_me") == "timestamp"

    def test_blockchain_com(self) -> None:
        assert get_date_col("blockchain_com") == "timestamp"

    def test_etherscan(self) -> None:
        assert get_date_col("etherscan") == "timestamp"

    def test_unknown_source_defaults_to_date(self) -> None:
        assert get_date_col("unknown") == "date"


# ---------------------------------------------------------------------------
# Load & Enrich
# ---------------------------------------------------------------------------


class TestLoadSilver:
    def test_load_delegates_to_silver(self) -> None:
        """load()는 silver processor에 위임."""
        mock_silver = MagicMock()
        expected = pd.DataFrame({"date": [1, 2], "value": [10, 20]})
        mock_silver.load.return_value = expected

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.load("defillama", "stablecoin_total")

        mock_silver.load.assert_called_once_with("defillama", "stablecoin_total")
        pd.testing.assert_frame_equal(result, expected)


class TestEnrich:
    def test_enrich_merge_asof(self) -> None:
        """OHLCV + on-chain merge_asof 병합."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
            index=dates,
        )

        onchain_dates = pd.date_range("2024-01-01", periods=3, freq="2h", tz="UTC")
        onchain = pd.DataFrame(
            {
                "date": onchain_dates,
                "total_circulating_usd": [1e9, 1.1e9, 1.2e9],
                "source": ["defillama"] * 3,
            },
        )

        mock_silver = MagicMock()
        mock_silver.load.return_value = onchain

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.enrich(ohlcv, "defillama", "stablecoin_total")

        assert "total_circulating_usd" in result.columns
        assert "source" not in result.columns  # 메타 컬럼 제외
        assert len(result) == len(ohlcv)

    def test_enrich_empty_onchain_returns_original(self) -> None:
        """On-chain 데이터 없으면 원본 반환."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)

        mock_silver = MagicMock()
        mock_silver.load.return_value = pd.DataFrame()

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.enrich(ohlcv, "defillama", "stablecoin_total")

        pd.testing.assert_frame_equal(result, ohlcv)

    @patch("src.data.onchain.service.logger")
    def test_enrich_with_specific_columns(self, _mock_logger: MagicMock) -> None:
        """특정 컬럼만 병합."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=dates)

        onchain_dates = pd.date_range("2024-01-01", periods=2, freq="2h", tz="UTC")
        onchain = pd.DataFrame(
            {
                "date": onchain_dates,
                "total_circulating_usd": [1e9, 1.1e9],
                "tvl_usd": [5e10, 5.1e10],
                "source": ["defillama"] * 2,
            },
        )

        mock_silver = MagicMock()
        mock_silver.load.return_value = onchain

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.enrich(
            ohlcv, "defillama", "stablecoin_total", columns=["total_circulating_usd"]
        )

        assert "total_circulating_usd" in result.columns
        assert "tvl_usd" not in result.columns

    def test_enrich_applies_lag_shift(self) -> None:
        """defillama(lag=1) 데이터가 T+1에서야 접근 가능."""
        # OHLCV: Jan 1 ~ Jan 3 daily
        ohlcv_dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=ohlcv_dates)

        # On-chain: Jan 1 ~ Jan 3 daily (defillama → lag=1)
        onchain_dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        onchain = pd.DataFrame(
            {
                "date": onchain_dates,
                "total_circulating_usd": [1e9, 1.1e9, 1.2e9],
                "source": ["defillama"] * 3,
            },
        )

        mock_silver = MagicMock()
        mock_silver.load.return_value = onchain

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.enrich(ohlcv, "defillama", "stablecoin_total")

        # lag=1이므로 Jan 1 on-chain data는 Jan 2에서야 접근 가능
        # Jan 1 OHLCV → on-chain NaN (아직 publish 안 됨)
        assert pd.isna(result.iloc[0]["total_circulating_usd"])
        # Jan 2 OHLCV → Jan 1 on-chain (shifted to Jan 2)
        assert result.iloc[1]["total_circulating_usd"] == 1e9

    def test_enrich_no_lag_for_realtime(self) -> None:
        """etherscan(lag=0) shift 없음."""
        ohlcv_dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=ohlcv_dates)

        onchain_dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        onchain = pd.DataFrame(
            {
                "timestamp": onchain_dates,
                "eth_supply": [120e6, 120.1e6, 120.2e6],
                "eth2_staking": [30e6, 30.1e6, 30.2e6],
                "burnt_fees": [3e6, 3.01e6, 3.02e6],
                "withdrawn_total": [1e6, 1.01e6, 1.02e6],
                "source": ["etherscan"] * 3,
            },
        )

        mock_silver = MagicMock()
        mock_silver.load.return_value = onchain

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.enrich(ohlcv, "etherscan", "eth_supply")

        assert SOURCE_LAG_DAYS["etherscan"] == 0
        # lag=0이므로 Jan 1 OHLCV에서 Jan 1 on-chain 데이터 접근 가능
        assert result.iloc[0]["eth_supply"] == 120e6

    def test_enrich_custom_lag_override(self) -> None:
        """lag_days=2 명시 시 기본값 무시."""
        ohlcv_dates = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        ohlcv = pd.DataFrame({"close": [100.0] * 5}, index=ohlcv_dates)

        onchain_dates = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
        onchain = pd.DataFrame(
            {
                "date": onchain_dates,
                "total_circulating_usd": [1e9, 1.1e9, 1.2e9],
                "source": ["defillama"] * 3,
            },
        )

        mock_silver = MagicMock()
        mock_silver.load.return_value = onchain

        service = OnchainDataService.__new__(OnchainDataService)
        service._settings = MagicMock()
        service._silver = mock_silver

        result = service.enrich(ohlcv, "defillama", "stablecoin_total", lag_days=2)

        # lag=2이므로 Jan 1 data는 Jan 3에서야 접근 가능
        assert pd.isna(result.iloc[0]["total_circulating_usd"])  # Jan 1
        assert pd.isna(result.iloc[1]["total_circulating_usd"])  # Jan 2
        assert result.iloc[2]["total_circulating_usd"] == 1e9  # Jan 3
