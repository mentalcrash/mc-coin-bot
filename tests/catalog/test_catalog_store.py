"""Tests for src/catalog/store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.catalog.models import DataType
from src.catalog.store import DataCatalogStore

_SAMPLE_YAML = {
    "sources": [
        {
            "id": "defillama",
            "name": "DeFiLlama",
            "api_url": "https://api.llama.fi",
            "date_column": "date",
            "lag_days": 1,
            "rate_limit_per_min": 300,
        },
        {
            "id": "coinmetrics",
            "name": "Coin Metrics",
            "date_column": "time",
            "lag_days": 1,
        },
        {
            "id": "alternative_me",
            "name": "Alternative.me",
            "date_column": "timestamp",
            "lag_days": 1,
        },
        {
            "id": "fred",
            "name": "FRED",
            "date_column": "date",
            "lag_days": 1,
        },
    ],
    "datasets": [
        {
            "id": "stablecoin_total",
            "name": "Stablecoin Total",
            "data_type": "onchain",
            "source_id": "defillama",
            "batch_group": "stablecoin",
            "columns": ["total_circulating_usd"],
            "fetch_key": "stablecoin_total",
            "strategy_hints": ["유동성 유입 지표"],
            "enrichment": {
                "scope": "global",
                "columns": ["total_circulating_usd"],
                "rename_map": {"total_circulating_usd": "oc_stablecoin_total_usd"},
            },
        },
        {
            "id": "tvl_total",
            "name": "DeFi TVL",
            "data_type": "onchain",
            "source_id": "defillama",
            "batch_group": "tvl",
            "columns": ["tvl_usd"],
            "fetch_key": "tvl_total",
            "enrichment": {
                "scope": "global",
                "columns": ["tvl_usd"],
                "rename_map": {"tvl_usd": "oc_tvl_usd"},
            },
        },
        {
            "id": "fear_greed",
            "name": "Fear & Greed",
            "data_type": "onchain",
            "source_id": "alternative_me",
            "batch_group": "sentiment",
            "columns": ["value"],
            "fetch_key": "fear_greed",
            "enrichment": {
                "scope": "global",
                "columns": ["value"],
                "rename_map": {"value": "oc_fear_greed"},
            },
        },
        {
            "id": "btc_metrics",
            "name": "BTC Metrics",
            "data_type": "onchain",
            "source_id": "coinmetrics",
            "batch_group": "coinmetrics",
            "columns": ["CapMVRVCur", "CapMrktCurUSD"],
            "fetch_key": "btc_metrics",
            "enrichment": {
                "scope": "asset",
                "target_assets": ["BTC"],
                "columns": ["CapMVRVCur", "CapMrktCurUSD"],
                "rename_map": {
                    "CapMVRVCur": "oc_mvrv",
                    "CapMrktCurUSD": "oc_mktcap_usd",
                },
            },
        },
        {
            "id": "ohlcv_1m",
            "name": "OHLCV",
            "data_type": "ohlcv",
            "source_id": "binance_spot",
            "batch_group": "ohlcv",
            "columns": ["open", "high", "low", "close", "volume"],
        },
        {
            "id": "fred_m2",
            "name": "M2 Money Supply",
            "data_type": "macro",
            "source_id": "fred",
            "batch_group": "fred",
            "columns": ["value"],
            "fetch_key": "m2",
            "lag_days": 14,
        },
    ],
}


@pytest.fixture
def catalog_yaml_path(tmp_path: Path) -> Path:
    """임시 datasets.yaml 작성."""
    path = tmp_path / "datasets.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def catalog_store(catalog_yaml_path: Path) -> DataCatalogStore:
    return DataCatalogStore(path=catalog_yaml_path)


class TestDataCatalogStore:
    def test_load_all_count(self, catalog_store: DataCatalogStore) -> None:
        datasets = catalog_store.load_all()
        assert len(datasets) == 6

    def test_load_single(self, catalog_store: DataCatalogStore) -> None:
        ds = catalog_store.load("btc_metrics")
        assert ds.name == "BTC Metrics"
        assert ds.data_type == DataType.ONCHAIN
        assert ds.source_id == "coinmetrics"

    def test_load_not_found(self, catalog_store: DataCatalogStore) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            catalog_store.load("nonexistent")

    def test_file_not_found(self, tmp_path: Path) -> None:
        store = DataCatalogStore(path=tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            store.load_all()

    def test_get_source(self, catalog_store: DataCatalogStore) -> None:
        source = catalog_store.get_source("defillama")
        assert source.name == "DeFiLlama"
        assert source.date_column == "date"
        assert source.lag_days == 1

    def test_get_source_not_found(self, catalog_store: DataCatalogStore) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            catalog_store.get_source("nonexistent")

    def test_get_all_sources(self, catalog_store: DataCatalogStore) -> None:
        sources = catalog_store.get_all_sources()
        assert len(sources) == 4

    def test_get_by_type(self, catalog_store: DataCatalogStore) -> None:
        onchain = catalog_store.get_by_type(DataType.ONCHAIN)
        assert len(onchain) == 4
        ohlcv = catalog_store.get_by_type(DataType.OHLCV)
        assert len(ohlcv) == 1

    def test_get_by_group(self, catalog_store: DataCatalogStore) -> None:
        stablecoin = catalog_store.get_by_group("stablecoin")
        assert len(stablecoin) == 1
        assert stablecoin[0].id == "stablecoin_total"

    def test_get_by_group_empty(self, catalog_store: DataCatalogStore) -> None:
        result = catalog_store.get_by_group("nonexistent_group")
        assert result == []

    def test_cache_hit(self, catalog_store: DataCatalogStore) -> None:
        catalog_store.load_all()
        assert catalog_store._catalog is not None
        datasets = catalog_store.load_all()
        assert len(datasets) == 6


class TestCompatAPI:
    """호환 API 테스트 (Phase 2 연동용)."""

    def test_get_batch_definitions(self, catalog_store: DataCatalogStore) -> None:
        defs = catalog_store.get_batch_definitions("stablecoin")
        assert len(defs) == 1
        assert defs[0] == ("defillama", "stablecoin_total")

    def test_get_batch_definitions_all(self, catalog_store: DataCatalogStore) -> None:
        defs = catalog_store.get_batch_definitions("all")
        assert len(defs) == 4  # 4 onchain datasets

    def test_get_date_col(self, catalog_store: DataCatalogStore) -> None:
        assert catalog_store.get_date_col("defillama") == "date"
        assert catalog_store.get_date_col("coinmetrics") == "time"
        assert catalog_store.get_date_col("alternative_me") == "timestamp"

    def test_get_lag_days(self, catalog_store: DataCatalogStore) -> None:
        assert catalog_store.get_lag_days("defillama") == 1
        assert catalog_store.get_lag_days("coinmetrics") == 1
        assert catalog_store.get_lag_days("fred") == 1

    def test_get_lag_days_dataset_override(self, catalog_store: DataCatalogStore) -> None:
        """Dataset-level lag_days가 source-level보다 우선."""
        # fred source: lag_days=1, fred_m2 dataset: lag_days=14
        assert catalog_store.get_lag_days("fred", dataset_id="fred_m2") == 14
        # dataset에 lag_days 없으면 source fallback
        assert catalog_store.get_lag_days("defillama", dataset_id="stablecoin_total") == 1
        # 존재하지 않는 dataset → source fallback
        assert catalog_store.get_lag_days("fred", dataset_id="nonexistent") == 1

    def test_build_precompute_map_global(self, catalog_store: DataCatalogStore) -> None:
        result = catalog_store.build_precompute_map(["SOL/USDT"])
        assert "SOL/USDT" in result
        sources = result["SOL/USDT"]
        # SOL should get global sources only (stablecoin_total, tvl_total, fear_greed)
        assert len(sources) == 3
        source_names = [s[1] for s in sources]
        assert "stablecoin_total" in source_names
        assert "tvl_total" in source_names
        assert "fear_greed" in source_names

    def test_build_precompute_map_btc(self, catalog_store: DataCatalogStore) -> None:
        result = catalog_store.build_precompute_map(["BTC/USDT"])
        sources = result["BTC/USDT"]
        # BTC gets global (3) + asset-specific (btc_metrics)
        assert len(sources) == 4
        source_names = [s[1] for s in sources]
        assert "btc_metrics" in source_names

    def test_build_precompute_map_rename(self, catalog_store: DataCatalogStore) -> None:
        result = catalog_store.build_precompute_map(["BTC/USDT"])
        btc_sources = result["BTC/USDT"]
        # Find btc_metrics entry
        btc_metrics = next(s for s in btc_sources if s[1] == "btc_metrics")
        _, _, columns, rename_map = btc_metrics
        assert "CapMVRVCur" in columns
        assert rename_map["CapMVRVCur"] == "oc_mvrv"

    def test_build_precompute_map_multiple_symbols(self, catalog_store: DataCatalogStore) -> None:
        result = catalog_store.build_precompute_map(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        assert len(result) == 3
        assert len(result["BTC/USDT"]) == 4
        assert len(result["ETH/USDT"]) == 3  # No ETH-specific in sample
        assert len(result["SOL/USDT"]) == 3


class TestRealCatalogYaml:
    """실제 catalogs/datasets.yaml 로드 테스트."""

    def test_load_real_yaml(self) -> None:
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)
        datasets = store.load_all()
        assert len(datasets) >= 25

    def test_real_yaml_sources(self) -> None:
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)
        sources = store.get_all_sources()
        ids = [s.id for s in sources]
        assert "defillama" in ids
        assert "coinmetrics" in ids
        assert "alternative_me" in ids

    def test_real_yaml_batch_definitions_compat(self) -> None:
        """ONCHAIN_BATCH_DEFINITIONS와 호환성 검증."""
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)

        stablecoin_defs = store.get_batch_definitions("stablecoin")
        assert len(stablecoin_defs) >= 8
        assert ("defillama", "stablecoin_total") in stablecoin_defs

    def test_real_yaml_date_col_compat(self) -> None:
        """SOURCE_DATE_COLUMNS와 호환성 검증."""
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)

        assert store.get_date_col("defillama") == "date"
        assert store.get_date_col("coinmetrics") == "time"
        assert store.get_date_col("alternative_me") == "timestamp"
        assert store.get_date_col("blockchain_com") == "timestamp"
        assert store.get_date_col("etherscan") == "timestamp"
        assert store.get_date_col("mempool_space") == "timestamp"

    def test_real_yaml_lag_days_compat(self) -> None:
        """SOURCE_LAG_DAYS와 호환성 검증."""
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)

        assert store.get_lag_days("defillama") == 1
        assert store.get_lag_days("coinmetrics") == 1
        assert store.get_lag_days("alternative_me") == 1
        assert store.get_lag_days("blockchain_com") == 1
        assert store.get_lag_days("etherscan") == 0
        assert store.get_lag_days("mempool_space") == 0

    def test_real_yaml_m2_dataset_lag(self) -> None:
        """fred_m2 dataset-level lag_days=14 검증."""
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)

        # fred source: lag_days=1, fred_m2 dataset: lag_days=14
        assert store.get_lag_days("fred") == 1
        assert store.get_lag_days("fred", dataset_id="fred_m2") == 14

    def test_real_yaml_precompute_map_compat(self) -> None:
        """build_precompute_map과 _GLOBAL_SOURCES/_ASSET_SOURCES 호환성 검증."""
        real_path = Path("catalogs/datasets.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/datasets.yaml not found")
        store = DataCatalogStore(path=real_path)

        result = store.build_precompute_map(["BTC/USDT", "ETH/USDT", "SOL/USDT"])

        # BTC should have global + BTC-specific
        btc_sources = result["BTC/USDT"]
        btc_names = [s[1] for s in btc_sources]
        assert "stablecoin_total" in btc_names
        assert "tvl_total" in btc_names
        assert "dex_volume" in btc_names
        assert "fear_greed" in btc_names
        assert "btc_metrics" in btc_names
        assert "bc_hash-rate" in btc_names
        assert "mining" in btc_names

        # ETH should have global + ETH-specific
        eth_sources = result["ETH/USDT"]
        eth_names = [s[1] for s in eth_sources]
        assert "eth_metrics" in eth_names
        assert "eth_supply" in eth_names

        # SOL should have only global sources
        sol_sources = result["SOL/USDT"]
        sol_names = [s[1] for s in sol_sources]
        assert "stablecoin_total" in sol_names
        assert "btc_metrics" not in sol_names
