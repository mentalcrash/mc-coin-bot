"""Tests for src/catalog/models.py."""

from __future__ import annotations

import pytest

from src.catalog.models import (
    DataCatalog,
    DatasetEntry,
    DataType,
    EnrichmentConfig,
    EnrichmentScope,
    SourceMeta,
)


class TestEnums:
    def test_data_type_values(self) -> None:
        assert DataType.OHLCV == "ohlcv"
        assert DataType.DERIVATIVES == "derivatives"
        assert DataType.ONCHAIN == "onchain"

    def test_enrichment_scope_values(self) -> None:
        assert EnrichmentScope.GLOBAL == "global"
        assert EnrichmentScope.ASSET == "asset"

    def test_data_type_from_string(self) -> None:
        assert DataType("ohlcv") == DataType.OHLCV
        assert DataType("onchain") == DataType.ONCHAIN

    def test_invalid_data_type(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            DataType("invalid")


class TestSourceMeta:
    def test_create(self) -> None:
        source = SourceMeta(
            id="defillama",
            name="DeFiLlama",
            api_url="https://api.llama.fi",
            date_column="date",
            lag_days=1,
            rate_limit_per_min=300,
        )
        assert source.id == "defillama"
        assert source.lag_days == 1

    def test_frozen(self) -> None:
        source = SourceMeta(id="test", name="Test")
        with pytest.raises(Exception):  # noqa: B017
            source.id = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        source = SourceMeta(id="test", name="Test")
        assert source.date_column == "date"
        assert source.lag_days == 0
        assert source.rate_limit_per_min == 0
        assert source.api_url == ""


class TestEnrichmentConfig:
    def test_create_global(self) -> None:
        config = EnrichmentConfig(
            scope=EnrichmentScope.GLOBAL,
            columns=["value"],
            rename_map={"value": "oc_value"},
        )
        assert config.scope == EnrichmentScope.GLOBAL
        assert config.target_assets == []

    def test_create_asset(self) -> None:
        config = EnrichmentConfig(
            scope=EnrichmentScope.ASSET,
            target_assets=["BTC", "ETH"],
            columns=["CapMVRVCur"],
            rename_map={"CapMVRVCur": "oc_mvrv"},
        )
        assert config.target_assets == ["BTC", "ETH"]

    def test_frozen(self) -> None:
        config = EnrichmentConfig(scope=EnrichmentScope.GLOBAL)
        with pytest.raises(Exception):  # noqa: B017
            config.scope = EnrichmentScope.ASSET  # type: ignore[misc]


class TestDatasetEntry:
    def test_create_minimal(self) -> None:
        ds = DatasetEntry(
            id="ohlcv_1m",
            name="OHLCV 1-Minute",
            data_type=DataType.OHLCV,
            source_id="binance_spot",
        )
        assert ds.id == "ohlcv_1m"
        assert ds.data_type == DataType.OHLCV
        assert ds.enrichment is None
        assert ds.strategy_hints == []

    def test_create_with_enrichment(self) -> None:
        ds = DatasetEntry(
            id="fear_greed",
            name="Fear & Greed",
            data_type=DataType.ONCHAIN,
            source_id="alternative_me",
            batch_group="sentiment",
            enrichment=EnrichmentConfig(
                scope=EnrichmentScope.GLOBAL,
                columns=["value"],
                rename_map={"value": "oc_fear_greed"},
            ),
        )
        assert ds.enrichment is not None
        assert ds.enrichment.scope == EnrichmentScope.GLOBAL

    def test_frozen(self) -> None:
        ds = DatasetEntry(
            id="test",
            name="Test",
            data_type=DataType.OHLCV,
            source_id="test",
        )
        with pytest.raises(Exception):  # noqa: B017
            ds.id = "changed"  # type: ignore[misc]


class TestDataCatalog:
    def test_create(self) -> None:
        catalog = DataCatalog(
            sources=[SourceMeta(id="s1", name="Source 1")],
            datasets=[
                DatasetEntry(
                    id="d1",
                    name="Dataset 1",
                    data_type=DataType.OHLCV,
                    source_id="s1",
                )
            ],
        )
        assert len(catalog.sources) == 1
        assert len(catalog.datasets) == 1

    def test_empty_catalog(self) -> None:
        catalog = DataCatalog()
        assert catalog.sources == []
        assert catalog.datasets == []
