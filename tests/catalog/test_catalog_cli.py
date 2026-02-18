"""Tests for src/cli/catalog.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.catalog import app

_SAMPLE_YAML = {
    "sources": [
        {
            "id": "defillama",
            "name": "DeFiLlama",
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
    ],
    "datasets": [
        {
            "id": "stablecoin_total",
            "name": "Stablecoin Total",
            "data_type": "onchain",
            "source_id": "defillama",
            "batch_group": "stablecoin",
            "resolution": "1d",
            "columns": ["total_circulating_usd"],
            "strategy_hints": ["유동성 유입 지표"],
            "enrichment": {
                "scope": "global",
                "columns": ["total_circulating_usd"],
                "rename_map": {"total_circulating_usd": "oc_stablecoin_total_usd"},
            },
        },
        {
            "id": "btc_metrics",
            "name": "BTC Metrics",
            "data_type": "onchain",
            "source_id": "coinmetrics",
            "batch_group": "coinmetrics",
            "resolution": "1d",
            "columns": ["CapMVRVCur", "CapMrktCurUSD"],
            "strategy_hints": ["MVRV 기반 전략"],
        },
        {
            "id": "ohlcv_1m",
            "name": "OHLCV 1m",
            "data_type": "ohlcv",
            "source_id": "binance_spot",
            "batch_group": "ohlcv",
            "resolution": "1m",
            "columns": ["open", "high", "low", "close", "volume"],
        },
    ],
}

runner = CliRunner()


@pytest.fixture
def catalog_yaml_path(tmp_path: Path) -> Path:
    path = tmp_path / "datasets.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


class TestCatalogList:
    def test_list_all(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "3 datasets" in result.output
        assert "Stablecoin Total" in result.output
        assert "BTC Metrics" in result.output
        assert "OHLCV 1m" in result.output

    def test_list_by_type(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["list", "--type", "onchain"])
        assert result.exit_code == 0
        assert "2 datasets" in result.output
        assert "Stablecoin Total" in result.output
        assert "BTC Metrics" in result.output

    def test_list_by_group(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["list", "--group", "stablecoin"])
        assert result.exit_code == 0
        assert "1 datasets" in result.output
        assert "Stablecoin Total" in result.output

    def test_list_invalid_type(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["list", "--type", "invalid"])
        assert result.exit_code == 1
        assert "Invalid type" in result.output

    def test_list_empty_result(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["list", "--group", "nonexistent"])
        assert result.exit_code == 0
        assert "No datasets" in result.output


class TestCatalogShow:
    def test_show(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["show", "btc_metrics"])
        assert result.exit_code == 0
        assert "BTC Metrics" in result.output
        assert "coinmetrics" in result.output
        assert "CapMVRVCur" in result.output

    def test_show_with_enrichment(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["show", "stablecoin_total"])
        assert result.exit_code == 0
        assert "Enrichment" in result.output
        assert "global" in result.output

    def test_show_with_hints(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["show", "stablecoin_total"])
        assert result.exit_code == 0
        assert "Strategy Hints" in result.output

    def test_show_not_found(self, catalog_yaml_path: Path) -> None:
        with patch("src.cli.catalog.DataCatalogStore") as mock_cls:
            from src.catalog.store import DataCatalogStore

            mock_cls.return_value = DataCatalogStore(path=catalog_yaml_path)
            result = runner.invoke(app, ["show", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output
