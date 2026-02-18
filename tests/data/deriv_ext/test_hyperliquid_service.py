"""Tests for Hyperliquid service extensions."""

from src.data.deriv_ext.service import (
    ASSET_PRECOMPUTE_DEFS,
    DERIV_EXT_BATCH_DEFINITIONS,
    SOURCE_DATE_COLUMNS,
    SOURCE_LAG_DAYS,
)


class TestHyperliquidServiceDefs:
    """Hyperliquid batch definitions 테스트."""

    def test_hyperliquid_in_batch_definitions(self) -> None:
        """hyperliquid가 DERIV_EXT_BATCH_DEFINITIONS에 포함."""
        assert "hyperliquid" in DERIV_EXT_BATCH_DEFINITIONS
        defs = DERIV_EXT_BATCH_DEFINITIONS["hyperliquid"]
        assert len(defs) == 2
        sources = {s for s, _ in defs}
        assert sources == {"hyperliquid"}
        names = {n for _, n in defs}
        assert "hl_asset_contexts" in names
        assert "hl_predicted_fundings" in names

    def test_hyperliquid_lag_days(self) -> None:
        """hyperliquid lag days는 0."""
        assert SOURCE_LAG_DAYS["hyperliquid"] == 0

    def test_hyperliquid_date_column(self) -> None:
        """hyperliquid date column은 date."""
        assert SOURCE_DATE_COLUMNS["hyperliquid"] == "date"

    def test_hyperliquid_in_precompute_defs(self) -> None:
        """Hyperliquid asset_contexts가 precompute에 포함."""
        for asset in ("BTC", "ETH"):
            defs = ASSET_PRECOMPUTE_DEFS[asset]
            hl_defs = [d for d in defs if d[0] == "hyperliquid"]
            assert len(hl_defs) == 1
            _source, name, columns, rename_map = hl_defs[0]
            assert name == "hl_asset_contexts"
            assert "open_interest" in columns
            assert "funding" in columns
            assert rename_map["open_interest"] == "dext_hl_oi"
            assert rename_map["funding"] == "dext_hl_funding"
