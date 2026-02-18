"""Tests for CoinGecko service extensions."""

from src.data.macro.service import MACRO_BATCH_DEFINITIONS, SOURCE_DATE_COLUMNS, SOURCE_LAG_DAYS


class TestCoinGeckoServiceDefs:
    """CoinGecko batch definitions 테스트."""

    def test_coingecko_in_batch_definitions(self) -> None:
        """coingecko가 MACRO_BATCH_DEFINITIONS에 포함."""
        assert "coingecko" in MACRO_BATCH_DEFINITIONS
        defs = MACRO_BATCH_DEFINITIONS["coingecko"]
        assert len(defs) == 2
        sources = {s for s, _ in defs}
        assert sources == {"coingecko"}
        names = {n for _, n in defs}
        assert "global_metrics" in names
        assert "defi_global" in names

    def test_coingecko_lag_days(self) -> None:
        """coingecko lag days는 0."""
        assert SOURCE_LAG_DAYS["coingecko"] == 0

    def test_coingecko_date_column(self) -> None:
        """coingecko date column은 date."""
        assert SOURCE_DATE_COLUMNS["coingecko"] == "date"
