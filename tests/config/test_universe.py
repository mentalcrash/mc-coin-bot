"""Tests for src/config/universe.py — Asset Universe 상수 검증."""

from __future__ import annotations

from src.config.universe import (
    ALL_SYMBOLS,
    CURRENT_YEAR,
    DERIV_START_YEAR,
    SYMBOL_RENAME_MAP,
    TIER1_OHLCV_START_YEAR,
    TIER1_SYMBOLS,
    TIER2_OHLCV_START_YEAR,
    TIER2_SYMBOLS,
)


class TestTierSymbols:
    def test_tier1_count(self) -> None:
        assert len(TIER1_SYMBOLS) == 8

    def test_tier2_count(self) -> None:
        assert len(TIER2_SYMBOLS) == 8

    def test_all_symbols_is_union(self) -> None:
        assert ALL_SYMBOLS == TIER1_SYMBOLS + TIER2_SYMBOLS

    def test_all_symbols_count(self) -> None:
        assert len(ALL_SYMBOLS) == 16

    def test_no_duplicates(self) -> None:
        assert len(set(ALL_SYMBOLS)) == len(ALL_SYMBOLS)

    def test_all_usdt_pairs(self) -> None:
        for sym in ALL_SYMBOLS:
            assert sym.endswith("/USDT"), f"{sym} is not a USDT pair"

    def test_symbol_format(self) -> None:
        """All symbols follow BASE/USDT format."""
        for sym in ALL_SYMBOLS:
            parts = sym.split("/")
            assert len(parts) == 2, f"Invalid format: {sym}"
            assert parts[0].isalpha(), f"Base not alphabetic: {sym}"
            assert parts[1] == "USDT"

    def test_tier1_contains_expected(self) -> None:
        expected = {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
        assert expected.issubset(set(TIER1_SYMBOLS))

    def test_tier2_contains_expected(self) -> None:
        expected = {"XRP/USDT", "DOT/USDT", "POL/USDT", "LTC/USDT"}
        assert expected.issubset(set(TIER2_SYMBOLS))

    def test_no_overlap(self) -> None:
        assert set(TIER1_SYMBOLS).isdisjoint(set(TIER2_SYMBOLS))

    def test_tuples_are_immutable(self) -> None:
        assert isinstance(TIER1_SYMBOLS, tuple)
        assert isinstance(TIER2_SYMBOLS, tuple)
        assert isinstance(ALL_SYMBOLS, tuple)


class TestSymbolRenameMap:
    def test_pol_rename_entry(self) -> None:
        assert "POL/USDT" in SYMBOL_RENAME_MAP
        old_sym, cutover_date = SYMBOL_RENAME_MAP["POL/USDT"]
        assert old_sym == "MATIC/USDT"
        assert cutover_date == "2024-09-10"

    def test_rename_keys_in_all_symbols(self) -> None:
        for key in SYMBOL_RENAME_MAP:
            assert key in ALL_SYMBOLS


class TestYearConstants:
    def test_tier1_start_before_tier2(self) -> None:
        assert TIER1_OHLCV_START_YEAR < TIER2_OHLCV_START_YEAR

    def test_current_year(self) -> None:
        assert CURRENT_YEAR == 2026

    def test_deriv_start(self) -> None:
        assert DERIV_START_YEAR == 2020
