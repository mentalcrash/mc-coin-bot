"""Tests for src/pipeline/ic_scanner.py — IC Batch Scanner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.ic_scanner import (
    ICBatchScanner,
    ScanConfig,
    ScanEntry,
    ScanReport,
    _detect_source,
)


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + abs(rng.normal(0, 1, n)),
            "low": close - abs(rng.normal(0, 1, n)),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        },
        index=dates,
    )


class TestScanConfig:
    def test_default_values(self) -> None:
        config = ScanConfig()
        assert len(config.symbols) == 5
        assert config.timeframe == "1D"
        assert config.min_data_coverage == 0.5

    def test_custom_config(self) -> None:
        config = ScanConfig(symbols=("BTC/USDT",), forward_periods=(1, 5))
        assert len(config.symbols) == 1
        assert config.forward_periods == (1, 5)

    def test_frozen(self) -> None:
        config = ScanConfig()
        with pytest.raises(AttributeError):
            config.timeframe = "4H"  # type: ignore[misc]


class TestICBatchScanner:
    def test_scan_ohlcv_returns_entries(self) -> None:
        scanner = ICBatchScanner()
        data = {"BTC/USDT": _make_ohlcv()}
        entries = scanner.scan_ohlcv_indicators(data)
        assert len(entries) > 0
        assert all(isinstance(e, ScanEntry) for e in entries)

    def test_scan_ohlcv_source_is_ohlcv(self) -> None:
        scanner = ICBatchScanner()
        entries = scanner.scan_ohlcv_indicators({"BTC/USDT": _make_ohlcv()})
        assert all(e.source == "ohlcv" for e in entries)

    def test_scan_enriched_detects_sources(self) -> None:
        scanner = ICBatchScanner()
        df = _make_ohlcv()
        rng = np.random.default_rng(123)
        df["oc_fear_greed"] = rng.uniform(0, 100, len(df))
        df["macro_dxy"] = rng.normal(100, 5, len(df))
        entries = scanner.scan_enriched_columns({"BTC/USDT": df})
        sources = {e.source for e in entries}
        assert "onchain" in sources
        assert "macro" in sources

    def test_scan_enriched_skips_ohlcv_columns(self) -> None:
        scanner = ICBatchScanner()
        entries = scanner.scan_enriched_columns({"BTC/USDT": _make_ohlcv()})
        names = {e.indicator_name for e in entries}
        assert "close" not in names
        assert "volume" not in names

    def test_scan_all_combines_results(self) -> None:
        scanner = ICBatchScanner()
        ohlcv = {"BTC/USDT": _make_ohlcv()}
        df = _make_ohlcv()
        rng = np.random.default_rng(99)
        df["oc_mvrv"] = rng.normal(2, 0.5, len(df))
        enriched = {"BTC/USDT": df}
        report = scanner.scan_all(ohlcv, enriched)
        assert isinstance(report, ScanReport)
        ohlcv_entries = [e for e in report.entries if e.source == "ohlcv"]
        onchain_entries = [e for e in report.entries if e.source == "onchain"]
        assert len(ohlcv_entries) > 0
        assert len(onchain_entries) > 0

    def test_low_coverage_skipped(self) -> None:
        scanner = ICBatchScanner(ScanConfig(min_data_coverage=0.99))
        df = _make_ohlcv(n=100)
        # Add mostly NaN enriched column
        df["oc_test"] = np.nan
        df.iloc[:5, df.columns.get_loc("oc_test")] = 1.0
        entries = scanner.scan_enriched_columns({"BTC/USDT": df})
        oc_test = [e for e in entries if e.indicator_name == "oc_test"]
        assert len(oc_test) == 1
        assert oc_test[0].error is not None
        assert "coverage" in oc_test[0].error.lower()

    def test_scan_ohlcv_without_enriched(self) -> None:
        scanner = ICBatchScanner()
        report = scanner.scan_all({"BTC/USDT": _make_ohlcv()}, enriched_data=None)
        assert report.total > 0
        assert all(e.source == "ohlcv" for e in report.entries)


class TestScanReport:
    @pytest.fixture()
    def report(self) -> ScanReport:
        scanner = ICBatchScanner()
        data = {"BTC/USDT": _make_ohlcv(), "ETH/USDT": _make_ohlcv(seed=99)}
        return scanner.scan_all(data)

    def test_total_matches_entries(self, report: ScanReport) -> None:
        assert report.total == len(report.entries)

    def test_passed_failed_skipped_sum(self, report: ScanReport) -> None:
        assert report.passed + report.failed + report.skipped == report.total

    def test_top_n_returns_sorted(self, report: ScanReport) -> None:
        top = report.top_n(5)
        assert len(top) <= 5
        ics = [abs(e.ic_result.rank_ic) for e in top if e.ic_result]
        assert ics == sorted(ics, reverse=True)

    def test_cross_asset_stable(self, report: ScanReport) -> None:
        stable = report.cross_asset_stable(min_assets=2)
        # All returned entries should have ic_result
        assert all(e.ic_result is not None for e in stable)


class TestDetectSource:
    def test_onchain_prefix(self) -> None:
        assert _detect_source("oc_fear_greed") == "onchain"

    def test_macro_prefix(self) -> None:
        assert _detect_source("macro_dxy") == "macro"

    def test_options_prefix(self) -> None:
        assert _detect_source("opt_iv_30d") == "options"

    def test_derivatives_prefixes(self) -> None:
        assert _detect_source("oi_momentum") == "derivatives"
        assert _detect_source("funding_rate") == "derivatives"
        assert _detect_source("basis_spread") == "derivatives"

    def test_unknown_prefix(self) -> None:
        assert _detect_source("custom_indicator") == "unknown"
