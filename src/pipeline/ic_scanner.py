"""IC Batch Scanner — 다중 지표 x 다중 에셋 IC 일괄 스캔.

기존 compute_indicator()와 ICAnalyzer.analyze()를 재사용하여
OHLCV 기반 지표 + enriched 데이터 컬럼의 예측력을 일괄 측정합니다.

Design Principles:
    - I/O 없음: 데이터 로딩은 CLI 책임
    - Frozen dataclass: 불변 결과 객체
    - Cross-asset: 다중 에셋 안정성 필터링

Usage:
    >>> scanner = ICBatchScanner()
    >>> report = scanner.scan_all(ohlcv_data, enriched_data)
    >>> for entry in report.top_n(20):
    ...     print(entry.indicator_name, entry.ic_result.rank_ic)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.backtest.ic_analyzer import ICResult

# ---------------------------------------------------------------------------
# Enriched column source detection
# ---------------------------------------------------------------------------

ENRICHED_PREFIXES: dict[str, str] = {
    "oc_": "onchain",
    "macro_": "macro",
    "opt_": "options",
    "deriv_": "derivatives",
    "oi_": "derivatives",
    "funding_": "derivatives",
    "basis_": "derivatives",
    "ls_": "derivatives",
    "liq_": "derivatives",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "timestamp", "date"})


@dataclass(frozen=True)
class ScanConfig:
    """IC Batch Scanner 설정."""

    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT")
    timeframe: str = "1D"
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
    forward_periods: tuple[int, ...] = (1,)
    min_data_coverage: float = 0.5


@dataclass(frozen=True)
class ScanEntry:
    """IC Batch Scanner 개별 결과."""

    indicator_name: str
    source: str  # "ohlcv" | "derivatives" | "onchain" | "macro" | "options"
    symbol: str
    forward_period: int
    ic_result: ICResult | None
    error: str | None = None


@dataclass
class ScanReport:
    """IC Batch Scanner 전체 리포트."""

    entries: list[ScanEntry]
    config: ScanConfig

    @property
    def total(self) -> int:
        """전체 항목 수."""
        return len(self.entries)

    @property
    def passed(self) -> int:
        """IC PASS 항목 수."""
        return sum(1 for e in self.entries if e.ic_result and e.ic_result.verdict.value == "PASS")

    @property
    def failed(self) -> int:
        """IC FAIL 항목 수."""
        return sum(1 for e in self.entries if e.ic_result and e.ic_result.verdict.value == "FAIL")

    @property
    def skipped(self) -> int:
        """에러로 스킵된 항목 수."""
        return sum(1 for e in self.entries if e.error is not None)

    def top_n(self, n: int = 20) -> list[ScanEntry]:
        """상위 N개 항목 (|Rank IC| 기준 내림차순)."""
        valid = [e for e in self.entries if e.ic_result is not None]
        valid.sort(key=lambda e: abs(e.ic_result.rank_ic), reverse=True)  # type: ignore[union-attr]
        return valid[:n]

    def cross_asset_stable(self, min_assets: int = 3) -> list[ScanEntry]:
        """다중 에셋에서 IC PASS인 지표만 필터."""
        pass_counts: Counter[str] = Counter()
        for e in self.entries:
            if e.ic_result and e.ic_result.verdict.value == "PASS":
                pass_counts[e.indicator_name] += 1

        stable_indicators = {name for name, count in pass_counts.items() if count >= min_assets}
        return [
            e
            for e in self.entries
            if e.indicator_name in stable_indicators and e.ic_result is not None
        ]


# ---------------------------------------------------------------------------
# ICBatchScanner
# ---------------------------------------------------------------------------


class ICBatchScanner:
    """IC Batch Scanner — 다중 지표 x 다중 에셋 IC 일괄 스캔."""

    def __init__(self, config: ScanConfig | None = None) -> None:
        self._config = config or ScanConfig()

    @property
    def config(self) -> ScanConfig:
        """스캔 설정."""
        return self._config

    def scan_ohlcv_indicators(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        forward_periods: tuple[int, ...] | None = None,
    ) -> list[ScanEntry]:
        """OHLCV 기반 지표 일괄 IC 스캔.

        DEFAULT_SPECS의 모든 지표에 대해 IC를 계산합니다.

        Args:
            ohlcv_data: {symbol: OHLCV DataFrame} mapping
            forward_periods: forward return 기간 (default: config)

        Returns:
            ScanEntry 리스트
        """
        from src.backtest.ic_analyzer import ICAnalyzer
        from src.market.feature_store import DEFAULT_SPECS, compute_indicator

        periods = forward_periods or self._config.forward_periods
        entries: list[ScanEntry] = []

        for symbol, df in ohlcv_data.items():
            close: pd.Series = df["close"]  # type: ignore[assignment]
            for spec in DEFAULT_SPECS:
                for fwd in periods:
                    fwd_ret: pd.Series = close.pct_change(fwd).shift(-fwd)  # type: ignore[assignment]
                    try:
                        indicator = compute_indicator(spec, df)
                        # Check min data coverage
                        valid_ratio = (
                            float(indicator.notna().sum()) / len(df) if len(df) > 0 else 0.0
                        )
                        if valid_ratio < self._config.min_data_coverage:
                            entries.append(
                                ScanEntry(
                                    indicator_name=spec.name,
                                    source="ohlcv",
                                    symbol=symbol,
                                    forward_period=fwd,
                                    ic_result=None,
                                    error=f"Insufficient data coverage: {valid_ratio:.1%}",
                                )
                            )
                            continue
                        result = ICAnalyzer.analyze(indicator, fwd_ret)
                        entries.append(
                            ScanEntry(
                                indicator_name=spec.name,
                                source="ohlcv",
                                symbol=symbol,
                                forward_period=fwd,
                                ic_result=result,
                            )
                        )
                    except Exception as exc:
                        entries.append(
                            ScanEntry(
                                indicator_name=spec.name,
                                source="ohlcv",
                                symbol=symbol,
                                forward_period=fwd,
                                ic_result=None,
                                error=str(exc),
                            )
                        )
        return entries

    def scan_enriched_columns(
        self,
        enriched_data: dict[str, pd.DataFrame],
        forward_periods: tuple[int, ...] | None = None,
    ) -> list[ScanEntry]:
        """Enriched DataFrame의 비-OHLCV 컬럼 자동 감지 및 IC 스캔.

        prefix로 데이터 소스를 감지합니다: oc_*, macro_*, opt_*, deriv_*,
        oi_*, funding_*, basis_*, ls_*, liq_*

        Args:
            enriched_data: {symbol: enriched DataFrame} mapping
            forward_periods: forward return 기간 (default: config)

        Returns:
            ScanEntry 리스트
        """
        from src.backtest.ic_analyzer import ICAnalyzer

        periods = forward_periods or self._config.forward_periods
        entries: list[ScanEntry] = []

        for symbol, df in enriched_data.items():
            close: pd.Series = df["close"]  # type: ignore[assignment]
            enriched_cols = [
                c for c in df.columns if c not in _OHLCV_COLUMNS and not c.startswith("_")
            ]

            for col in enriched_cols:
                source = _detect_source(col)
                for fwd in periods:
                    fwd_ret: pd.Series = close.pct_change(fwd).shift(-fwd)  # type: ignore[assignment]
                    try:
                        ind_series: pd.Series = df[col]  # type: ignore[assignment]
                        valid_ratio = (
                            float(ind_series.notna().sum()) / len(df) if len(df) > 0 else 0.0
                        )
                        if valid_ratio < self._config.min_data_coverage:
                            entries.append(
                                ScanEntry(
                                    indicator_name=col,
                                    source=source,
                                    symbol=symbol,
                                    forward_period=fwd,
                                    ic_result=None,
                                    error=f"Insufficient data coverage: {valid_ratio:.1%}",
                                )
                            )
                            continue
                        result = ICAnalyzer.analyze(ind_series, fwd_ret)
                        entries.append(
                            ScanEntry(
                                indicator_name=col,
                                source=source,
                                symbol=symbol,
                                forward_period=fwd,
                                ic_result=result,
                            )
                        )
                    except Exception as exc:
                        entries.append(
                            ScanEntry(
                                indicator_name=col,
                                source=source,
                                symbol=symbol,
                                forward_period=fwd,
                                ic_result=None,
                                error=str(exc),
                            )
                        )
        return entries

    def scan_all(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        enriched_data: dict[str, pd.DataFrame] | None = None,
    ) -> ScanReport:
        """OHLCV + enriched 통합 스캔.

        Args:
            ohlcv_data: {symbol: OHLCV DataFrame} mapping
            enriched_data: {symbol: enriched DataFrame} mapping (optional)

        Returns:
            ScanReport
        """
        entries = self.scan_ohlcv_indicators(ohlcv_data)
        if enriched_data:
            entries.extend(self.scan_enriched_columns(enriched_data))
        return ScanReport(entries=entries, config=self._config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_source(column_name: str) -> str:
    """컬럼 이름에서 데이터 소스 감지."""
    for prefix, source in ENRICHED_PREFIXES.items():
        if column_name.startswith(prefix):
            return source
    return "unknown"
