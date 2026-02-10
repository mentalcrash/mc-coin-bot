"""ChartGenerator 테스트 — PNG 생성 검증."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd

from src.models.backtest import TradeRecord
from src.monitoring.chart_generator import ChartGenerator

_PNG_MAGIC = b"\x89PNG"


def _make_equity_series(days: int = 60, start_val: float = 10000.0) -> pd.Series:
    """테스트용 equity 시리즈 생성."""
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2025-01-01", periods=days, freq="D", tz=UTC)
    returns = rng.normal(0.001, 0.02, size=days)
    values = start_val * np.cumprod(1 + returns)
    return pd.Series(values, index=dates, dtype=float)


def _make_trades(n: int = 20) -> list[TradeRecord]:
    """테스트용 거래 목록 생성."""
    import numpy as np

    rng = np.random.default_rng(42)
    trades: list[TradeRecord] = []
    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    for i in range(n):
        pnl = float(rng.normal(50, 200))
        trades.append(
            TradeRecord(
                entry_time=base_time + timedelta(days=i),
                exit_time=base_time + timedelta(days=i, hours=12),
                symbol="BTC/USDT",
                direction="LONG" if pnl >= 0 else "SHORT",
                entry_price=Decimal(40000),
                exit_price=Decimal(40100) if pnl >= 0 else Decimal(39900),
                size=Decimal("0.1"),
                pnl=Decimal(str(round(pnl, 2))),
                pnl_pct=pnl / 40000 * 100,
                fees=Decimal("2.00"),
            )
        )
    return trades


class TestEquityCurve:
    def test_non_empty_series(self) -> None:
        gen = ChartGenerator()
        series = _make_equity_series()
        result = gen.generate_equity_curve(series)
        assert len(result) > 0
        assert result[:4] == _PNG_MAGIC

    def test_empty_series(self) -> None:
        gen = ChartGenerator()
        result = gen.generate_equity_curve(pd.Series(dtype=float))
        assert result == b""


class TestDrawdown:
    def test_non_empty_series(self) -> None:
        gen = ChartGenerator()
        series = _make_equity_series()
        result = gen.generate_drawdown(series)
        assert len(result) > 0
        assert result[:4] == _PNG_MAGIC

    def test_empty_series(self) -> None:
        gen = ChartGenerator()
        result = gen.generate_drawdown(pd.Series(dtype=float))
        assert result == b""


class TestMonthlyHeatmap:
    def test_non_empty_series(self) -> None:
        gen = ChartGenerator()
        # 최소 2개월 이상 데이터 필요
        series = _make_equity_series(days=180)
        result = gen.generate_monthly_heatmap(series)
        assert len(result) > 0
        assert result[:4] == _PNG_MAGIC

    def test_short_series(self) -> None:
        gen = ChartGenerator()
        # 단일 값이면 빈 bytes
        result = gen.generate_monthly_heatmap(pd.Series([100.0], dtype=float))
        assert result == b""


class TestTradePnlDistribution:
    def test_non_empty_trades(self) -> None:
        gen = ChartGenerator()
        trades = _make_trades()
        result = gen.generate_trade_pnl_distribution(trades)
        assert len(result) > 0
        assert result[:4] == _PNG_MAGIC

    def test_empty_trades(self) -> None:
        gen = ChartGenerator()
        result = gen.generate_trade_pnl_distribution([])
        assert result == b""


class TestDailyReport:
    def test_returns_equity_chart(self) -> None:
        gen = ChartGenerator()
        series = _make_equity_series()
        trades = _make_trades()
        # metrics는 daily_report에서 직접 사용하지 않지만 인터페이스 호환
        from unittest.mock import MagicMock

        metrics = MagicMock()
        result = gen.generate_daily_report(series, trades, metrics)
        assert len(result) >= 1
        filenames = [name for name, _ in result]
        assert "equity_curve.png" in filenames

    def test_empty_data(self) -> None:
        gen = ChartGenerator()
        from unittest.mock import MagicMock

        metrics = MagicMock()
        result = gen.generate_daily_report(pd.Series(dtype=float), [], metrics)
        assert result == []


class TestWeeklyReport:
    def test_returns_multiple_charts(self) -> None:
        gen = ChartGenerator()
        series = _make_equity_series(days=180)
        trades = _make_trades()
        from unittest.mock import MagicMock

        metrics = MagicMock()
        result = gen.generate_weekly_report(series, trades, metrics)
        filenames = [name for name, _ in result]
        assert "equity_curve.png" in filenames
        assert "drawdown.png" in filenames

    def test_all_pngs_valid(self) -> None:
        gen = ChartGenerator()
        series = _make_equity_series(days=180)
        trades = _make_trades()
        from unittest.mock import MagicMock

        metrics = MagicMock()
        result = gen.generate_weekly_report(series, trades, metrics)
        for name, data in result:
            assert data[:4] == _PNG_MAGIC, f"{name} is not valid PNG"
