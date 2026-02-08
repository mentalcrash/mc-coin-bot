"""build_performance_metrics() 통합 테스트.

Phase 6-C: 통합 Metrics 엔진의 핵심 함수 검증.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd

from src.backtest.metrics import (
    TradeStatsResult,
    build_performance_metrics,
    compute_trade_stats,
    freq_to_periods_per_year,
)
from src.models.backtest import TradeRecord


def _make_trade(
    pnl: float,
    pnl_pct: float,
    direction: str = "LONG",
) -> TradeRecord:
    """테스트용 TradeRecord 생성."""
    return TradeRecord(
        entry_time=datetime(2024, 1, 1, tzinfo=UTC),
        exit_time=datetime(2024, 1, 2, tzinfo=UTC),
        symbol="BTC/USDT",
        direction=direction,
        entry_price=Decimal(50000),
        exit_price=Decimal(51000) if pnl > 0 else Decimal(49000),
        size=Decimal("0.1"),
        pnl=Decimal(str(pnl)),
        pnl_pct=pnl_pct,
        fees=Decimal("1.0"),
    )


class TestFreqToPeriodsPerYear:
    """freq_to_periods_per_year() 변환 테스트."""

    def test_daily(self) -> None:
        assert freq_to_periods_per_year("1D") == 365.0

    def test_4h(self) -> None:
        assert freq_to_periods_per_year("4h") == 2190.0

    def test_1h(self) -> None:
        assert freq_to_periods_per_year("1h") == 8760.0

    def test_15m(self) -> None:
        assert freq_to_periods_per_year("15T") == 35040.0

    def test_default_unknown_unit(self) -> None:
        """알 수 없는 단위 → 일봉(365)으로 기본 처리."""
        assert freq_to_periods_per_year("1X") == 365.0


class TestComputeTradeStats:
    """compute_trade_stats() 테스트."""

    def test_empty_trades(self) -> None:
        result = compute_trade_stats([])
        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.win_rate == 0.0
        assert result.avg_win is None
        assert result.avg_loss is None
        assert result.profit_factor is None

    def test_all_winners(self) -> None:
        trades = [
            _make_trade(pnl=100.0, pnl_pct=0.02),
            _make_trade(pnl=200.0, pnl_pct=0.04),
        ]
        result = compute_trade_stats(trades)
        assert result.total_trades == 2
        assert result.winning_trades == 2
        assert result.losing_trades == 0
        assert result.win_rate == 100.0
        assert result.avg_win is not None
        assert result.avg_win > 0
        assert result.avg_loss is None

    def test_mixed_trades(self) -> None:
        trades = [
            _make_trade(pnl=100.0, pnl_pct=0.02),
            _make_trade(pnl=-50.0, pnl_pct=-0.01),
        ]
        result = compute_trade_stats(trades)
        assert result.total_trades == 2
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.win_rate == 50.0

    def test_profit_factor(self) -> None:
        trades = [
            _make_trade(pnl=300.0, pnl_pct=0.06),
            _make_trade(pnl=-100.0, pnl_pct=-0.02),
        ]
        result = compute_trade_stats(trades)
        assert result.profit_factor is not None
        assert result.profit_factor == 3.0

    def test_returns_named_tuple(self) -> None:
        result = compute_trade_stats([])
        assert isinstance(result, TradeStatsResult)


class TestBuildPerformanceMetrics:
    """build_performance_metrics() 통합 테스트."""

    def test_monotonic_equity_no_drawdown(self) -> None:
        """단조 증가 equity → MDD 0, positive returns."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(30)], index=idx, dtype=float)

        metrics = build_performance_metrics(equity, trades=[], periods_per_year=365.0)

        assert metrics.total_return > 0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_trades == 0
        assert metrics.sharpe_ratio > 0

    def test_with_drawdown(self) -> None:
        """drawdown 포함 equity → MDD > 0 (양수), calmar 계산."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        # peak at 12000, then drop to 10800 (10% DD), then recover
        values = [10000, 11000, 12000, 11000, 10800, 11200, 11500, 12000, 12500, 13000]
        equity = pd.Series(values, index=idx, dtype=float)

        metrics = build_performance_metrics(equity, trades=[], periods_per_year=365.0)

        assert metrics.max_drawdown > 0  # 양수 (EDA 규약)
        assert metrics.max_drawdown == 10.0  # (12000-10800)/12000*100
        assert metrics.calmar_ratio is not None

    def test_with_trades(self) -> None:
        """TradeRecord 포함 → win_rate, profit_factor 등 정확."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        equity = pd.Series([10000 + i * 50 for i in range(10)], index=idx, dtype=float)
        trades = [
            _make_trade(pnl=200.0, pnl_pct=0.04),
            _make_trade(pnl=-100.0, pnl_pct=-0.02),
        ]

        metrics = build_performance_metrics(equity, trades=trades, periods_per_year=365.0)

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 50.0
        assert metrics.profit_factor is not None
        assert metrics.profit_factor == 2.0

    def test_empty_equity(self) -> None:
        """빈/짧은 equity → 기본값 반환."""
        equity = pd.Series([10000.0], dtype=float)
        metrics = build_performance_metrics(equity, trades=[])

        assert metrics.total_return == 0.0
        assert metrics.cagr == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0

    def test_funding_drag_reduces_return(self) -> None:
        """funding_drag > 0 → total_return 감소."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(30)], index=idx, dtype=float)

        metrics_no_drag = build_performance_metrics(equity, trades=[], periods_per_year=365.0)
        metrics_drag = build_performance_metrics(
            equity,
            trades=[],
            periods_per_year=365.0,
            funding_drag_per_period=0.001,
        )

        assert metrics_drag.total_return < metrics_no_drag.total_return

    def test_sortino_computed(self) -> None:
        """sortino_ratio가 None이 아닌 값 반환."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        # 약간의 변동이 있는 equity (하방 포함)
        values = [10000 + i * 50 + ((-1) ** i) * 30 for i in range(30)]
        equity = pd.Series(values, index=idx, dtype=float)

        metrics = build_performance_metrics(equity, trades=[], periods_per_year=365.0)

        assert metrics.sortino_ratio is not None

    def test_skewness_kurtosis_computed(self) -> None:
        """skewness, kurtosis 필드 채워짐."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(30)], index=idx, dtype=float)

        metrics = build_performance_metrics(equity, trades=[], periods_per_year=365.0)

        assert metrics.skewness is not None
        assert metrics.kurtosis is not None

    def test_volatility_computed(self) -> None:
        """volatility 필드 채워짐."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        equity = pd.Series([10000 + i * 100 for i in range(30)], index=idx, dtype=float)

        metrics = build_performance_metrics(equity, trades=[], periods_per_year=365.0)

        assert metrics.volatility is not None
        assert metrics.volatility > 0

    def test_max_drawdown_is_positive(self) -> None:
        """max_drawdown은 양수 (EDA 규약)."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        equity = pd.Series([10000, 11000, 9000, 10000, 10500], index=idx, dtype=float)

        metrics = build_performance_metrics(equity, trades=[], periods_per_year=365.0)

        assert metrics.max_drawdown > 0
