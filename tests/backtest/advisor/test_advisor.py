"""Tests for Strategy Advisor."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.backtest.advisor import StrategyAdvisor
from src.backtest.advisor.models import AdvisorReport
from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataSet
from src.portfolio import Portfolio
from src.strategy.tsmom import TSMOMStrategy


@pytest.fixture
def sample_market_data() -> MarketDataSet:
    """Create sample market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D", tz=UTC)
    # Create realistic price data with some trend and volatility
    import numpy as np

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 365)
    prices = [100.0]
    for r in returns[:-1]:
        prices.append(prices[-1] * (1 + r))

    ohlcv = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1000.0] * 365,
        },
        index=dates,
    )
    return MarketDataSet(
        symbol="BTC/USDT",
        timeframe="1D",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 12, 31, tzinfo=UTC),
        ohlcv=ohlcv,
    )


@pytest.fixture
def strategy() -> TSMOMStrategy:
    """Create strategy for testing."""
    return TSMOMStrategy()


@pytest.fixture
def portfolio() -> Portfolio:
    """Create portfolio for testing."""
    return Portfolio.create(initial_capital=Decimal("10000"))


class TestStrategyAdvisor:
    """Tests for StrategyAdvisor."""

    def test_analyze_basic(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test basic advisor analysis."""
        # First run backtest
        engine = BacktestEngine()
        request = BacktestRequest(
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result, strategy_returns, benchmark_returns = engine.run_with_returns(request)

        # Run advisor
        advisor = StrategyAdvisor()
        report = advisor.analyze(
            result=result,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
        )

        # Check report structure
        assert isinstance(report, AdvisorReport)
        assert report.strategy_name == strategy.name
        assert 0 <= report.overall_score <= 100
        assert report.readiness_level in ["development", "testing", "production"]

    def test_loss_concentration(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test loss concentration analysis."""
        engine = BacktestEngine()
        request = BacktestRequest(
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result, strategy_returns, benchmark_returns = engine.run_with_returns(request)

        advisor = StrategyAdvisor()
        report = advisor.analyze(
            result=result,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
        )

        loss = report.loss_concentration

        # Check hourly PnL exists (daily data will have fewer unique hours)
        assert len(loss.hourly_pnl) >= 1

        # Check weekday PnL has entries (daily data should have multiple weekdays)
        assert len(loss.weekday_pnl) >= 1

        # Check worst hours is a tuple
        assert isinstance(loss.worst_hours, tuple)
        # Worst hours should have entries based on available data
        assert len(loss.worst_hours) >= 1

    def test_regime_profile(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test regime profile analysis."""
        engine = BacktestEngine()
        request = BacktestRequest(
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result, strategy_returns, benchmark_returns = engine.run_with_returns(request)

        advisor = StrategyAdvisor()
        report = advisor.analyze(
            result=result,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
        )

        regime = report.regime_profile

        # Check regime sharpes are numeric
        assert isinstance(regime.bull_sharpe, float)
        assert isinstance(regime.bear_sharpe, float)
        assert isinstance(regime.sideways_sharpe, float)

        # Check weakest regime is identified
        assert regime.weakest_regime in ["bull", "bear", "sideways"]

    def test_signal_quality(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test signal quality analysis."""
        engine = BacktestEngine()
        request = BacktestRequest(
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result, strategy_returns, benchmark_returns = engine.run_with_returns(request)

        advisor = StrategyAdvisor()
        report = advisor.analyze(
            result=result,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
        )

        signal = report.signal_quality

        # Check hit rate is within valid range
        assert 0 <= signal.hit_rate <= 100

        # Check holding distribution has correct keys
        expected_keys = {"<1h", "1h-4h", "4h-1d", "1d-1w", ">1w"}
        assert set(signal.holding_distribution.keys()) == expected_keys

    def test_suggestions_generated(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test that suggestions are generated."""
        engine = BacktestEngine()
        request = BacktestRequest(
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result, strategy_returns, benchmark_returns = engine.run_with_returns(request)

        advisor = StrategyAdvisor()
        report = advisor.analyze(
            result=result,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
        )

        # Suggestions should be a tuple
        assert isinstance(report.suggestions, tuple)

        # Each suggestion should have required fields
        for suggestion in report.suggestions:
            assert suggestion.priority in ["high", "medium", "low"]
            assert suggestion.category in ["signal", "risk", "execution", "data"]
            assert len(suggestion.title) > 0
            assert len(suggestion.description) > 0

    def test_summary(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test report summary method."""
        engine = BacktestEngine()
        request = BacktestRequest(
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result, strategy_returns, benchmark_returns = engine.run_with_returns(request)

        advisor = StrategyAdvisor()
        report = advisor.analyze(
            result=result,
            returns=strategy_returns,
            benchmark_returns=benchmark_returns,
        )

        summary = report.summary()

        # Check summary has expected keys
        assert "strategy" in summary
        assert "overall_score" in summary
        assert "readiness" in summary
        assert "suggestions" in summary
