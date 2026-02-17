"""Tests for TieredValidator."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.backtest.validation.levels import ValidationLevel
from src.backtest.validation.validator import TieredValidator
from src.data.market_data import MarketDataSet
from src.portfolio import Portfolio
from src.strategy.tsmom import TSMOMStrategy

_VBT_AVAILABLE = True
try:
    import vectorbt  # noqa: F401
except ImportError:
    _VBT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _VBT_AVAILABLE, reason="vectorbt not installed")


@pytest.fixture
def sample_market_data() -> MarketDataSet:
    """Create sample market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D", tz=UTC)
    # Create realistic price data with some trend
    close_prices = [100.0 * (1 + 0.001 * i) for i in range(365)]
    ohlcv = pd.DataFrame(
        {
            "open": close_prices,
            "high": [p * 1.02 for p in close_prices],
            "low": [p * 0.98 for p in close_prices],
            "close": close_prices,
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
    return Portfolio.create(initial_capital=Decimal(10000))


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_levels_exist(self) -> None:
        """Test that all levels exist."""
        assert ValidationLevel.QUICK == "quick"
        assert ValidationLevel.MILESTONE == "milestone"
        assert ValidationLevel.FINAL == "final"

    def test_string_conversion(self) -> None:
        """Test string conversion."""
        assert str(ValidationLevel.QUICK) == "quick"


class TestTieredValidator:
    """Tests for TieredValidator."""

    def test_quick_validation(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test quick (IS/OOS) validation."""
        validator = TieredValidator()

        result = validator.validate(
            level=ValidationLevel.QUICK,
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )

        # Should have exactly 1 fold for IS/OOS
        assert len(result.fold_results) == 1
        assert result.level == ValidationLevel.QUICK

        # Check metrics exist
        assert result.avg_train_sharpe is not None
        assert result.avg_test_sharpe is not None

    def test_milestone_validation(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test milestone (Walk-Forward) validation."""
        validator = TieredValidator()

        result = validator.validate(
            level=ValidationLevel.MILESTONE,
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
            n_folds=3,  # Use fewer folds for faster testing
        )

        # Should have at least 1 fold (may be fewer if data is insufficient)
        assert len(result.fold_results) >= 1
        assert result.level == ValidationLevel.MILESTONE

    def test_verdict_property(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test that verdict is computed."""
        validator = TieredValidator()

        result = validator.validate(
            level=ValidationLevel.QUICK,
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )

        # Verdict should be one of the expected values
        assert result.verdict in ["PASS", "WARN", "FAIL"]

    def test_overfit_probability(
        self,
        sample_market_data: MarketDataSet,
        strategy: TSMOMStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Test that overfit probability is computed."""
        validator = TieredValidator()

        result = validator.validate(
            level=ValidationLevel.QUICK,
            data=sample_market_data,
            strategy=strategy,
            portfolio=portfolio,
        )

        # Overfit probability should be between 0 and 1
        assert 0.0 <= result.overfit_probability <= 1.0
