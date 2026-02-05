"""Tests for data splitters."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.backtest.validation.splitters import (
    get_split_info_is_oos,
    split_is_oos,
    split_walk_forward,
)
from src.data.market_data import MarketDataSet


@pytest.fixture
def sample_market_data() -> MarketDataSet:
    """Create sample market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D", tz=UTC)
    ohlcv = pd.DataFrame(
        {
            "open": [100.0] * 365,
            "high": [105.0] * 365,
            "low": [95.0] * 365,
            "close": [102.0] * 365,
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


class TestSplitIsOos:
    """Tests for IS/OOS split."""

    def test_default_ratio(self, sample_market_data: MarketDataSet) -> None:
        """Test default 70/30 split."""
        train, test = split_is_oos(sample_market_data)

        total_periods = sample_market_data.periods
        expected_train = int(total_periods * 0.7)

        assert train.periods == expected_train
        assert test.periods == total_periods - expected_train
        assert train.symbol == sample_market_data.symbol
        assert test.symbol == sample_market_data.symbol

    def test_custom_ratio(self, sample_market_data: MarketDataSet) -> None:
        """Test custom ratio."""
        train, test = split_is_oos(sample_market_data, ratio=0.8)

        total_periods = sample_market_data.periods
        expected_train = int(total_periods * 0.8)

        assert train.periods == expected_train
        assert test.periods == total_periods - expected_train

    def test_no_overlap(self, sample_market_data: MarketDataSet) -> None:
        """Test that train and test sets don't overlap."""
        train, test = split_is_oos(sample_market_data)

        assert train.end < test.start


class TestSplitWalkForward:
    """Tests for Walk-Forward split."""

    def test_creates_folds(self, sample_market_data: MarketDataSet) -> None:
        """Test that folds are created."""
        folds = split_walk_forward(sample_market_data, n_folds=3)

        # Should create at least some folds (may be fewer if data is insufficient)
        assert len(folds) >= 1

        for train, test, info in folds:
            assert train.periods > 0
            assert test.periods > 0
            assert train.symbol == sample_market_data.symbol
            assert info.fold_id is not None

    def test_expanding_window_train_sizes(self, sample_market_data: MarketDataSet) -> None:
        """Test that expanding window increases train size."""
        folds = split_walk_forward(sample_market_data, n_folds=3, expanding=True)

        if len(folds) < 2:
            pytest.skip("Not enough data for multiple folds")

        train_sizes = [train.periods for train, _, _ in folds]

        # Each subsequent fold should have more or equal training data (expanding)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_no_overlap_between_train_test(self, sample_market_data: MarketDataSet) -> None:
        """Test that train and test sets don't overlap within each fold."""
        folds = split_walk_forward(sample_market_data, n_folds=3)

        for train, test, _ in folds:
            # Train should end before test starts
            assert train.end <= test.start


class TestGetSplitInfoIsOos:
    """Tests for split info extraction."""

    def test_split_info(self, sample_market_data: MarketDataSet) -> None:
        """Test split info metadata."""
        info = get_split_info_is_oos(sample_market_data, ratio=0.7)

        assert info.train_start is not None
        assert info.train_end is not None
        assert info.test_start is not None
        assert info.test_end is not None
        assert info.train_periods > 0
        assert info.test_periods > 0
        assert info.train_end < info.test_start
