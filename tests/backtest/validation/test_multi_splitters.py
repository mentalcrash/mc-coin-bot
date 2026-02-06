"""Tests for multi-asset splitters."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.validation.splitters import (
    split_multi_cpcv,
    split_multi_is_oos,
    split_multi_walk_forward,
)
from src.data.market_data import MultiSymbolData


@pytest.fixture
def multi_data() -> MultiSymbolData:
    """Create 2-asset MultiSymbolData for testing."""
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D", tz=UTC)
    rng = np.random.default_rng(42)

    ohlcv = {}
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        close = 100.0 + np.cumsum(rng.normal(0, 1, 365))
        ohlcv[symbol] = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1000, 5000, 365),
            },
            index=dates,
        )

    return MultiSymbolData(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1D",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 12, 31, tzinfo=UTC),
        ohlcv=ohlcv,
    )


class TestSplitMultiIsOos:
    """Tests for split_multi_is_oos()."""

    def test_default_ratio(self, multi_data: MultiSymbolData) -> None:
        """Test default 70/30 split."""
        train, test = split_multi_is_oos(multi_data)

        assert len(train.symbols) == len(multi_data.symbols)
        assert len(test.symbols) == len(multi_data.symbols)

        # Check periods are split correctly
        total = len(multi_data.ohlcv["BTC/USDT"])
        expected_train = int(total * 0.7)
        assert len(train.ohlcv["BTC/USDT"]) == expected_train
        assert len(test.ohlcv["BTC/USDT"]) == total - expected_train

    def test_same_boundary_all_symbols(self, multi_data: MultiSymbolData) -> None:
        """All symbols should be split at the same time boundary."""
        train, test = split_multi_is_oos(multi_data)

        # Both symbols should have same number of rows in train and test
        for symbol in multi_data.symbols:
            assert len(train.ohlcv[symbol]) == len(train.ohlcv[multi_data.symbols[0]])
            assert len(test.ohlcv[symbol]) == len(test.ohlcv[multi_data.symbols[0]])

    def test_no_data_overlap(self, multi_data: MultiSymbolData) -> None:
        """Train and test should not overlap in time."""
        train, test = split_multi_is_oos(multi_data)

        for symbol in multi_data.symbols:
            train_end = train.ohlcv[symbol].index[-1]
            test_start = test.ohlcv[symbol].index[0]
            assert train_end < test_start


class TestSplitMultiWalkForward:
    """Tests for split_multi_walk_forward()."""

    def test_creates_folds(self, multi_data: MultiSymbolData) -> None:
        """Should create walk-forward folds."""
        folds = split_multi_walk_forward(multi_data, n_folds=3)

        assert len(folds) >= 1

        for train, test, info in folds:
            assert len(train.symbols) == len(multi_data.symbols)
            assert len(test.symbols) == len(multi_data.symbols)
            assert info.fold_id is not None

    def test_expanding_train_sizes(self, multi_data: MultiSymbolData) -> None:
        """Expanding window should increase train size."""
        folds = split_multi_walk_forward(multi_data, n_folds=3, expanding=True)

        if len(folds) < 2:
            pytest.skip("Not enough folds")

        train_sizes = [len(train.ohlcv[multi_data.symbols[0]]) for train, _, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_no_train_test_overlap(self, multi_data: MultiSymbolData) -> None:
        """Train and test should not overlap within each fold."""
        folds = split_multi_walk_forward(multi_data, n_folds=3)

        for train, test, _ in folds:
            sym = multi_data.symbols[0]
            assert train.ohlcv[sym].index[-1] < test.ohlcv[sym].index[0]


class TestSplitMultiCpcv:
    """Tests for split_multi_cpcv()."""

    def test_generates_combinations(self, multi_data: MultiSymbolData) -> None:
        """Should generate CPCV combinations."""
        folds = list(split_multi_cpcv(multi_data, n_splits=4, n_test_splits=1))

        # C(4,1) = 4 combinations
        expected_combinations = 4
        assert len(folds) == expected_combinations

    def test_all_symbols_present(self, multi_data: MultiSymbolData) -> None:
        """Each fold should contain all symbols."""
        for train, test, _ in split_multi_cpcv(multi_data, n_splits=4, n_test_splits=1):
            assert set(train.symbols) == set(multi_data.symbols)
            assert set(test.symbols) == set(multi_data.symbols)

    def test_invalid_splits_raises(self, multi_data: MultiSymbolData) -> None:
        """n_test_splits >= n_splits should raise."""
        with pytest.raises(ValueError, match="must be <"):
            list(split_multi_cpcv(multi_data, n_splits=3, n_test_splits=3))

    def test_purge_reduces_train_size(self, multi_data: MultiSymbolData) -> None:
        """Purge periods should reduce training data."""
        folds_no_purge = list(
            split_multi_cpcv(multi_data, n_splits=4, n_test_splits=1, purge_periods=0)
        )
        folds_with_purge = list(
            split_multi_cpcv(multi_data, n_splits=4, n_test_splits=1, purge_periods=10)
        )

        # Average train size should be smaller with purge
        sym = multi_data.symbols[0]
        avg_no_purge = sum(len(t.ohlcv[sym]) for t, _, _ in folds_no_purge) / len(folds_no_purge)
        avg_with_purge = sum(len(t.ohlcv[sym]) for t, _, _ in folds_with_purge) / len(
            folds_with_purge
        )
        assert avg_with_purge <= avg_no_purge
