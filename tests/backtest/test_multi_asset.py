"""Tests for multi-asset backtest infrastructure."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import MultiSymbolData


@pytest.fixture
def multi_symbol_data() -> MultiSymbolData:
    """Create 2-asset MultiSymbolData."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D", tz=UTC)
    rng = np.random.default_rng(42)

    ohlcv = {}
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        close = 100.0 + np.cumsum(rng.normal(0.1, 1, 200))
        ohlcv[symbol] = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1000, 5000, 200),
            },
            index=dates,
        )

    return MultiSymbolData(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1D",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 7, 18, tzinfo=UTC),
        ohlcv=ohlcv,
    )


class TestMultiSymbolData:
    """Tests for MultiSymbolData container."""

    def test_creation(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test basic creation."""
        assert len(multi_symbol_data.symbols) == 2
        assert multi_symbol_data.timeframe == "1D"
        assert multi_symbol_data.n_assets == 2

    def test_close_matrix(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test close_matrix property creates correct DataFrame."""
        close_matrix = multi_symbol_data.close_matrix
        assert isinstance(close_matrix, pd.DataFrame)
        assert set(close_matrix.columns) == {"BTC/USDT", "ETH/USDT"}
        assert len(close_matrix) == 200

    def test_freq(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test freq property."""
        assert multi_symbol_data.freq == "1D"

    def test_periods(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test periods property."""
        assert multi_symbol_data.periods == 200

    def test_get_single(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test extracting single symbol MarketDataSet."""
        btc = multi_symbol_data.get_single("BTC/USDT")
        assert btc.symbol == "BTC/USDT"
        assert btc.periods == 200
        assert "close" in btc.ohlcv.columns

    def test_slice_time(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test time-based slicing."""
        start = datetime(2024, 3, 1, tzinfo=UTC)
        end = datetime(2024, 5, 1, tzinfo=UTC)
        sliced = multi_symbol_data.slice_time(start, end)

        assert len(sliced.symbols) == 2
        for symbol in sliced.symbols:
            df = sliced.ohlcv[symbol]
            assert df.index[0] >= pd.Timestamp(start)
            assert df.index[-1] <= pd.Timestamp(end)

    def test_slice_iloc(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test integer-based slicing."""
        sliced = multi_symbol_data.slice_iloc(10, 50)

        assert len(sliced.symbols) == 2
        for symbol in sliced.symbols:
            assert len(sliced.ohlcv[symbol]) == 40

    def test_repr(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test string representation."""
        repr_str = repr(multi_symbol_data)
        assert "BTC/USDT" in repr_str
        assert "periods=200" in repr_str


class TestMultiAssetBacktestRequest:
    """Tests for MultiAssetBacktestRequest DTO."""

    def test_equal_weight_default(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test that weights default to equal weight."""
        from src.backtest.request import MultiAssetBacktestRequest
        from src.portfolio import Portfolio
        from src.strategy.tsmom import TSMOMStrategy

        request = MultiAssetBacktestRequest(
            data=multi_symbol_data,
            strategy=TSMOMStrategy(),
            portfolio=Portfolio.create(),
        )

        weights = request.asset_weights
        assert len(weights) == 2
        assert abs(weights["BTC/USDT"] - 0.5) < 1e-10
        assert abs(weights["ETH/USDT"] - 0.5) < 1e-10

    def test_custom_weights(self, multi_symbol_data: MultiSymbolData) -> None:
        """Test custom weight specification."""
        from src.backtest.request import MultiAssetBacktestRequest
        from src.portfolio import Portfolio
        from src.strategy.tsmom import TSMOMStrategy

        custom_weights = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        request = MultiAssetBacktestRequest(
            data=multi_symbol_data,
            strategy=TSMOMStrategy(),
            portfolio=Portfolio.create(),
            weights=custom_weights,
        )

        assert request.asset_weights == custom_weights


class TestMultiAssetResultModels:
    """Tests for MultiAssetConfig and MultiAssetBacktestResult models."""

    def test_multi_asset_config_creation(self) -> None:
        """Test MultiAssetConfig creation."""
        from src.models.backtest import MultiAssetConfig

        config = MultiAssetConfig(
            strategy_name="tsmom",
            symbols=("BTC/USDT", "ETH/USDT"),
            timeframe="1D",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
            initial_capital=100000,
            asset_weights={"BTC/USDT": 0.5, "ETH/USDT": 0.5},
            fees=0.001,
            strategy_params={"lookback": 30},
        )

        assert config.strategy_name == "tsmom"
        assert len(config.symbols) == 2

    def test_multi_asset_result_creation(self) -> None:
        """Test MultiAssetBacktestResult creation."""
        from src.models.backtest import (
            MultiAssetBacktestResult,
            MultiAssetConfig,
            PerformanceMetrics,
        )

        config = MultiAssetConfig(
            strategy_name="tsmom",
            symbols=("BTC/USDT", "ETH/USDT"),
            timeframe="1D",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 31, tzinfo=UTC),
            initial_capital=100000,
            asset_weights={"BTC/USDT": 0.5, "ETH/USDT": 0.5},
            fees=0.001,
            strategy_params={},
        )

        metrics = PerformanceMetrics(
            total_return=50.0,
            cagr=40.0,
            sharpe_ratio=2.0,
            sortino_ratio=3.0,
            max_drawdown=15.0,
            calmar_ratio=2.5,
            volatility=20.0,
            win_rate=55.0,
            profit_factor=1.5,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            avg_trade_return=0.5,
            max_consecutive_losses=5,
        )

        result = MultiAssetBacktestResult(
            config=config,
            portfolio_metrics=metrics,
            per_symbol_metrics={"BTC/USDT": metrics, "ETH/USDT": metrics},
            correlation_matrix={
                "BTC/USDT": {"BTC/USDT": 1.0, "ETH/USDT": 0.5},
                "ETH/USDT": {"BTC/USDT": 0.5, "ETH/USDT": 1.0},
            },
            contribution={"BTC/USDT": 30.0, "ETH/USDT": 20.0},
        )

        assert result.portfolio_metrics.sharpe_ratio == 2.0
        assert result.contribution["BTC/USDT"] == 30.0
