"""Tests for Disposition CGO strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.disposition_cgo.config import DispositionCgoConfig
from src.strategy.disposition_cgo.strategy import DispositionCgoStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "disposition-cgo" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("disposition-cgo")
        assert cls is DispositionCgoStrategy


class TestDispositionCgoStrategy:
    def _make_ohlcv(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )

    def test_name(self) -> None:
        strategy = DispositionCgoStrategy()
        assert strategy.name == "disposition-cgo"

    def test_required_columns(self) -> None:
        strategy = DispositionCgoStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = DispositionCgoStrategy()
        assert isinstance(strategy.config, DispositionCgoConfig)

    def test_preprocess(self) -> None:
        strategy = DispositionCgoStrategy()
        df = self._make_ohlcv()
        result = strategy.preprocess(df)
        assert len(result) == len(df)

    def test_generate_signals(self) -> None:
        strategy = DispositionCgoStrategy()
        df = self._make_ohlcv()
        processed = strategy.preprocess(df)
        signals = strategy.generate_signals(processed)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self) -> None:
        strategy = DispositionCgoStrategy()
        df = self._make_ohlcv()
        processed, signals = strategy.run(df)
        assert len(processed) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self) -> None:
        strategy = DispositionCgoStrategy.from_params(turnover_window=90)
        assert isinstance(strategy, DispositionCgoStrategy)

    def test_recommended_config(self) -> None:
        config = DispositionCgoStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = DispositionCgoStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "turnover_window" in info

    def test_custom_config(self) -> None:
        config = DispositionCgoConfig(turnover_window=90)
        strategy = DispositionCgoStrategy(config=config)
        assert strategy._config.turnover_window == 90

    def test_params_property(self) -> None:
        strategy = DispositionCgoStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "turnover_window" in params

    def test_repr(self) -> None:
        strategy = DispositionCgoStrategy()
        assert "disposition-cgo" in strategy.name
        assert repr(strategy)
