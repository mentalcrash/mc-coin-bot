"""Tests for R2 Consensus Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.r2_consensus.config import R2ConsensusConfig
from src.strategy.r2_consensus.strategy import R2ConsensusStrategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "r2-consensus" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("r2-consensus")
        assert cls is R2ConsensusStrategy


class TestR2ConsensusStrategy:
    def test_name(self) -> None:
        strategy = R2ConsensusStrategy()
        assert strategy.name == "r2-consensus"

    def test_required_columns(self) -> None:
        strategy = R2ConsensusStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "open" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns

    def test_config(self) -> None:
        strategy = R2ConsensusStrategy()
        assert isinstance(strategy.config, R2ConsensusConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = R2ConsensusStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = R2ConsensusStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = R2ConsensusStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = R2ConsensusStrategy.from_params(
            lookback_short=10, lookback_mid=40, lookback_long=80
        )
        assert isinstance(strategy, R2ConsensusStrategy)

    def test_from_params_default(self) -> None:
        strategy = R2ConsensusStrategy.from_params()
        assert isinstance(strategy, R2ConsensusStrategy)

    def test_recommended_config(self) -> None:
        config = R2ConsensusStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config
        assert "trailing_stop_enabled" in config

    def test_get_startup_info(self) -> None:
        strategy = R2ConsensusStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "lookback_short" in info
        assert "lookback_mid" in info
        assert "lookback_long" in info
        assert "r2_threshold" in info
        assert "entry_threshold" in info
        assert "vol_target" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = R2ConsensusConfig(lookback_short=10, lookback_mid=40, lookback_long=80)
        strategy = R2ConsensusStrategy(config=config)
        assert strategy._config.lookback_short == 10

    def test_params_property(self) -> None:
        strategy = R2ConsensusStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "lookback_short" in params
        assert "lookback_mid" in params
        assert "lookback_long" in params

    def test_repr(self) -> None:
        strategy = R2ConsensusStrategy()
        assert "r2-consensus" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    def test_warmup_consistent(self) -> None:
        """warmup_periods가 config에서 일관되게 반환."""
        strategy = R2ConsensusStrategy()
        config = strategy._config
        assert (
            config.warmup_periods()
            == max(config.lookback_long, config.vol_window, config.atr_period) + 10
        )
