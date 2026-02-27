"""Unit tests for EntropyGateTrend4hStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import BaseStrategy
from src.strategy.entropy_gate_trend_4h.config import EntropyGateTrend4hConfig, ShortMode
from src.strategy.entropy_gate_trend_4h.strategy import EntropyGateTrend4hStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestStrategyRegistry:
    def test_strategy_registered(self) -> None:
        assert "entropy-gate-trend-4h" in list_strategies()

    def test_get_strategy(self) -> None:
        strategy_cls = get_strategy("entropy-gate-trend-4h")
        assert strategy_cls is EntropyGateTrend4hStrategy

    def test_is_base_strategy_subclass(self) -> None:
        assert issubclass(EntropyGateTrend4hStrategy, BaseStrategy)


class TestStrategyProperties:
    def test_strategy_name(self) -> None:
        strategy = EntropyGateTrend4hStrategy()
        assert strategy.name == "entropy-gate-trend-4h"

    def test_required_columns(self) -> None:
        strategy = EntropyGateTrend4hStrategy()
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]

    def test_config_property(self) -> None:
        strategy = EntropyGateTrend4hStrategy()
        assert isinstance(strategy.config, EntropyGateTrend4hConfig)

    def test_default_config(self) -> None:
        strategy = EntropyGateTrend4hStrategy(config=None)
        assert isinstance(strategy.config, EntropyGateTrend4hConfig)


class TestRunPipeline:
    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = EntropyGateTrend4hStrategy()
        _processed_df, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_missing_columns_raises(self) -> None:
        strategy = EntropyGateTrend4hStrategy()
        bad_df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing"):
            strategy.run(bad_df)


class TestFromParams:
    def test_from_params(self) -> None:
        strategy = EntropyGateTrend4hStrategy.from_params()
        assert isinstance(strategy, EntropyGateTrend4hStrategy)

    def test_from_params_custom(self) -> None:
        strategy = EntropyGateTrend4hStrategy.from_params(vol_target=0.25)
        assert strategy.config.vol_target == 0.25


class TestRecommendedConfig:
    def test_recommended_config(self) -> None:
        config = EntropyGateTrend4hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "trailing_stop_enabled" in config
        assert config["use_intrabar_trailing_stop"] is False


class TestShortMode:
    def test_short_mode_disabled(self, sample_ohlcv: pd.DataFrame) -> None:
        config = EntropyGateTrend4hConfig(short_mode=ShortMode.DISABLED)
        strategy = EntropyGateTrend4hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert (signals.direction >= 0).all()

    def test_short_mode_full(self, sample_ohlcv: pd.DataFrame) -> None:
        config = EntropyGateTrend4hConfig(short_mode=ShortMode.FULL)
        strategy = EntropyGateTrend4hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)

    def test_short_mode_hedge_only(self, sample_ohlcv: pd.DataFrame) -> None:
        config = EntropyGateTrend4hConfig(short_mode=ShortMode.HEDGE_ONLY)
        strategy = EntropyGateTrend4hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)


class TestStartupInfo:
    def test_startup_info_keys(self) -> None:
        strategy = EntropyGateTrend4hStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert all(isinstance(v, str) for v in info.values())


class TestEntropyGating:
    """Test entropy gating effect: random data -> more flat signals."""

    def test_entropy_columns_computed(self, sample_ohlcv: pd.DataFrame) -> None:
        """Verify perm_entropy and is_predictable columns are computed."""
        strategy = EntropyGateTrend4hStrategy()
        processed_df, _ = strategy.run(sample_ohlcv)
        assert "perm_entropy" in processed_df.columns
        assert "is_predictable" in processed_df.columns
        # is_predictable should be binary (0 or 1)
        valid_mask = processed_df["is_predictable"].notna()
        assert set(processed_df.loc[valid_mask, "is_predictable"].unique()).issubset({0, 1})

    def test_high_entropy_reduces_signals(self) -> None:
        """With highly random data, entropy gate should produce more flat signals
        compared to a very low entropy threshold that passes everything."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="4h")
        # Create random walk data (high entropy)
        close = 50000 + np.cumsum(np.random.randn(n) * 500)
        high = close + np.abs(np.random.randn(n) * 200)
        low = close - np.abs(np.random.randn(n) * 200)
        open_ = close + np.random.randn(n) * 100
        volume = np.random.uniform(100, 10000, n)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        # Strict gate (low threshold -> more signals blocked)
        strict_config = EntropyGateTrend4hConfig(entropy_threshold=0.5)
        strict_strategy = EntropyGateTrend4hStrategy(config=strict_config)
        _, strict_signals = strict_strategy.run(df)

        # Permissive gate (high threshold -> more signals pass)
        permissive_config = EntropyGateTrend4hConfig(entropy_threshold=4.0)
        permissive_strategy = EntropyGateTrend4hStrategy(config=permissive_config)
        _, permissive_signals = permissive_strategy.run(df)

        # Strict gate should have fewer (or equal) non-zero direction signals
        strict_active = (strict_signals.direction != 0).sum()
        permissive_active = (permissive_signals.direction != 0).sum()
        assert strict_active <= permissive_active
