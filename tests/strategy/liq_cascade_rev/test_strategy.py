"""Tests for Liquidation Cascade Reversal strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.liq_cascade_rev.config import LiqCascadeRevConfig, ShortMode
from src.strategy.liq_cascade_rev.preprocessor import preprocess
from src.strategy.liq_cascade_rev.signal import generate_signals
from src.strategy.liq_cascade_rev.strategy import LiqCascadeRevStrategy


@pytest.fixture
def sample_4h_ohlcv() -> pd.DataFrame:
    """4H OHLCV + FR data (600 bars = 100 days)."""
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 200)
    high = close + np.abs(np.random.randn(n) * 150)
    low = close - np.abs(np.random.randn(n) * 150)
    open_ = close + np.random.randn(n) * 50
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.random.randint(1000, 10000, n).astype(float)
    funding_rate = np.random.randn(n) * 0.0003

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=dates,
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "liq-cascade-rev" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("liq-cascade-rev")
        assert cls is LiqCascadeRevStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = LiqCascadeRevConfig()
        assert config.fr_buildup_threshold == 2.0
        assert config.cascade_return_multiplier == 2.5
        assert config.short_mode == ShortMode.FULL

    def test_custom_config(self) -> None:
        config = LiqCascadeRevConfig(cascade_return_multiplier=3.0, max_hold_bars=24)
        assert config.cascade_return_multiplier == 3.0
        assert config.max_hold_bars == 24

    def test_validation_error_vol(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            LiqCascadeRevConfig(vol_target=0.01, min_volatility=0.05)

    def test_validation_error_vol_windows(self) -> None:
        with pytest.raises(ValueError, match="vol_short_window"):
            LiqCascadeRevConfig(vol_short_window=50, vol_long_window=42)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = LiqCascadeRevConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "fr_z",
            "atr",
            "rv_ratio",
            "body_recovery",
            "return_atr_ratio",
            "return_dir",
        }
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = LiqCascadeRevConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_funding_rate_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = LiqCascadeRevConfig()
        df = sample_4h_ohlcv.drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = LiqCascadeRevConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = LiqCascadeRevConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = LiqCascadeRevConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = LiqCascadeRevStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = LiqCascadeRevStrategy.from_params(max_hold_bars=24)
        assert isinstance(strategy, LiqCascadeRevStrategy)

    def test_recommended_config(self) -> None:
        config = LiqCascadeRevStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_name(self) -> None:
        assert LiqCascadeRevStrategy().name == "liq-cascade-rev"

    def test_required_columns(self) -> None:
        cols = LiqCascadeRevStrategy().required_columns
        assert "close" in cols
        assert "funding_rate" in cols

    def test_config_type(self) -> None:
        assert isinstance(LiqCascadeRevStrategy().config, LiqCascadeRevConfig)

    def test_params_property(self) -> None:
        params = LiqCascadeRevStrategy().params
        assert isinstance(params, dict)
        assert "fr_buildup_threshold" in params


class TestEdgeCases:
    def test_all_nan_fr_rejected(self) -> None:
        np.random.seed(42)
        n = 600
        dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        close = 50000 + np.cumsum(np.random.randn(n) * 200)
        high = close + np.abs(np.random.randn(n) * 150)
        low = close - np.abs(np.random.randn(n) * 150)
        open_ = close + np.random.randn(n) * 50
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
                "funding_rate": np.nan,
            },
            index=dates,
        )
        strategy = LiqCascadeRevStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = LiqCascadeRevConfig()
        warmup = config.warmup_periods()
        assert (
            warmup == max(config.fr_zscore_window, config.vol_long_window, config.vol_window) + 10
        )

    def test_get_startup_info(self) -> None:
        info = LiqCascadeRevStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "cascade_return_multiplier" in info
