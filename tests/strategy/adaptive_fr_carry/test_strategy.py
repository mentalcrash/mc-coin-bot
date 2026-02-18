"""Tests for Adaptive FR Carry strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.adaptive_fr_carry.config import AdaptiveFrCarryConfig, ShortMode
from src.strategy.adaptive_fr_carry.preprocessor import preprocess
from src.strategy.adaptive_fr_carry.signal import generate_signals
from src.strategy.adaptive_fr_carry.strategy import AdaptiveFrCarryStrategy


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
        assert "adaptive-fr-carry" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("adaptive-fr-carry")
        assert cls is AdaptiveFrCarryStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = AdaptiveFrCarryConfig()
        assert config.fr_ma_window == 9
        assert config.fr_zscore_window == 42
        assert config.fr_entry_threshold == 2.0
        assert config.short_mode == ShortMode.FULL

    def test_custom_config(self) -> None:
        config = AdaptiveFrCarryConfig(fr_entry_threshold=3.0, er_max=0.5)
        assert config.fr_entry_threshold == 3.0
        assert config.er_max == 0.5

    def test_validation_error_vol(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            AdaptiveFrCarryConfig(vol_target=0.01, min_volatility=0.05)

    def test_validation_error_exit_entry(self) -> None:
        with pytest.raises(ValueError, match="fr_exit_threshold"):
            AdaptiveFrCarryConfig(fr_exit_threshold=3.0, fr_entry_threshold=2.0)

    def test_validation_error_atr_windows(self) -> None:
        with pytest.raises(ValueError, match="atr_short_window"):
            AdaptiveFrCarryConfig(atr_short_window=50, atr_long_window=42)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = AdaptiveFrCarryConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {"returns", "realized_vol", "vol_scalar", "fr_z", "atr", "atr_ratio", "er"}
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = AdaptiveFrCarryConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_funding_rate_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = AdaptiveFrCarryConfig()
        df = sample_4h_ohlcv.drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = AdaptiveFrCarryConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = AdaptiveFrCarryConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = AdaptiveFrCarryConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = AdaptiveFrCarryStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = AdaptiveFrCarryStrategy.from_params(fr_entry_threshold=3.0)
        assert isinstance(strategy, AdaptiveFrCarryStrategy)

    def test_recommended_config(self) -> None:
        config = AdaptiveFrCarryStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_name(self) -> None:
        assert AdaptiveFrCarryStrategy().name == "adaptive-fr-carry"

    def test_required_columns(self) -> None:
        cols = AdaptiveFrCarryStrategy().required_columns
        assert "close" in cols
        assert "funding_rate" in cols

    def test_config_type(self) -> None:
        assert isinstance(AdaptiveFrCarryStrategy().config, AdaptiveFrCarryConfig)

    def test_params_property(self) -> None:
        params = AdaptiveFrCarryStrategy().params
        assert isinstance(params, dict)
        assert "fr_ma_window" in params


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
        strategy = AdaptiveFrCarryStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = AdaptiveFrCarryConfig()
        warmup = config.warmup_periods()
        assert (
            warmup == max(config.fr_zscore_window, config.vol_window, config.atr_long_window) + 10
        )

    def test_get_startup_info(self) -> None:
        info = AdaptiveFrCarryStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "fr_entry_threshold" in info
