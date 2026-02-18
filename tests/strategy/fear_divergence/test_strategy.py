"""Tests for Fear-Greed Divergence strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.fear_divergence.config import FearDivergenceConfig, ShortMode
from src.strategy.fear_divergence.preprocessor import preprocess
from src.strategy.fear_divergence.signal import generate_signals
from src.strategy.fear_divergence.strategy import FearDivergenceStrategy


@pytest.fixture
def sample_4h_ohlcv() -> pd.DataFrame:
    """4H OHLCV + F&G data (600 bars = 100 days)."""
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

    # F&G: daily (forward-filled to 4H), range [0, 100]
    daily_fg = np.clip(50 + np.cumsum(np.random.randn(n // 6) * 5), 0, 100)
    fg_4h = np.repeat(daily_fg, 6)[:n]

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_fear_greed": fg_4h,
        },
        index=dates,
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "fear-divergence" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fear-divergence")
        assert cls is FearDivergenceStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = FearDivergenceConfig()
        assert config.fg_fear_threshold == 20
        assert config.fg_greed_threshold == 80
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_custom_config(self) -> None:
        config = FearDivergenceConfig(fg_fear_threshold=15, fg_deviation=20.0)
        assert config.fg_fear_threshold == 15
        assert config.fg_deviation == 20.0

    def test_validation_error_vol(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            FearDivergenceConfig(vol_target=0.01, min_volatility=0.05)

    def test_validation_error_thresholds(self) -> None:
        with pytest.raises(ValueError, match="fg_fear_threshold"):
            FearDivergenceConfig(fg_fear_threshold=90, fg_greed_threshold=80)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FearDivergenceConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {"returns", "realized_vol", "vol_scalar", "fg_ma", "price_roc", "er", "atr"}
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FearDivergenceConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_fg_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FearDivergenceConfig()
        df = sample_4h_ohlcv.drop(columns=["oc_fear_greed"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FearDivergenceConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FearDivergenceConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FearDivergenceConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = FearDivergenceStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = FearDivergenceStrategy.from_params(fg_fear_threshold=15)
        assert isinstance(strategy, FearDivergenceStrategy)

    def test_recommended_config(self) -> None:
        config = FearDivergenceStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_name(self) -> None:
        assert FearDivergenceStrategy().name == "fear-divergence"

    def test_required_columns(self) -> None:
        cols = FearDivergenceStrategy().required_columns
        assert "close" in cols
        assert "oc_fear_greed" in cols

    def test_config_type(self) -> None:
        assert isinstance(FearDivergenceStrategy().config, FearDivergenceConfig)

    def test_params_property(self) -> None:
        params = FearDivergenceStrategy().params
        assert isinstance(params, dict)
        assert "fg_fear_threshold" in params


class TestEdgeCases:
    def test_all_nan_fg_rejected(self) -> None:
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
                "oc_fear_greed": np.nan,
            },
            index=dates,
        )
        strategy = FearDivergenceStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = FearDivergenceConfig()
        warmup = config.warmup_periods()
        assert warmup == max(config.fg_ma_window, config.er_window, config.vol_window) + 10

    def test_get_startup_info(self) -> None:
        info = FearDivergenceStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "fg_fear_threshold" in info
