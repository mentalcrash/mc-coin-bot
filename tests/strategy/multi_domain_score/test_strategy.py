"""Tests for Multi-Domain Score strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.multi_domain_score.config import MultiDomainScoreConfig, ShortMode
from src.strategy.multi_domain_score.preprocessor import preprocess
from src.strategy.multi_domain_score.signal import generate_signals
from src.strategy.multi_domain_score.strategy import MultiDomainScoreStrategy


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
        assert "multi-domain-score" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("multi-domain-score")
        assert cls is MultiDomainScoreStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = MultiDomainScoreConfig()
        assert config.w_trend == 0.35
        assert config.w_volume == 0.25
        assert config.entry_threshold == 0.45
        assert config.short_mode == ShortMode.FULL

    def test_custom_config(self) -> None:
        config = MultiDomainScoreConfig(
            w_trend=0.4, w_volume=0.2, w_derivatives=0.2, w_volatility=0.2
        )
        assert config.w_trend == 0.4

    def test_validation_error_weights_sum(self) -> None:
        with pytest.raises(ValueError, match=r"weights must sum to 1\.0"):
            MultiDomainScoreConfig(w_trend=0.5, w_volume=0.5)

    def test_validation_error_vol(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            MultiDomainScoreConfig(vol_target=0.01, min_volatility=0.05)

    def test_validation_error_rv_windows(self) -> None:
        with pytest.raises(ValueError, match="rv_short_window"):
            MultiDomainScoreConfig(rv_short_window=50, rv_long_window=42)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = MultiDomainScoreConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "er",
            "sma_direction",
            "obv_slope",
            "fr_z",
            "rv_ratio",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = MultiDomainScoreConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_funding_rate_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = MultiDomainScoreConfig()
        df = sample_4h_ohlcv.drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = MultiDomainScoreConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = MultiDomainScoreConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = MultiDomainScoreConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = MultiDomainScoreStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = MultiDomainScoreStrategy.from_params(entry_threshold=0.5)
        assert isinstance(strategy, MultiDomainScoreStrategy)

    def test_recommended_config(self) -> None:
        config = MultiDomainScoreStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_name(self) -> None:
        assert MultiDomainScoreStrategy().name == "multi-domain-score"

    def test_required_columns(self) -> None:
        cols = MultiDomainScoreStrategy().required_columns
        assert "close" in cols
        assert "funding_rate" in cols

    def test_config_type(self) -> None:
        assert isinstance(MultiDomainScoreStrategy().config, MultiDomainScoreConfig)

    def test_params_property(self) -> None:
        params = MultiDomainScoreStrategy().params
        assert isinstance(params, dict)
        assert "w_trend" in params


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
        strategy = MultiDomainScoreStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = MultiDomainScoreConfig()
        warmup = config.warmup_periods()
        expected = (
            max(
                config.er_window,
                config.sma_window,
                config.fr_zscore_window,
                config.rv_long_window,
                config.vol_window,
            )
            + 10
        )
        assert warmup == expected

    def test_get_startup_info(self) -> None:
        info = MultiDomainScoreStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "entry_threshold" in info
