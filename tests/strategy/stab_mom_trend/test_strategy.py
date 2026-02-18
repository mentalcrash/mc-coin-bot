"""Tests for Stablecoin Momentum Trend strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.stab_mom_trend.config import ShortMode, StabMomTrendConfig
from src.strategy.stab_mom_trend.preprocessor import preprocess
from src.strategy.stab_mom_trend.signal import generate_signals
from src.strategy.stab_mom_trend.strategy import StabMomTrendStrategy


@pytest.fixture
def sample_4h_ohlcv() -> pd.DataFrame:
    """4H OHLCV + on-chain stablecoin data (600 bars = 100 days)."""
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

    # On-chain: daily stablecoin supply (forward-filled to 4H)
    daily_stab = np.linspace(1e11, 1.2e11, n // 6)
    stab_4h = np.repeat(daily_stab, 6)[:n]
    stab_4h = stab_4h + np.random.randn(n) * 1e8

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_stablecoin_total_circulating_usd": stab_4h,
        },
        index=dates,
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "stab-mom-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("stab-mom-trend")
        assert cls is StabMomTrendStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = StabMomTrendConfig()
        assert config.stab_change_period == 42
        assert config.zscore_window == 540
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_custom_config(self) -> None:
        config = StabMomTrendConfig(stab_change_period=30, ema_fast_period=12, ema_slow_period=48)
        assert config.stab_change_period == 30
        assert config.ema_fast_period == 12

    def test_validation_error_ema_order(self) -> None:
        with pytest.raises(ValueError, match="ema_fast_period"):
            StabMomTrendConfig(ema_fast_period=100, ema_slow_period=50)

    def test_validation_error_threshold_order(self) -> None:
        with pytest.raises(ValueError, match="stab_short_threshold"):
            StabMomTrendConfig(stab_long_threshold=0.0, stab_short_threshold=0.0)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "stab_z",
            "ema_fast",
            "ema_slow",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_column_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig()
        df = sample_4h_ohlcv.drop(columns=["oc_stablecoin_total_circulating_usd"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_hedge_only_reduces_short_strength(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig(short_mode=ShortMode.HEDGE_ONLY, hedge_strength_ratio=0.5)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            # HEDGE_ONLY: short strength magnitude should be reduced
            assert (signals.strength[short_mask] <= 0).all()

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = StabMomTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = StabMomTrendStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = StabMomTrendStrategy.from_params(stab_change_period=30)
        assert isinstance(strategy, StabMomTrendStrategy)

    def test_recommended_config(self) -> None:
        config = StabMomTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = StabMomTrendStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "zscore_window" in info

    def test_name(self) -> None:
        assert StabMomTrendStrategy().name == "stab-mom-trend"

    def test_required_columns(self) -> None:
        cols = StabMomTrendStrategy().required_columns
        assert "close" in cols
        assert "oc_stablecoin_total_circulating_usd" in cols

    def test_config_type(self) -> None:
        assert isinstance(StabMomTrendStrategy().config, StabMomTrendConfig)

    def test_params_property(self) -> None:
        params = StabMomTrendStrategy().params
        assert isinstance(params, dict)
        assert "stab_change_period" in params

    def test_repr(self) -> None:
        assert repr(StabMomTrendStrategy())


class TestEdgeCases:
    def test_all_nan_onchain_rejected(self) -> None:
        """All-NaN on-chain column should be rejected by validate_input."""
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
                "oc_stablecoin_total_circulating_usd": np.nan,
            },
            index=dates,
        )
        strategy = StabMomTrendStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = StabMomTrendConfig()
        warmup = config.warmup_periods()
        assert warmup == config.zscore_window + config.stab_change_period + 10
