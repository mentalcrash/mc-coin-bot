"""Tests for Funding Rate + Stablecoin Confluence strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.fr_stab_conf.config import FrStabConfConfig, ShortMode
from src.strategy.fr_stab_conf.preprocessor import preprocess
from src.strategy.fr_stab_conf.signal import generate_signals
from src.strategy.fr_stab_conf.strategy import FrStabConfStrategy


@pytest.fixture
def sample_4h_ohlcv() -> pd.DataFrame:
    """4H OHLCV + derivatives + on-chain data (600 bars = 100 days)."""
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

    # Derivatives: funding rate (mean ~0, some extremes)
    funding_rate = np.random.randn(n) * 0.0003

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
            "funding_rate": funding_rate,
            "oc_stablecoin_total_circulating_usd": stab_4h,
        },
        index=dates,
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "fr-stab-conf" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fr-stab-conf")
        assert cls is FrStabConfStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = FrStabConfConfig()
        assert config.fr_ma_window == 9
        assert config.fr_zscore_window == 540
        assert config.short_mode == ShortMode.FULL

    def test_custom_config(self) -> None:
        config = FrStabConfConfig(fr_short_threshold=3.0, fr_long_threshold=-2.0)
        assert config.fr_short_threshold == 3.0
        assert config.fr_long_threshold == -2.0

    def test_validation_error_vol(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            FrStabConfConfig(vol_target=0.01, min_volatility=0.05)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {"returns", "realized_vol", "vol_scalar", "fr_z", "stab_z", "atr"}
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_funding_rate_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig()
        df = sample_4h_ohlcv.drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_stablecoin_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig()
        df = sample_4h_ohlcv.drop(columns=["oc_stablecoin_total_circulating_usd"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = FrStabConfConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_carry_forward(self) -> None:
        """State machine: position should be held until exit."""
        np.random.seed(99)
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")

        # Construct fr_z and stab_z directly to control signals
        fr_z = pd.Series(np.zeros(n), index=dates)
        stab_z = pd.Series(np.zeros(n), index=dates)
        vol_scalar = pd.Series(np.ones(n), index=dates)

        # Keep fr_z extreme at bars 3-6 to prevent premature exit
        # (|fr_z| < 0.5 triggers exit, so non-extreme bars would exit)
        fr_z.iloc[3] = -2.0
        fr_z.iloc[4] = -2.0
        fr_z.iloc[5] = -2.0
        fr_z.iloc[6] = -2.0
        stab_z.iloc[3] = 0.5
        # Exit at bar 7 (|fr_z| < 0.5) → after shift, exit at bar 8
        fr_z.iloc[7] = 0.0

        df = pd.DataFrame(
            {
                "fr_z": fr_z,
                "stab_z": stab_z,
                "vol_scalar": vol_scalar,
                "close": 50000.0,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "volume": 1000.0,
            },
            index=dates,
        )

        config = FrStabConfConfig()
        signals = generate_signals(df, config)

        # After shift(1): entry signal appears at bar 4, exit at bar 8
        # Bars 4-7 should be LONG (direction=1), bar 8+ should be FLAT
        assert signals.direction.iloc[4] == 1
        assert signals.direction.iloc[5] == 1
        assert signals.direction.iloc[6] == 1
        assert signals.direction.iloc[7] == 1
        assert signals.direction.iloc[8] == 0


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = FrStabConfStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = FrStabConfStrategy.from_params(fr_short_threshold=3.0)
        assert isinstance(strategy, FrStabConfStrategy)

    def test_recommended_config(self) -> None:
        config = FrStabConfStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = FrStabConfStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "fr_zscore_window" in info

    def test_name(self) -> None:
        assert FrStabConfStrategy().name == "fr-stab-conf"

    def test_required_columns(self) -> None:
        cols = FrStabConfStrategy().required_columns
        assert "close" in cols
        assert "funding_rate" in cols
        assert "oc_stablecoin_total_circulating_usd" in cols

    def test_config_type(self) -> None:
        assert isinstance(FrStabConfStrategy().config, FrStabConfConfig)

    def test_params_property(self) -> None:
        params = FrStabConfStrategy().params
        assert isinstance(params, dict)
        assert "fr_ma_window" in params

    def test_repr(self) -> None:
        assert repr(FrStabConfStrategy())


class TestEdgeCases:
    def test_all_nan_onchain_rejected(self) -> None:
        """All-NaN on-chain + derivatives should be rejected by validate_input."""
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
                "oc_stablecoin_total_circulating_usd": np.nan,
            },
            index=dates,
        )
        strategy = FrStabConfStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = FrStabConfConfig()
        warmup = config.warmup_periods()
        assert (
            warmup
            == max(
                config.fr_zscore_window,
                config.stab_zscore_window + config.stab_change_period,
            )
            + 10
        )
