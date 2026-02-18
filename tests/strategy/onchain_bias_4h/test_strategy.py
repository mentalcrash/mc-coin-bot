"""Tests for On-chain Bias 4H strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.onchain_bias_4h.config import OnchainBias4hConfig, ShortMode
from src.strategy.onchain_bias_4h.preprocessor import preprocess
from src.strategy.onchain_bias_4h.signal import generate_signals
from src.strategy.onchain_bias_4h.strategy import OnchainBias4hStrategy


@pytest.fixture
def sample_4h_ohlcv() -> pd.DataFrame:
    """4H OHLCV + on-chain data (600 bars = 100 days)."""
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

    # On-chain: MVRV (daily → 4H forward-fill), range ~1.0-4.0
    daily_mvrv = np.clip(2.0 + np.cumsum(np.random.randn(n // 6) * 0.1), 0.5, 5.0)
    mvrv_4h = np.repeat(daily_mvrv, 6)[:n]

    # Exchange flows
    flow_in = np.abs(np.random.randn(n)) * 1e8 + 5e8
    flow_out = np.abs(np.random.randn(n)) * 1e8 + 5e8

    # Stablecoin supply
    stab = np.linspace(1e11, 1.2e11, n) + np.random.randn(n) * 1e8

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_mvrv": mvrv_4h,
            "oc_flow_in_ex_usd": flow_in,
            "oc_flow_out_ex_usd": flow_out,
            "oc_stablecoin_total_circulating_usd": stab,
        },
        index=dates,
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "onchain-bias-4h" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("onchain-bias-4h")
        assert cls is OnchainBias4hStrategy


class TestConfig:
    def test_default_config(self) -> None:
        config = OnchainBias4hConfig()
        assert config.mvrv_accumulation == 1.8
        assert config.mvrv_distribution == 3.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_custom_config(self) -> None:
        config = OnchainBias4hConfig(mvrv_accumulation=1.5, er_min=0.2)
        assert config.mvrv_accumulation == 1.5
        assert config.er_min == 0.2

    def test_validation_error_vol(self) -> None:
        with pytest.raises(ValueError, match="vol_target"):
            OnchainBias4hConfig(vol_target=0.01, min_volatility=0.05)

    def test_validation_error_mvrv(self) -> None:
        with pytest.raises(ValueError, match="mvrv_accumulation"):
            OnchainBias4hConfig(mvrv_accumulation=4.0, mvrv_distribution=3.0)


class TestPreprocess:
    def test_output_columns(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig()
        result = preprocess(sample_4h_ohlcv, config)
        expected = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "net_flow",
            "stab_roc",
            "phase",
            "er",
            "price_roc",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_output_length(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig()
        result = preprocess(sample_4h_ohlcv, config)
        assert len(result) == len(sample_4h_ohlcv)

    def test_missing_mvrv_raises(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig()
        df = sample_4h_ohlcv.drop(columns=["oc_mvrv"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_phase_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig()
        result = preprocess(sample_4h_ohlcv, config)
        unique = set(result["phase"].dropna().unique())
        assert unique.issubset({-1, 0, 1})


class TestSignal:
    def test_output_shape(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)
        assert len(signals.exits) == len(df)
        assert len(signals.direction) == len(df)
        assert len(signals.strength) == len(df)

    def test_direction_values(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig()
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        unique = set(signals.direction.dropna().unique())
        assert unique.issubset({-1, 0, 1})

    def test_disabled_no_shorts(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        config = OnchainBias4hConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_4h_ohlcv, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()


class TestStrategy:
    def test_run_pipeline(self, sample_4h_ohlcv: pd.DataFrame) -> None:
        processed, signals = OnchainBias4hStrategy().run(sample_4h_ohlcv)
        assert len(processed) == len(sample_4h_ohlcv)
        assert len(signals.entries) == len(sample_4h_ohlcv)

    def test_from_params(self) -> None:
        strategy = OnchainBias4hStrategy.from_params(er_min=0.2)
        assert isinstance(strategy, OnchainBias4hStrategy)

    def test_recommended_config(self) -> None:
        config = OnchainBias4hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_name(self) -> None:
        assert OnchainBias4hStrategy().name == "onchain-bias-4h"

    def test_required_columns(self) -> None:
        cols = OnchainBias4hStrategy().required_columns
        assert "close" in cols
        assert "oc_mvrv" in cols
        assert "oc_stablecoin_total_circulating_usd" in cols

    def test_config_type(self) -> None:
        assert isinstance(OnchainBias4hStrategy().config, OnchainBias4hConfig)

    def test_params_property(self) -> None:
        params = OnchainBias4hStrategy().params
        assert isinstance(params, dict)
        assert "mvrv_accumulation" in params


class TestEdgeCases:
    def test_all_nan_onchain_rejected(self) -> None:
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
                "oc_mvrv": np.nan,
                "oc_flow_in_ex_usd": np.nan,
                "oc_flow_out_ex_usd": np.nan,
                "oc_stablecoin_total_circulating_usd": np.nan,
            },
            index=dates,
        )
        strategy = OnchainBias4hStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = OnchainBias4hConfig()
        warmup = config.warmup_periods()
        expected = (
            max(
                config.stab_roc_window,
                config.er_window,
                config.vol_window,
                config.roc_window,
            )
            + 10
        )
        assert warmup == expected

    def test_get_startup_info(self) -> None:
        info = OnchainBias4hStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "mvrv_accumulation" in info
