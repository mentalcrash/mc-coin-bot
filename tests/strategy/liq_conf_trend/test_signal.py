"""Tests for Liquidity-Confirmed Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.liq_conf_trend.config import LiqConfTrendConfig, ShortMode
from src.strategy.liq_conf_trend.preprocessor import preprocess
from src.strategy.liq_conf_trend.signal import generate_signals


@pytest.fixture
def config() -> LiqConfTrendConfig:
    return LiqConfTrendConfig()


@pytest.fixture
def sample_full_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    stablecoin = 100e9 + np.cumsum(np.random.randn(n) * 1e8)
    tvl = 50e9 + np.cumsum(np.random.randn(n) * 5e7)
    fear_greed = np.random.randint(10, 90, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_stablecoin_total_usd": stablecoin,
            "oc_tvl_usd": tvl,
            "oc_fear_greed": fear_greed,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def preprocessed_df(sample_full_df: pd.DataFrame, config: LiqConfTrendConfig) -> pd.DataFrame:
    return preprocess(sample_full_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: LiqConfTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: LiqConfTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModeVariants:
    def test_disabled_no_shorts(self, sample_full_df: pd.DataFrame) -> None:
        config = LiqConfTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_full_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_full_df: pd.DataFrame) -> None:
        config = LiqConfTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_full_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_mode(self, sample_full_df: pd.DataFrame) -> None:
        config = LiqConfTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_full_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestLiqConfTrendLogic:
    def test_no_signal_without_onchain_and_momentum_zero(self) -> None:
        """Without on-chain, liq_score=0, so no confirmation → no signal."""
        n = 200
        np.random.seed(42)
        # Flat price → near-zero momentum
        close = np.full(n, 100.0)
        high = close + 0.5
        low = close - 0.5
        config = LiqConfTrendConfig()
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # Without on-chain + flat price → all neutral
        assert (signals.direction == 0).all()

    def test_fear_extreme_overrides_to_long(self) -> None:
        """F&G extreme fear forces long even with negative momentum."""
        n = 200
        np.random.seed(42)
        close = 100 - np.arange(n) * 0.5  # Downtrend
        high = close + 1
        low = close - 1
        config = LiqConfTrendConfig(fg_fear_threshold=20)
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "oc_fear_greed": np.full(n, 10.0),  # Extreme fear
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # Fear extreme → force long
        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            assert (valid == 1).all()

    def test_greed_extreme_overrides_to_short_full(self) -> None:
        """F&G extreme greed forces short in FULL mode."""
        n = 200
        np.random.seed(42)
        close = 100 + np.arange(n) * 0.5  # Uptrend
        high = close + 1
        low = close - 1
        config = LiqConfTrendConfig(
            fg_greed_threshold=80,
            short_mode=ShortMode.FULL,
        )
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "oc_fear_greed": np.full(n, 90.0),  # Extreme greed
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            assert (valid == -1).all()

    def test_liquidity_confirms_uptrend(self) -> None:
        """Growing stablecoin + TVL confirms uptrend momentum."""
        n = 200
        np.random.seed(42)
        close = 100 + np.arange(n) * 0.5  # Clear uptrend
        high = close + 1
        low = close - 1
        config = LiqConfTrendConfig(
            liq_score_threshold=1,
            fg_fear_threshold=5,
            fg_greed_threshold=95,
        )
        # Growing liquidity
        stablecoin = 100e9 + np.arange(n) * 1e8  # Steadily growing
        tvl = 50e9 + np.arange(n) * 5e7  # Steadily growing
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "oc_stablecoin_total_usd": stablecoin,
                "oc_tvl_usd": tvl,
                "oc_fear_greed": np.full(n, 50.0),  # Neutral F&G
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # After warmup, should see long signals
        late = signals.direction.iloc[50:]
        assert (late >= 0).all()
        # At least some long signals
        assert (late == 1).any()

    def test_strength_nonzero_when_active_after_warmup(
        self, preprocessed_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        active = post_warmup != 0
        if active.any():
            assert (signals.strength.iloc[warmup:][active].abs() > 0).all()
