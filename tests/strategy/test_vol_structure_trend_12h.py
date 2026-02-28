"""Vol-Structure-Trend 12H 전략 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_structure_trend_12h.config import ShortMode, VolStructureTrendConfig
from src.strategy.vol_structure_trend_12h.preprocessor import preprocess
from src.strategy.vol_structure_trend_12h.signal import generate_signals
from src.strategy.vol_structure_trend_12h.strategy import VolStructureTrend12hStrategy

# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """100-bar OHLCV DataFrame with trending pattern."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="12h")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    opn = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture()
def config() -> VolStructureTrendConfig:
    return VolStructureTrendConfig(
        scale_short=10,
        scale_mid=20,
        scale_long=40,
        roc_lookback=10,
        vol_window=15,
    )


# ── Config Tests ──────────────────────────────────────────


class TestConfig:
    def test_default_config(self) -> None:
        cfg = VolStructureTrendConfig()
        assert cfg.scale_short == 14
        assert cfg.scale_mid == 30
        assert cfg.scale_long == 60
        assert cfg.short_mode == ShortMode.HEDGE_ONLY

    def test_scale_order_validation(self) -> None:
        with pytest.raises(ValueError, match="scale_short < scale_mid < scale_long"):
            VolStructureTrendConfig(scale_short=30, scale_mid=20, scale_long=60)

    def test_warmup_periods(self) -> None:
        cfg = VolStructureTrendConfig()
        assert cfg.warmup_periods() >= cfg.scale_long + 10

    def test_frozen(self) -> None:
        cfg = VolStructureTrendConfig()
        with pytest.raises(Exception):  # noqa: B017
            cfg.scale_short = 999  # type: ignore[misc]


# ── Preprocessor Tests ────────────────────────────────────


class TestPreprocessor:
    def test_preprocess_adds_columns(self, ohlcv_df: pd.DataFrame, config: VolStructureTrendConfig) -> None:
        result = preprocess(ohlcv_df, config)
        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "vol_agreement_10",
            "vol_agreement_20",
            "vol_agreement_40",
            "roc_direction",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_preprocess_missing_columns(self, config: VolStructureTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_agreement_range(self, ohlcv_df: pd.DataFrame, config: VolStructureTrendConfig) -> None:
        result = preprocess(ohlcv_df, config)
        for s in (10, 20, 40):
            col = f"vol_agreement_{s}"
            valid = result[col].dropna()
            assert (valid >= 0).all()
            assert (valid <= 1).all()


# ── Signal Tests ──────────────────────────────────────────


class TestSignal:
    def test_signal_shape(self, ohlcv_df: pd.DataFrame, config: VolStructureTrendConfig) -> None:
        processed = preprocess(ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert len(signals.entries) == len(ohlcv_df)
        assert len(signals.exits) == len(ohlcv_df)
        assert len(signals.direction) == len(ohlcv_df)
        assert len(signals.strength) == len(ohlcv_df)

    def test_direction_values(self, ohlcv_df: pd.DataFrame, config: VolStructureTrendConfig) -> None:
        processed = preprocess(ohlcv_df, config)
        signals = generate_signals(processed, config)
        unique_dirs = set(signals.direction.dropna().unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_entries_exits_exclusive(self, ohlcv_df: pd.DataFrame, config: VolStructureTrendConfig) -> None:
        processed = preprocess(ohlcv_df, config)
        signals = generate_signals(processed, config)
        both = signals.entries & signals.exits
        assert not both.any()

    def test_short_mode_disabled(self, ohlcv_df: pd.DataFrame) -> None:
        cfg = VolStructureTrendConfig(
            scale_short=10,
            scale_mid=20,
            scale_long=40,
            roc_lookback=10,
            vol_window=15,
            short_mode=ShortMode.DISABLED,
        )
        processed = preprocess(ohlcv_df, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction >= 0).all()

    def test_short_mode_full(self, ohlcv_df: pd.DataFrame) -> None:
        cfg = VolStructureTrendConfig(
            scale_short=10,
            scale_mid=20,
            scale_long=40,
            roc_lookback=10,
            vol_window=15,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(ohlcv_df, cfg)
        signals = generate_signals(processed, cfg)
        unique_dirs = set(signals.direction.dropna().unique())
        # FULL mode can have -1
        assert unique_dirs.issubset({-1, 0, 1})


# ── Strategy Tests ────────────────────────────────────────


class TestStrategy:
    def test_registry(self) -> None:
        from src.strategy import get_strategy

        cls = get_strategy("vol-structure-trend-12h")
        assert cls is VolStructureTrend12hStrategy

    def test_run(self, ohlcv_df: pd.DataFrame) -> None:
        strategy = VolStructureTrend12hStrategy(
            config=VolStructureTrendConfig(
                scale_short=10,
                scale_mid=20,
                scale_long=40,
                roc_lookback=10,
                vol_window=15,
            ),
        )
        processed, signals = strategy.run(ohlcv_df)
        assert len(processed) == len(ohlcv_df)
        assert len(signals.entries) == len(ohlcv_df)

    def test_from_params(self) -> None:
        strategy = VolStructureTrend12hStrategy.from_params(
            scale_short=10,
            scale_mid=20,
            scale_long=40,
        )
        assert strategy.name == "vol-structure-trend-12h"

    def test_recommended_config(self) -> None:
        cfg = VolStructureTrend12hStrategy.recommended_config()
        assert "trailing_stop_atr_multiplier" in cfg
        assert cfg["use_intrabar_trailing_stop"] is False

    def test_startup_info(self) -> None:
        strategy = VolStructureTrend12hStrategy()
        info = strategy.get_startup_info()
        assert "scale_short" in info
        assert "short_mode" in info
