"""Macro-Context-Trend 12H 전략 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.macro_context_trend_12h.config import MacroContextTrendConfig, ShortMode
from src.strategy.macro_context_trend_12h.preprocessor import preprocess
from src.strategy.macro_context_trend_12h.signal import generate_signals
from src.strategy.macro_context_trend_12h.strategy import MacroContextTrend12hStrategy

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
def ohlcv_with_macro(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + macro columns."""
    df = ohlcv_df.copy()
    np.random.seed(99)
    n = len(df)
    df["macro_vix"] = 20 + np.random.randn(n) * 5
    df["macro_hy_spread"] = 3.0 + np.random.randn(n) * 0.5
    return df


@pytest.fixture()
def config() -> MacroContextTrendConfig:
    return MacroContextTrendConfig(
        ema_fast=8,
        ema_slow=20,
        vol_window=15,
        macro_window=15,
    )


# ── Config Tests ──────────────────────────────────────────


class TestConfig:
    def test_default_config(self) -> None:
        cfg = MacroContextTrendConfig()
        assert cfg.ema_fast == 12
        assert cfg.ema_slow == 26
        assert cfg.short_mode == ShortMode.HEDGE_ONLY

    def test_ema_order_validation(self) -> None:
        with pytest.raises(ValueError, match="ema_fast"):
            MacroContextTrendConfig(ema_fast=30, ema_slow=20)

    def test_macro_weight_validation(self) -> None:
        with pytest.raises(ValueError, match="macro_min_weight"):
            MacroContextTrendConfig(macro_min_weight=1.5, macro_max_weight=1.0)

    def test_warmup_periods(self) -> None:
        cfg = MacroContextTrendConfig()
        assert cfg.warmup_periods() >= cfg.ema_slow + 10


# ── Preprocessor Tests ────────────────────────────────────


class TestPreprocessor:
    def test_preprocess_adds_columns(self, ohlcv_df: pd.DataFrame, config: MacroContextTrendConfig) -> None:
        result = preprocess(ohlcv_df, config)
        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "ema_fast",
            "ema_slow",
            "ema_diff",
            "trend_direction",
            "trend_confirmed",
            "macro_context",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_graceful_degradation_no_macro(
        self,
        ohlcv_df: pd.DataFrame,
        config: MacroContextTrendConfig,
    ) -> None:
        """매크로 데이터 없을 때 중립 가중치 1.0."""
        result = preprocess(ohlcv_df, config)
        assert (result["macro_context"] == 1.0).all()

    def test_macro_context_with_data(
        self,
        ohlcv_with_macro: pd.DataFrame,
        config: MacroContextTrendConfig,
    ) -> None:
        """매크로 데이터 있을 때 가중치 변동."""
        result = preprocess(ohlcv_with_macro, config)
        ctx = result["macro_context"]
        # 가중치는 1.0과 다른 값도 포함해야 함
        assert not (ctx == 1.0).all()
        # 범위 확인 (blended이므로 극단 불가)
        assert (ctx.dropna() > 0).all()

    def test_preprocess_missing_columns(self, config: MacroContextTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


# ── Signal Tests ──────────────────────────────────────────


class TestSignal:
    def test_signal_shape(self, ohlcv_df: pd.DataFrame, config: MacroContextTrendConfig) -> None:
        processed = preprocess(ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert len(signals.entries) == len(ohlcv_df)
        assert len(signals.direction) == len(ohlcv_df)

    def test_direction_values(self, ohlcv_df: pd.DataFrame, config: MacroContextTrendConfig) -> None:
        processed = preprocess(ohlcv_df, config)
        signals = generate_signals(processed, config)
        unique_dirs = set(signals.direction.dropna().unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_entries_exits_exclusive(self, ohlcv_df: pd.DataFrame, config: MacroContextTrendConfig) -> None:
        processed = preprocess(ohlcv_df, config)
        signals = generate_signals(processed, config)
        both = signals.entries & signals.exits
        assert not both.any()

    def test_short_mode_disabled(self, ohlcv_df: pd.DataFrame) -> None:
        cfg = MacroContextTrendConfig(
            ema_fast=8,
            ema_slow=20,
            vol_window=15,
            short_mode=ShortMode.DISABLED,
        )
        processed = preprocess(ohlcv_df, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction >= 0).all()

    def test_macro_affects_strength(self, ohlcv_with_macro: pd.DataFrame, config: MacroContextTrendConfig) -> None:
        """매크로 컨텍스트가 strength에 영향."""
        # 매크로 없이 (중립)
        ohlcv_only = ohlcv_with_macro.drop(columns=["macro_vix", "macro_hy_spread"])
        p1 = preprocess(ohlcv_only, config)
        s1 = generate_signals(p1, config)

        # 매크로 있이
        p2 = preprocess(ohlcv_with_macro, config)
        s2 = generate_signals(p2, config)

        # strength 차이 존재
        mask = (s1.strength != 0) | (s2.strength != 0)
        if mask.any():
            diff = (s1.strength[mask] - s2.strength[mask]).abs()
            assert diff.sum() > 0


# ── Strategy Tests ────────────────────────────────────────


class TestStrategy:
    def test_registry(self) -> None:
        from src.strategy import get_strategy

        cls = get_strategy("macro-context-trend-12h")
        assert cls is MacroContextTrend12hStrategy

    def test_run(self, ohlcv_df: pd.DataFrame) -> None:
        strategy = MacroContextTrend12hStrategy(
            config=MacroContextTrendConfig(
                ema_fast=8,
                ema_slow=20,
                vol_window=15,
            ),
        )
        processed, _signals = strategy.run(ohlcv_df)
        assert len(processed) == len(ohlcv_df)

    def test_from_params(self) -> None:
        strategy = MacroContextTrend12hStrategy.from_params(ema_fast=8, ema_slow=20)
        assert strategy.name == "macro-context-trend-12h"

    def test_recommended_config(self) -> None:
        cfg = MacroContextTrend12hStrategy.recommended_config()
        assert cfg["use_intrabar_trailing_stop"] is False

    def test_startup_info(self) -> None:
        strategy = MacroContextTrend12hStrategy()
        info = strategy.get_startup_info()
        assert "ema_fast" in info
        assert "macro_risk_weight" in info
