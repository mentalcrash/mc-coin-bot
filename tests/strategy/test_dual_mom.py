"""Tests for Dual Momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.dual_mom.config import DualMomConfig
from src.strategy.dual_mom.preprocessor import preprocess
from src.strategy.dual_mom.signal import generate_signals
from src.strategy.dual_mom.strategy import DualMomStrategy
from src.strategy.tsmom.config import ShortMode

# ── Helpers ─────────────────────────────────────────────────────


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="12h")
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    high = close * (1 + rng.uniform(0.001, 0.03, n))
    low = close * (1 - rng.uniform(0.001, 0.03, n))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.uniform(1000, 5000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ── TestConfig ───────────────────────────────────────────────────


class TestConfig:
    """Config defaults/validation 테스트."""

    def test_defaults(self) -> None:
        config = DualMomConfig()
        assert config.lookback == 42
        assert config.vol_window == 30
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.use_log_returns is True
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen(self) -> None:
        config = DualMomConfig()
        with pytest.raises(Exception):  # noqa: B017
            config.lookback = 10  # type: ignore[misc]

    def test_vol_target_below_min_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"vol_target.*min_volatility"):
            DualMomConfig(vol_target=0.05, min_volatility=0.06)

    def test_warmup_periods(self) -> None:
        config = DualMomConfig(lookback=42, vol_window=30)
        assert config.warmup_periods() == 43  # max(42, 30) + 1

    def test_custom_params(self) -> None:
        config = DualMomConfig(
            lookback=21,
            vol_target=0.20,
            annualization_factor=365.0,
        )
        assert config.lookback == 21
        assert config.vol_target == 0.20
        assert config.annualization_factor == 365.0


# ── TestPreprocessor ─────────────────────────────────────────────


class TestPreprocessor:
    """Preprocessor output columns 테스트."""

    def test_output_columns(self) -> None:
        df = _make_ohlcv()
        config = DualMomConfig()
        result = preprocess(df, config)

        expected_cols = {"returns", "realized_vol", "rolling_return", "vol_scalar", "atr"}
        assert expected_cols.issubset(set(result.columns))

    def test_no_modification_of_input(self) -> None:
        df = _make_ohlcv()
        original_cols = set(df.columns)
        config = DualMomConfig()
        preprocess(df, config)
        assert set(df.columns) == original_cols

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"close": [100, 101]})
        config = DualMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_warmup_nan(self) -> None:
        """워밍업 기간 동안 NaN 존재."""
        df = _make_ohlcv(n=100)
        config = DualMomConfig(lookback=42)
        result = preprocess(df, config)

        # 처음 lookback 기간은 rolling_return이 NaN
        assert result["rolling_return"].iloc[:42].isna().any()
        # 이후에는 유효한 값
        assert result["rolling_return"].iloc[42:].notna().all()


# ── TestSignal ───────────────────────────────────────────────────


class TestSignal:
    """Signal 생성 테스트."""

    def test_shift_one_applied(self) -> None:
        """shift(1) 적용 → 첫 번째 값은 0."""
        df = _make_ohlcv()
        config = DualMomConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        assert signals.strength.iloc[0] == 0.0
        assert signals.direction.iloc[0] == 0

    def test_long_only_default(self) -> None:
        """기본 short_mode=DISABLED → 숏 시그널 없음."""
        df = _make_ohlcv(n=300)
        config = DualMomConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_mode_allows_short(self) -> None:
        """short_mode=FULL → 숏 시그널 존재 가능."""
        df = _make_ohlcv(n=300, seed=123)
        config = DualMomConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 충분한 데이터로 양방향 시그널 존재 가능
        assert signals.direction.dtype == int

    def test_output_structure(self) -> None:
        """StrategySignals 구조 확인."""
        df = _make_ohlcv()
        config = DualMomConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")
        assert len(signals.entries) == len(df)

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"close": [100, 101], "high": [102, 103], "low": [99, 100]})
        config = DualMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)


# ── TestStrategy ─────────────────────────────────────────────────


class TestStrategy:
    """Strategy class 테스트."""

    def test_registry_registered(self) -> None:
        """Registry에 등록 확인."""
        from src.strategy.registry import get_strategy

        cls = get_strategy("dual-mom")
        assert cls is DualMomStrategy

    def test_from_params(self) -> None:
        """from_params() 동작 확인."""
        strategy = DualMomStrategy.from_params(lookback=21, vol_target=0.20)
        assert strategy.config.lookback == 21
        assert strategy.config.vol_target == 0.20

    def test_name(self) -> None:
        strategy = DualMomStrategy()
        assert strategy.name == "DualMom"

    def test_required_columns(self) -> None:
        strategy = DualMomStrategy()
        assert "close" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns

    def test_warmup_periods(self) -> None:
        strategy = DualMomStrategy()
        assert strategy.warmup_periods() == 43

    def test_run(self) -> None:
        """전체 run() 파이프라인."""
        df = _make_ohlcv(n=200)
        strategy = DualMomStrategy()
        processed, signals = strategy.run(df)

        assert "rolling_return" in processed.columns
        assert len(signals.direction) == len(df)

    def test_get_startup_info(self) -> None:
        strategy = DualMomStrategy()
        info = strategy.get_startup_info()
        assert "lookback" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_recommended_config(self) -> None:
        config = DualMomStrategy.recommended_config()
        assert "max_leverage_cap" in config
        assert config["max_leverage_cap"] == 1.0
