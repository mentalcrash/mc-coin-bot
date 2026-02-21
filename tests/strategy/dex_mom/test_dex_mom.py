"""Tests for DEX Activity Momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.types import Direction
from src.strategy.dex_mom.config import DexMomConfig
from src.strategy.dex_mom.preprocessor import preprocess
from src.strategy.dex_mom.signal import generate_signals
from src.strategy.dex_mom.strategy import DexMomStrategy
from src.strategy.tsmom.config import ShortMode


@pytest.fixture
def sample_ohlcv_with_dex() -> pd.DataFrame:
    """200일 OHLCV + DEX volume 샘플."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    close = base + np.cumsum(np.random.randn(n) * 300)
    close = np.maximum(close, base * 0.5)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)

    # DEX volume: 상승 추세 + 변동
    dex_vol = 5e9 + np.cumsum(np.random.randn(n) * 1e8)
    dex_vol = np.maximum(dex_vol, 1e9)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
            "oc_dex_volume_usd": dex_vol,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def sample_ohlcv_only() -> pd.DataFrame:
    """200일 OHLCV only (on-chain 없음)."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    close = base + np.cumsum(np.random.randn(n) * 300)
    close = np.maximum(close, base * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    """DexMomConfig 검증."""

    def test_defaults(self) -> None:
        cfg = DexMomConfig()
        assert cfg.roc_short_window == 7
        assert cfg.roc_long_window == 30
        assert cfg.short_mode == ShortMode.FULL

    def test_warmup(self) -> None:
        cfg = DexMomConfig()
        expected = cfg.roc_long_window + cfg.vol_window + 10
        assert cfg.warmup_periods() == expected

    def test_custom_params(self) -> None:
        cfg = DexMomConfig(roc_short_window=5, roc_long_window=20)
        assert cfg.roc_short_window == 5
        assert cfg.roc_long_window == 20

    def test_from_params(self) -> None:
        s = DexMomStrategy.from_params(roc_short_window=10, vol_target=0.25)
        assert s.config.roc_short_window == 10
        assert s.config.vol_target == 0.25

    def test_frozen(self) -> None:
        cfg = DexMomConfig()
        with pytest.raises(Exception):  # noqa: B017
            cfg.roc_short_window = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Preprocessor 검증."""

    def test_adds_columns_with_dex(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        cfg = DexMomConfig()
        result = preprocess(sample_ohlcv_with_dex, cfg)
        expected = {
            "dex_roc_short",
            "dex_roc_long",
            "realized_vol",
            "vol_scalar",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_without_dex(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """On-chain 없으면 NaN 컬럼."""
        cfg = DexMomConfig()
        result = preprocess(sample_ohlcv_only, cfg)
        assert result["dex_roc_short"].isna().all()
        assert result["dex_roc_long"].isna().all()

    def test_same_length(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        cfg = DexMomConfig()
        result = preprocess(sample_ohlcv_with_dex, cfg)
        assert len(result) == len(sample_ohlcv_with_dex)

    def test_missing_ohlcv_columns(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, DexMomConfig())


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Signal 생성 검증."""

    def test_output_shape(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        cfg = DexMomConfig()
        processed = preprocess(sample_ohlcv_with_dex, cfg)
        signals = generate_signals(processed, cfg)
        assert len(signals.direction) == len(sample_ohlcv_with_dex)
        assert len(signals.strength) == len(sample_ohlcv_with_dex)

    def test_direction_values_full(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        """FULL short mode → -1, 0, 1 가능."""
        cfg = DexMomConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_with_dex, cfg)
        signals = generate_signals(processed, cfg)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_no_lookahead(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        """첫 번째 바 시그널은 0."""
        cfg = DexMomConfig()
        processed = preprocess(sample_ohlcv_with_dex, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.strength.iloc[0] == 0.0

    def test_no_signal_without_dex(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """On-chain 없으면 시그널 없음."""
        cfg = DexMomConfig()
        processed = preprocess(sample_ohlcv_only, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction == 0).all()

    def test_disabled_short_mode(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        """DISABLED → 숏 없음."""
        cfg = DexMomConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_with_dex, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction != Direction.SHORT).all()

    def test_both_roc_positive_long(self) -> None:
        """양쪽 ROC 양수 → LONG."""
        n = 50
        df = pd.DataFrame(
            {
                "close": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "volume": [1000.0] * n,
                "dex_roc_short": [0.05] * n,
                "dex_roc_long": [0.02] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = DexMomConfig()
        signals = generate_signals(df, cfg)
        assert signals.direction.iloc[1] == Direction.LONG

    def test_both_roc_negative_short(self) -> None:
        """양쪽 ROC 음수 → SHORT (FULL mode)."""
        n = 50
        df = pd.DataFrame(
            {
                "close": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "volume": [1000.0] * n,
                "dex_roc_short": [-0.05] * n,
                "dex_roc_long": [-0.02] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = DexMomConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(df, cfg)
        assert signals.direction.iloc[1] == Direction.SHORT

    def test_mixed_roc_flat(self) -> None:
        """ROC 혼합 → FLAT."""
        n = 50
        df = pd.DataFrame(
            {
                "close": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "volume": [1000.0] * n,
                "dex_roc_short": [0.05] * n,
                "dex_roc_long": [-0.02] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = DexMomConfig()
        signals = generate_signals(df, cfg)
        assert (signals.direction == Direction.NEUTRAL).all()

    def test_entries_exits_bool(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        cfg = DexMomConfig()
        processed = preprocess(sample_ohlcv_with_dex, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class TestStrategy:
    """Strategy 클래스 통합 검증."""

    def test_registry(self) -> None:
        from src.strategy.registry import get_strategy

        cls = get_strategy("dex-mom")
        assert cls is DexMomStrategy

    def test_name(self) -> None:
        assert DexMomStrategy().name == "Dex-Mom"

    def test_required_columns(self) -> None:
        s = DexMomStrategy()
        assert "close" in s.required_columns

    def test_run(self, sample_ohlcv_with_dex: pd.DataFrame) -> None:
        strategy = DexMomStrategy()
        _processed, signals = strategy.run(sample_ohlcv_with_dex)
        assert len(signals.direction) == len(sample_ohlcv_with_dex)

    def test_warmup(self) -> None:
        assert DexMomStrategy().warmup_periods() > 0

    def test_recommended_config(self) -> None:
        rec = DexMomStrategy.recommended_config()
        assert rec["trailing_stop_atr_multiplier"] == 3.5

    def test_startup_info(self) -> None:
        info = DexMomStrategy().get_startup_info()
        assert "roc_windows" in info
        assert "mode" in info
