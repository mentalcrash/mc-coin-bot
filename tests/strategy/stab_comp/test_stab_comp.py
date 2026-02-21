"""Tests for Stablecoin Composition Shift strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.types import Direction
from src.strategy.stab_comp.config import StabCompConfig
from src.strategy.stab_comp.preprocessor import preprocess
from src.strategy.stab_comp.signal import generate_signals
from src.strategy.stab_comp.strategy import StabCompStrategy
from src.strategy.tsmom.config import ShortMode


@pytest.fixture
def sample_ohlcv_with_stablecoin() -> pd.DataFrame:
    """200일 OHLCV + stablecoin 샘플."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    close = base + np.cumsum(np.random.randn(n) * 300)
    close = np.maximum(close, base * 0.5)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)

    # USDT: 상승 추세 (리테일 유입)
    usdt = 80e9 + np.cumsum(np.abs(np.random.randn(n)) * 0.5e9)
    # USDC: 완만한 상승 (기관 유입)
    usdc = 30e9 + np.cumsum(np.abs(np.random.randn(n)) * 0.1e9)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
            "oc_stablecoin_usdt_usd": usdt,
            "oc_stablecoin_usdc_usd": usdc,
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
    """StabCompConfig 검증."""

    def test_defaults(self) -> None:
        cfg = StabCompConfig()
        assert cfg.roc_short_window == 7
        assert cfg.roc_long_window == 30
        assert cfg.short_mode == ShortMode.FULL

    def test_warmup(self) -> None:
        cfg = StabCompConfig()
        expected = cfg.roc_long_window + cfg.vol_window + 10
        assert cfg.warmup_periods() == expected

    def test_custom_params(self) -> None:
        cfg = StabCompConfig(roc_short_window=5, roc_long_window=20)
        assert cfg.roc_short_window == 5
        assert cfg.roc_long_window == 20

    def test_from_params(self) -> None:
        s = StabCompStrategy.from_params(roc_short_window=10, vol_target=0.25)
        assert s.config.roc_short_window == 10
        assert s.config.vol_target == 0.25

    def test_frozen(self) -> None:
        cfg = StabCompConfig()
        with pytest.raises(Exception):  # noqa: B017
            cfg.roc_short_window = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Preprocessor 검증."""

    def test_adds_columns_with_stablecoin(
        self, sample_ohlcv_with_stablecoin: pd.DataFrame
    ) -> None:
        cfg = StabCompConfig()
        result = preprocess(sample_ohlcv_with_stablecoin, cfg)
        expected = {
            "usdt_share",
            "share_roc_short",
            "share_roc_long",
            "realized_vol",
            "vol_scalar",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_usdt_share_range(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        """USDT share는 0~1 범위."""
        cfg = StabCompConfig()
        result = preprocess(sample_ohlcv_with_stablecoin, cfg)
        share = result["usdt_share"].dropna()
        assert (share >= 0).all()
        assert (share <= 1).all()

    def test_without_stablecoin(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """On-chain 없으면 NaN 컬럼."""
        cfg = StabCompConfig()
        result = preprocess(sample_ohlcv_only, cfg)
        assert result["share_roc_short"].isna().all()
        assert result["share_roc_long"].isna().all()
        assert result["usdt_share"].isna().all()

    def test_same_length(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        cfg = StabCompConfig()
        result = preprocess(sample_ohlcv_with_stablecoin, cfg)
        assert len(result) == len(sample_ohlcv_with_stablecoin)

    def test_missing_ohlcv_columns(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, StabCompConfig())


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Signal 생성 검증."""

    def test_output_shape(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        cfg = StabCompConfig()
        processed = preprocess(sample_ohlcv_with_stablecoin, cfg)
        signals = generate_signals(processed, cfg)
        assert len(signals.direction) == len(sample_ohlcv_with_stablecoin)
        assert len(signals.strength) == len(sample_ohlcv_with_stablecoin)

    def test_direction_values_full(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        """FULL short mode → -1, 0, 1 가능."""
        cfg = StabCompConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_with_stablecoin, cfg)
        signals = generate_signals(processed, cfg)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_no_lookahead(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        """첫 번째 바 시그널은 0."""
        cfg = StabCompConfig()
        processed = preprocess(sample_ohlcv_with_stablecoin, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.strength.iloc[0] == 0.0

    def test_no_signal_without_stablecoin(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """On-chain 없으면 시그널 없음."""
        cfg = StabCompConfig()
        processed = preprocess(sample_ohlcv_only, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction == 0).all()

    def test_disabled_short_mode(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        """DISABLED → 숏 없음."""
        cfg = StabCompConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_with_stablecoin, cfg)
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
                "share_roc_short": [0.01] * n,
                "share_roc_long": [0.005] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = StabCompConfig()
        signals = generate_signals(df, cfg)
        # shift(1) 때문에 두 번째 바부터 LONG
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
                "share_roc_short": [-0.01] * n,
                "share_roc_long": [-0.005] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = StabCompConfig(short_mode=ShortMode.FULL)
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
                "share_roc_short": [0.01] * n,
                "share_roc_long": [-0.005] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = StabCompConfig()
        signals = generate_signals(df, cfg)
        assert (signals.direction == Direction.NEUTRAL).all()

    def test_entries_exits_bool(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        cfg = StabCompConfig()
        processed = preprocess(sample_ohlcv_with_stablecoin, cfg)
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

        cls = get_strategy("stab-comp")
        assert cls is StabCompStrategy

    def test_name(self) -> None:
        assert StabCompStrategy().name == "Stab-Comp"

    def test_required_columns(self) -> None:
        s = StabCompStrategy()
        assert "close" in s.required_columns

    def test_run(self, sample_ohlcv_with_stablecoin: pd.DataFrame) -> None:
        strategy = StabCompStrategy()
        _processed, signals = strategy.run(sample_ohlcv_with_stablecoin)
        assert len(signals.direction) == len(sample_ohlcv_with_stablecoin)

    def test_warmup(self) -> None:
        assert StabCompStrategy().warmup_periods() > 0

    def test_recommended_config(self) -> None:
        rec = StabCompStrategy.recommended_config()
        assert rec["trailing_stop_atr_multiplier"] == 3.5

    def test_startup_info(self) -> None:
        info = StabCompStrategy().get_startup_info()
        assert "roc_windows" in info
        assert "mode" in info
