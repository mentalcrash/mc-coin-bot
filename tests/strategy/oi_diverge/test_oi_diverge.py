"""Tests for OI-Price Divergence strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.types import Direction
from src.strategy.oi_diverge.config import OiDivergeConfig
from src.strategy.oi_diverge.preprocessor import preprocess
from src.strategy.oi_diverge.signal import generate_signals
from src.strategy.oi_diverge.strategy import OiDivergeStrategy
from src.strategy.tsmom.config import ShortMode


@pytest.fixture
def sample_ohlcv_with_derivatives() -> pd.DataFrame:
    """200일 OHLCV + derivatives 샘플."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    close = base + np.cumsum(np.random.randn(n) * 300)
    close = np.maximum(close, base * 0.5)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)

    # Derivatives: funding_rate 일부 극단값, open_interest 추세
    fr = np.random.randn(n) * 0.0005
    fr[50:60] = -0.003  # 극단 음수 (short crowding)
    fr[120:130] = 0.003  # 극단 양수 (long crowding)

    oi = 5e9 + np.cumsum(np.random.randn(n) * 1e8)
    oi = np.maximum(oi, 1e9)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
            "funding_rate": fr,
            "open_interest": oi,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def sample_ohlcv_only() -> pd.DataFrame:
    """200일 OHLCV only (derivatives 없음)."""
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
    """OiDivergeConfig 검증."""

    def test_defaults(self) -> None:
        cfg = OiDivergeConfig()
        assert cfg.divergence_window == 14
        assert cfg.fr_zscore_threshold == 1.5
        assert cfg.vol_target == 0.30
        assert cfg.short_mode == ShortMode.FULL

    def test_warmup(self) -> None:
        cfg = OiDivergeConfig()
        assert cfg.warmup_periods() == cfg.fr_zscore_window + 1  # 91

    def test_divergence_threshold_validation(self) -> None:
        """divergence_threshold > 0이면 에러."""
        with pytest.raises(ValueError, match="divergence_threshold"):
            OiDivergeConfig(divergence_threshold=0.5)

    def test_from_params(self) -> None:
        s = OiDivergeStrategy.from_params(divergence_window=21, vol_target=0.40)
        assert s.config.divergence_window == 21
        assert s.config.vol_target == 0.40

    def test_sweep_params(self) -> None:
        """Sweep 범위 내 파라미터 생성."""
        for dw in [7, 14, 21]:
            cfg = OiDivergeConfig(divergence_window=dw)
            assert cfg.divergence_window == dw


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Preprocessor 검증."""

    def test_adds_columns_with_derivatives(
        self, sample_ohlcv_with_derivatives: pd.DataFrame
    ) -> None:
        cfg = OiDivergeConfig()
        result = preprocess(sample_ohlcv_with_derivatives, cfg)
        expected = {"oi_price_div", "fr_zscore", "oi_roc", "realized_vol", "vol_scalar", "atr"}
        assert expected.issubset(set(result.columns))

    def test_without_derivatives(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """Derivatives 없으면 neutral 값."""
        cfg = OiDivergeConfig()
        result = preprocess(sample_ohlcv_only, cfg)
        assert (result["oi_price_div"] == 0.0).all()
        assert (result["fr_zscore"] == 0.0).all()

    def test_same_length(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        cfg = OiDivergeConfig()
        result = preprocess(sample_ohlcv_with_derivatives, cfg)
        assert len(result) == len(sample_ohlcv_with_derivatives)

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, OiDivergeConfig())


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Signal 생성 검증."""

    def test_output_shape(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        cfg = OiDivergeConfig()
        processed = preprocess(sample_ohlcv_with_derivatives, cfg)
        signals = generate_signals(processed, cfg)
        assert len(signals.direction) == len(sample_ohlcv_with_derivatives)
        assert len(signals.strength) == len(sample_ohlcv_with_derivatives)

    def test_direction_values(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        cfg = OiDivergeConfig()
        processed = preprocess(sample_ohlcv_with_derivatives, cfg)
        signals = generate_signals(processed, cfg)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_no_lookahead(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        """첫 번째 바 시그널은 0."""
        cfg = OiDivergeConfig()
        processed = preprocess(sample_ohlcv_with_derivatives, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.strength.iloc[0] == 0.0

    def test_no_signal_without_derivatives(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """Derivatives 없으면 시그널 없음."""
        cfg = OiDivergeConfig()
        processed = preprocess(sample_ohlcv_only, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction == 0).all()

    def test_short_mode_disabled(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        """DISABLED → 숏 시그널 없음."""
        cfg = OiDivergeConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_with_derivatives, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction != Direction.SHORT).all()

    def test_entries_exits_bool(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        cfg = OiDivergeConfig()
        processed = preprocess(sample_ohlcv_with_derivatives, cfg)
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

        cls = get_strategy("oi-diverge")
        assert cls is OiDivergeStrategy

    def test_name(self) -> None:
        assert OiDivergeStrategy().name == "OI-Diverge"

    def test_required_columns(self) -> None:
        s = OiDivergeStrategy()
        assert "close" in s.required_columns

    def test_run(self, sample_ohlcv_with_derivatives: pd.DataFrame) -> None:
        strategy = OiDivergeStrategy()
        processed, signals = strategy.run(sample_ohlcv_with_derivatives)
        assert len(signals.direction) == len(sample_ohlcv_with_derivatives)
        assert "oi_price_div" in processed.columns

    def test_warmup(self) -> None:
        assert OiDivergeStrategy().warmup_periods() > 0

    def test_recommended_config(self) -> None:
        rec = OiDivergeStrategy.recommended_config()
        assert rec["max_leverage_cap"] == 1.5

    def test_startup_info(self) -> None:
        info = OiDivergeStrategy().get_startup_info()
        assert "div_window" in info
        assert "mode" in info
