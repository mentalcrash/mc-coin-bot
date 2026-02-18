"""Tests for Hurst-Adaptive strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.types import Direction
from src.strategy.hurst_adaptive.config import HurstAdaptiveConfig
from src.strategy.hurst_adaptive.preprocessor import preprocess
from src.strategy.hurst_adaptive.signal import generate_signals
from src.strategy.hurst_adaptive.strategy import HurstAdaptiveStrategy
from src.strategy.tsmom.config import ShortMode


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200일 OHLCV 샘플 (trending pattern)."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    # 약간의 추세를 넣어 hurst > 0.5 유도
    trend = np.linspace(0, 3000, n)
    noise = np.cumsum(np.random.randn(n) * 100)
    close = base + trend + noise
    close = np.maximum(close, base * 0.5)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    """HurstAdaptiveConfig 검증."""

    def test_defaults(self) -> None:
        cfg = HurstAdaptiveConfig()
        assert cfg.hurst_window == 100
        assert cfg.er_period == 20
        assert cfg.vol_target == 0.35
        assert cfg.short_mode == ShortMode.HEDGE_ONLY

    def test_warmup(self) -> None:
        cfg = HurstAdaptiveConfig()
        assert cfg.warmup_periods() == cfg.hurst_window + 1  # 100 is max

    def test_threshold_validation(self) -> None:
        """hurst_trend > hurst_mr 위반 시 에러."""
        with pytest.raises(ValueError, match="hurst_trend_threshold"):
            HurstAdaptiveConfig(hurst_trend_threshold=0.40, hurst_mr_threshold=0.50)

    def test_equal_thresholds_rejected(self) -> None:
        """동일 threshold도 거부."""
        with pytest.raises(ValueError, match="hurst_trend_threshold"):
            HurstAdaptiveConfig(hurst_trend_threshold=0.50, hurst_mr_threshold=0.50)

    def test_from_params(self) -> None:
        s = HurstAdaptiveStrategy.from_params(hurst_window=80, vol_target=0.25)
        assert s.config.hurst_window == 80
        assert s.config.vol_target == 0.25

    def test_sweep_params(self) -> None:
        """Sweep 범위 내 파라미터 생성."""
        for hw in [60, 80, 100, 120, 150]:
            cfg = HurstAdaptiveConfig(hurst_window=hw)
            assert cfg.hurst_window == hw


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Preprocessor 검증."""

    def test_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = HurstAdaptiveConfig()
        result = preprocess(sample_ohlcv, cfg)
        expected_cols = {
            "hurst",
            "er",
            "trend_momentum",
            "mr_score",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_preserves_original_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = HurstAdaptiveConfig()
        result = preprocess(sample_ohlcv, cfg)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_same_length(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = HurstAdaptiveConfig()
        result = preprocess(sample_ohlcv, cfg)
        assert len(result) == len(sample_ohlcv)

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, HurstAdaptiveConfig())

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar는 NaN을 제외하면 양수."""
        cfg = HurstAdaptiveConfig()
        result = preprocess(sample_ohlcv, cfg)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_hurst_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """hurst는 0~1 범위."""
        cfg = HurstAdaptiveConfig()
        result = preprocess(sample_ohlcv, cfg)
        valid_hurst = result["hurst"].dropna()
        assert (valid_hurst >= 0).all()
        assert (valid_hurst <= 1).all()

    def test_er_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """efficiency_ratio는 0~1 범위."""
        cfg = HurstAdaptiveConfig()
        result = preprocess(sample_ohlcv, cfg)
        valid_er = result["er"].dropna()
        assert (valid_er >= 0).all()
        assert (valid_er <= 1).all()


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Signal 생성 검증."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = HurstAdaptiveConfig()
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert len(signals.direction) == len(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.exits) == len(sample_ohlcv)

    def test_direction_values(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = HurstAdaptiveConfig()
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_entries_exits_bool(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = HurstAdaptiveConfig()
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_no_lookahead(self, sample_ohlcv: pd.DataFrame) -> None:
        """첫 번째 바의 시그널은 0 (shift(1) 적용)."""
        cfg = HurstAdaptiveConfig()
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.strength.iloc[0] == 0.0

    def test_missing_columns_error(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            generate_signals(df, HurstAdaptiveConfig())

    def test_short_mode_disabled(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED → 숏 시그널 없음."""
        cfg = HurstAdaptiveConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction != Direction.SHORT).all()

    def test_short_mode_full(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL → 숏 시그널 허용 (있을 수 있음)."""
        cfg = HurstAdaptiveConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        # FULL 모드에서는 direction에 -1이 있을 수 있음 (데이터 의존)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class TestStrategy:
    """Strategy 클래스 통합 검증."""

    def test_registry(self) -> None:
        from src.strategy.registry import get_strategy

        cls = get_strategy("hurst-adaptive")
        assert cls is HurstAdaptiveStrategy

    def test_name(self) -> None:
        assert HurstAdaptiveStrategy().name == "Hurst-Adaptive"

    def test_required_columns(self) -> None:
        s = HurstAdaptiveStrategy()
        assert "close" in s.required_columns
        assert "high" in s.required_columns

    def test_run(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = HurstAdaptiveStrategy()
        processed, signals = strategy.run(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert "hurst" in processed.columns

    def test_warmup(self) -> None:
        s = HurstAdaptiveStrategy()
        assert s.warmup_periods() > 0

    def test_recommended_config(self) -> None:
        rec = HurstAdaptiveStrategy.recommended_config()
        assert "max_leverage_cap" in rec
        assert "trailing_stop_atr_multiplier" in rec

    def test_startup_info(self) -> None:
        info = HurstAdaptiveStrategy().get_startup_info()
        assert "hurst_window" in info
        assert "mode" in info
