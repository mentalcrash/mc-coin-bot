"""Tests for On-chain Accumulation strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.types import Direction
from src.strategy.onchain_accum.config import OnchainAccumConfig
from src.strategy.onchain_accum.preprocessor import preprocess
from src.strategy.onchain_accum.signal import generate_signals
from src.strategy.onchain_accum.strategy import OnchainAccumStrategy
from src.strategy.tsmom.config import ShortMode


@pytest.fixture
def sample_ohlcv_with_onchain() -> pd.DataFrame:
    """200일 OHLCV + on-chain 샘플."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    close = base + np.cumsum(np.random.randn(n) * 300)
    close = np.maximum(close, base * 0.5)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)

    # MVRV: 초기 저평가, 후기 고평가
    mvrv = np.linspace(0.7, 3.5, n) + np.random.randn(n) * 0.3
    mvrv = np.maximum(mvrv, 0.3)

    # Exchange flows
    flow_in = 1e9 + np.random.randn(n) * 1e8
    flow_out = 1e9 + np.random.randn(n) * 1e8
    # 초기: outflow 우세 (accumulation)
    flow_out[:50] += 3e8

    # Stablecoin total
    stablecoin = 150e9 + np.cumsum(np.random.randn(n) * 1e9)
    stablecoin = np.maximum(stablecoin, 100e9)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
            "oc_mvrv": mvrv,
            "oc_flow_in_ex_usd": flow_in,
            "oc_flow_out_ex_usd": flow_out,
            "oc_stablecoin_total_usd": stablecoin,
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
    """OnchainAccumConfig 검증."""

    def test_defaults(self) -> None:
        cfg = OnchainAccumConfig()
        assert cfg.mvrv_undervalued == 1.0
        assert cfg.flow_threshold == 1.0
        assert cfg.stablecoin_roc_threshold == 0.02
        assert cfg.short_mode == ShortMode.DISABLED

    def test_warmup(self) -> None:
        cfg = OnchainAccumConfig()
        assert cfg.warmup_periods() == cfg.flow_zscore_window + 1  # 91

    def test_mvrv_validation(self) -> None:
        """mvrv_undervalued >= mvrv_overvalued이면 에러."""
        with pytest.raises(ValueError, match="mvrv_undervalued"):
            OnchainAccumConfig(mvrv_undervalued=3.0, mvrv_overvalued=2.0)

    def test_equal_mvrv_rejected(self) -> None:
        with pytest.raises(ValueError, match="mvrv_undervalued"):
            OnchainAccumConfig(mvrv_undervalued=2.0, mvrv_overvalued=2.0)

    def test_from_params(self) -> None:
        s = OnchainAccumStrategy.from_params(mvrv_undervalued=0.8, vol_target=0.25)
        assert s.config.mvrv_undervalued == 0.8

    def test_sweep_params(self) -> None:
        for mu in [0.8, 1.0, 1.2]:
            cfg = OnchainAccumConfig(mvrv_undervalued=mu)
            assert cfg.mvrv_undervalued == mu


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Preprocessor 검증."""

    def test_adds_columns_with_onchain(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        cfg = OnchainAccumConfig()
        result = preprocess(sample_ohlcv_with_onchain, cfg)
        expected = {
            "net_flow",
            "net_flow_zscore",
            "stablecoin_roc",
            "realized_vol",
            "vol_scalar",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_without_onchain(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """On-chain 없으면 NaN 컬럼."""
        cfg = OnchainAccumConfig()
        result = preprocess(sample_ohlcv_only, cfg)
        assert result["net_flow_zscore"].isna().all()
        assert result["stablecoin_roc"].isna().all()

    def test_same_length(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        cfg = OnchainAccumConfig()
        result = preprocess(sample_ohlcv_with_onchain, cfg)
        assert len(result) == len(sample_ohlcv_with_onchain)

    def test_missing_ohlcv_columns(self) -> None:
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, OnchainAccumConfig())


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Signal 생성 검증."""

    def test_output_shape(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        cfg = OnchainAccumConfig()
        processed = preprocess(sample_ohlcv_with_onchain, cfg)
        signals = generate_signals(processed, cfg)
        assert len(signals.direction) == len(sample_ohlcv_with_onchain)
        assert len(signals.strength) == len(sample_ohlcv_with_onchain)

    def test_direction_values(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        cfg = OnchainAccumConfig()
        processed = preprocess(sample_ohlcv_with_onchain, cfg)
        signals = generate_signals(processed, cfg)
        # DISABLED short mode → only 0, 1
        assert set(signals.direction.unique()).issubset({0, 1})

    def test_no_lookahead(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        """첫 번째 바 시그널은 0."""
        cfg = OnchainAccumConfig()
        processed = preprocess(sample_ohlcv_with_onchain, cfg)
        signals = generate_signals(processed, cfg)
        assert signals.strength.iloc[0] == 0.0

    def test_no_signal_without_onchain(self, sample_ohlcv_only: pd.DataFrame) -> None:
        """On-chain 없으면 시그널 없음 (모든 score NaN → 0)."""
        cfg = OnchainAccumConfig()
        processed = preprocess(sample_ohlcv_only, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction == 0).all()

    def test_long_only_default(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        """기본 DISABLED → 숏 없음."""
        cfg = OnchainAccumConfig()
        processed = preprocess(sample_ohlcv_with_onchain, cfg)
        signals = generate_signals(processed, cfg)
        assert (signals.direction != Direction.SHORT).all()

    def test_majority_vote_logic(self) -> None:
        """2/3 합의 로직 직접 검증."""
        n = 10
        df = pd.DataFrame(
            {
                "close": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "volume": [1000.0] * n,
                "oc_mvrv": [0.5] * n,  # undervalued → +1
                "oc_flow_in_ex_usd": [100.0] * n,
                "oc_flow_out_ex_usd": [100.0] * n,  # neutral → 0
                "oc_stablecoin_total_usd": [100.0] * n,  # stable → 0
                "vol_scalar": [1.0] * n,
                "net_flow": [0.0] * n,
                "net_flow_zscore": [0.0] * n,
                "stablecoin_roc": [0.0] * n,
                "realized_vol": [0.2] * n,
                "atr": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        cfg = OnchainAccumConfig()
        signals = generate_signals(df, cfg)
        # 1개만 +1이면 합의 미달 → direction = 0
        assert (signals.direction == 0).all()

    def test_entries_exits_bool(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        cfg = OnchainAccumConfig()
        processed = preprocess(sample_ohlcv_with_onchain, cfg)
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

        cls = get_strategy("onchain-accum")
        assert cls is OnchainAccumStrategy

    def test_name(self) -> None:
        assert OnchainAccumStrategy().name == "Onchain-Accum"

    def test_required_columns(self) -> None:
        s = OnchainAccumStrategy()
        assert "close" in s.required_columns

    def test_run(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        strategy = OnchainAccumStrategy()
        _processed, signals = strategy.run(sample_ohlcv_with_onchain)
        assert len(signals.direction) == len(sample_ohlcv_with_onchain)

    def test_warmup(self) -> None:
        assert OnchainAccumStrategy().warmup_periods() > 0

    def test_recommended_config(self) -> None:
        rec = OnchainAccumStrategy.recommended_config()
        assert rec["trailing_stop_atr_multiplier"] == 3.5

    def test_startup_info(self) -> None:
        info = OnchainAccumStrategy().get_startup_info()
        assert "mvrv_range" in info
        assert "mode" in info
