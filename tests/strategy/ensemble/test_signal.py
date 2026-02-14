"""Unit tests for ensemble signal generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategy.ensemble.config import (
    EnsembleConfig,
    ShortMode,
    SubStrategySpec,
)
from src.strategy.ensemble.signal import _collect_sub_signals, generate_signals
from src.strategy.types import StrategySignals


def _make_mock_strategy(
    direction_val: int, strength_val: float, n: int, index: pd.DatetimeIndex
) -> MagicMock:
    """Mock BaseStrategy with fixed signals."""
    mock = MagicMock()
    mock.run.return_value = (
        pd.DataFrame(index=index),
        StrategySignals(
            entries=pd.Series(False, index=index),
            exits=pd.Series(False, index=index),
            direction=pd.Series(direction_val, index=index, dtype=int),
            strength=pd.Series(strength_val, index=index, dtype=float),
        ),
    )
    return mock


def _make_preprocessed_df(n: int) -> pd.DataFrame:
    """vol_scalar 포함된 preprocessed DataFrame 생성."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1D")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame(
        {
            "open": close + 10,
            "high": close + 100,
            "low": close - 100,
            "close": close,
            "volume": np.random.uniform(100, 10000, n),
            "realized_vol": 0.5,
            "vol_scalar": 0.7,
        },
        index=idx,
    )


class TestCollectSubSignals:
    """_collect_sub_signals 검증."""

    def test_collects_from_multiple_strategies(self) -> None:
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)

        s1 = _make_mock_strategy(1, 0.5, n, idx)
        s2 = _make_mock_strategy(-1, -0.3, n, idx)

        directions, strengths = _collect_sub_signals(df, [s1, s2], ["s1", "s2"])
        assert directions.shape == (n, 2)
        assert strengths.shape == (n, 2)
        assert list(directions.columns) == ["s1", "s2"]

    def test_skips_failed_strategy(self) -> None:
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)

        s1 = _make_mock_strategy(1, 0.5, n, idx)
        s2 = MagicMock()
        s2.run.side_effect = RuntimeError("strategy failed")

        directions, _strengths = _collect_sub_signals(df, [s1, s2], ["s1", "s2"])
        assert directions.shape == (n, 1)
        assert list(directions.columns) == ["s1"]

    def test_all_failed_raises(self) -> None:
        n = 50
        df = _make_preprocessed_df(n)

        s1 = MagicMock()
        s1.run.side_effect = RuntimeError("fail")
        s2 = MagicMock()
        s2.run.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="All sub-strategies failed"):
            _collect_sub_signals(df, [s1, s2], ["s1", "s2"])

    def test_passes_df_copy(self) -> None:
        """각 전략에 df.copy()가 전달되는지 확인."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)

        s1 = _make_mock_strategy(1, 0.5, n, idx)
        _collect_sub_signals(df, [s1], ["s1"])

        # run() 호출 시 전달된 df가 원본과 다른 객체인지 확인
        passed_df = s1.run.call_args[0][0]
        assert passed_df is not df


class TestGenerateSignals:
    """generate_signals 통합 검증."""

    def _make_config(self, **kwargs: object) -> EnsembleConfig:
        specs = (
            SubStrategySpec(name="mock_a"),
            SubStrategySpec(name="mock_b"),
        )
        return EnsembleConfig(strategies=specs, **kwargs)  # type: ignore[arg-type]

    def test_basic_signal_generation(self) -> None:
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)
        config = self._make_config()

        s1 = _make_mock_strategy(1, 0.5, n, idx)
        s2 = _make_mock_strategy(1, 0.3, n, idx)
        weights = pd.Series({"mock_a": 1.0, "mock_b": 1.0})

        signals = generate_signals(df, [s1, s2], ["mock_a", "mock_b"], weights, config)

        assert isinstance(signals, StrategySignals)
        assert len(signals.direction) == n
        assert len(signals.strength) == n
        assert len(signals.entries) == n
        assert len(signals.exits) == n

    def test_short_mode_disabled_filters_shorts(self) -> None:
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)
        config = self._make_config(short_mode=ShortMode.DISABLED)

        # 두 전략 모두 short
        s1 = _make_mock_strategy(-1, -0.5, n, idx)
        s2 = _make_mock_strategy(-1, -0.3, n, idx)
        weights = pd.Series({"mock_a": 1.0, "mock_b": 1.0})

        signals = generate_signals(df, [s1, s2], ["mock_a", "mock_b"], weights, config)

        # short_mode=DISABLED → 모든 short → neutral
        assert (signals.direction == 0).all()

    def test_short_mode_full_allows_shorts(self) -> None:
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)
        config = self._make_config(short_mode=ShortMode.FULL)

        s1 = _make_mock_strategy(-1, -0.5, n, idx)
        s2 = _make_mock_strategy(-1, -0.3, n, idx)
        weights = pd.Series({"mock_a": 1.0, "mock_b": 1.0})

        signals = generate_signals(df, [s1, s2], ["mock_a", "mock_b"], weights, config)

        # short_mode=FULL → short 허용
        assert (signals.direction == -1).all()

    def test_entries_on_direction_change(self) -> None:
        """방향 전환 시 entry 발생."""
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = _make_preprocessed_df(n)
        config = self._make_config()

        # 전반 5개 neutral, 후반 5개 long
        dir_seq = [0] * 5 + [1] * 5
        str_seq = [0.0] * 5 + [0.5] * 5

        s1 = MagicMock()
        s1.run.return_value = (
            pd.DataFrame(index=idx),
            StrategySignals(
                entries=pd.Series(False, index=idx),
                exits=pd.Series(False, index=idx),
                direction=pd.Series(dir_seq, index=idx, dtype=int),
                strength=pd.Series(str_seq, index=idx, dtype=float),
            ),
        )
        s2 = MagicMock()
        s2.run.return_value = s1.run.return_value
        weights = pd.Series({"mock_a": 1.0, "mock_b": 1.0})

        signals = generate_signals(df, [s1, s2], ["mock_a", "mock_b"], weights, config)

        # index 5에서 0→1 전환 → entry True
        assert signals.entries.iloc[5]
