"""Tests for StrategyPod — 전략별 독립 실행 단위."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.orchestrator.config import PodConfig
from src.orchestrator.models import LifecycleState
from src.orchestrator.pod import _MAX_BUFFER_SIZE, StrategyPod
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1). strength = abs(close-open)/open."""

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0.01)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(**overrides: object) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": "pod-a",
        "strategy_name": "tsmom",
        "symbols": ("BTC/USDT",),
        "initial_fraction": 0.10,
        "max_fraction": 0.40,
        "min_fraction": 0.02,
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_pod(
    *,
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.25,
    warmup: int = 3,
    pod_id: str = "pod-a",
) -> StrategyPod:
    """테스트용 StrategyPod 생성. warmup을 낮게 설정."""
    config = _make_pod_config(pod_id=pod_id, symbols=symbols)
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    # warmup을 테스트에 유리하게 오버라이드
    pod._warmup = warmup
    return pod


def _make_bar_data(open_: float, close: float) -> dict[str, float]:
    return {
        "open": open_,
        "high": max(open_, close) * 1.01,
        "low": min(open_, close) * 0.99,
        "close": close,
        "volume": 1000.0,
    }


def _feed_warmup(pod: StrategyPod, symbol: str, n: int, base_ts: datetime) -> datetime:
    """n개 bar를 Pod에 공급하여 warmup을 채운다. 마지막 ts 반환."""
    ts = base_ts
    for i in range(n):
        bar = _make_bar_data(open_=100.0 + i, close=101.0 + i)
        pod.compute_signal(symbol, bar, ts)
        ts += timedelta(days=1)
    return ts


# ── TestPodCreation ───────────────────────────────────────────────


class TestPodCreation:
    def test_creates_with_defaults(self) -> None:
        pod = _make_pod()
        assert pod is not None

    def test_pod_id_matches_config(self) -> None:
        pod = _make_pod(pod_id="my-pod")
        assert pod.pod_id == "my-pod"

    def test_symbols_match_config(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT", "ETH/USDT"))
        assert pod.symbols == ("BTC/USDT", "ETH/USDT")

    def test_initial_state_incubation(self) -> None:
        pod = _make_pod()
        assert pod.state == LifecycleState.INCUBATION

    def test_initial_capital_fraction(self) -> None:
        pod = _make_pod(capital_fraction=0.3)
        assert pod.capital_fraction == pytest.approx(0.3)


# ── TestPodSignalComputation ─────────────────────────────────────


class TestPodSignalComputation:
    def test_no_signal_before_warmup(self) -> None:
        """warmup 미충족 시 None 반환."""
        pod = _make_pod(warmup=5)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        bar = _make_bar_data(100.0, 105.0)
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is None

    def test_signal_after_warmup(self) -> None:
        """warmup 충족 후 (direction, strength) 반환."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        # warmup 채우기 (3개 bar)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        # 3번째 bar → 시그널 생성
        bar = _make_bar_data(100.0, 105.0)
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None
        direction, strength = result
        assert isinstance(direction, int)
        assert isinstance(strength, float)

    def test_signal_direction_long(self) -> None:
        """close > open → LONG (+1)."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)  # close > open
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None
        direction, _ = result
        assert direction == 1

    def test_signal_direction_short(self) -> None:
        """close < open → SHORT (-1)."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(110.0, 100.0)  # close < open
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None
        direction, _ = result
        assert direction == -1

    def test_target_weight_stored(self) -> None:
        """compute_signal 후 _target_weights 업데이트."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)
        pod.compute_signal("BTC/USDT", bar, ts)
        weights = pod.get_target_weights()
        assert "BTC/USDT" in weights
        assert weights["BTC/USDT"] != 0.0


# ── TestPodGlobalWeights ────────────────────────────────────────


class TestPodGlobalWeights:
    def test_global_weights_scaled(self) -> None:
        """global_weight = internal * capital_fraction."""
        pod = _make_pod(capital_fraction=0.5, warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)
        pod.compute_signal("BTC/USDT", bar, ts)

        internal = pod.get_target_weights()["BTC/USDT"]
        global_ = pod.get_global_weights()["BTC/USDT"]
        assert global_ == pytest.approx(internal * 0.5)

    def test_global_weights_zero_fraction(self) -> None:
        """capital_fraction=0 → 모든 global weight = 0."""
        pod = _make_pod(capital_fraction=0.0, warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)
        pod.compute_signal("BTC/USDT", bar, ts)

        for v in pod.get_global_weights().values():
            assert v == pytest.approx(0.0)

    def test_global_weights_multiple_symbols(self) -> None:
        """멀티 심볼 global weights."""
        pod = _make_pod(
            symbols=("BTC/USDT", "ETH/USDT"),
            capital_fraction=0.4,
            warmup=3,
        )
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        # 두 심볼 warmup + 시그널
        for sym in ("BTC/USDT", "ETH/USDT"):
            sym_ts = ts
            for i in range(3):
                bar = _make_bar_data(100.0 + i, 101.0 + i)
                pod.compute_signal(sym, bar, sym_ts)
                sym_ts += timedelta(days=1)

        gw = pod.get_global_weights()
        assert len(gw) == 2
        assert "BTC/USDT" in gw
        assert "ETH/USDT" in gw


# ── TestPodPositionUpdate ────────────────────────────────────────


class TestPodPositionUpdate:
    def test_position_created_on_first_fill(self) -> None:
        """첫 체결 시 포지션 생성."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 0.1, 50000.0, 5.0, is_buy=True)
        assert "BTC/USDT" in pod._positions
        pos = pod._positions["BTC/USDT"]
        assert pos.notional_usd == pytest.approx(0.1 * 50000.0)
        assert pod.performance.trade_count == 1

    def test_trade_count_increments(self) -> None:
        """체결마다 trade_count 증가."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 0.1, 50000.0, 5.0, is_buy=True)
        pod.update_position("BTC/USDT", 0.05, 51000.0, 3.0, is_buy=False)
        assert pod.performance.trade_count == 2


# ── TestPodHelpers ──────────────────────────────────────────────


class TestPodHelpers:
    def test_accepts_symbol_true(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        assert pod.accepts_symbol("BTC/USDT") is True

    def test_accepts_symbol_false(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        assert pod.accepts_symbol("ETH/USDT") is False

    def test_is_active_vs_retired(self) -> None:
        pod = _make_pod()
        assert pod.is_active is True
        pod.state = LifecycleState.RETIRED
        assert pod.is_active is False

    def test_buffer_size_limit(self) -> None:
        """MAX_BUFFER_SIZE 초과 시 오래된 데이터 trim."""
        pod = _make_pod(warmup=1)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(_MAX_BUFFER_SIZE + 50):
            bar = _make_bar_data(100.0 + i * 0.01, 101.0 + i * 0.01)
            pod.compute_signal("BTC/USDT", bar, ts)
            ts += timedelta(days=1)
        assert len(pod._buffers["BTC/USDT"]) == _MAX_BUFFER_SIZE

    def test_record_daily_return(self) -> None:
        pod = _make_pod()
        pod.record_daily_return(0.01)
        pod.record_daily_return(-0.005)
        assert pod.performance.live_days == 2
        assert len(pod.daily_returns_series) == 2

    def test_inject_warmup(self) -> None:
        """inject_warmup으로 버퍼 주입."""
        pod = _make_pod(warmup=5)
        ts_list = [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(5)]
        bars = [_make_bar_data(100.0 + i, 101.0 + i) for i in range(5)]
        pod.inject_warmup("BTC/USDT", bars, ts_list)
        assert len(pod._buffers["BTC/USDT"]) == 5

    def test_inject_warmup_length_mismatch(self) -> None:
        pod = _make_pod()
        with pytest.raises(ValueError, match="length mismatch"):
            pod.inject_warmup("BTC/USDT", [_make_bar_data(100, 101)], [])

    def test_compute_signal_wrong_symbol_returns_none(self) -> None:
        """Pod에 등록되지 않은 심볼 → None."""
        pod = _make_pod(symbols=("BTC/USDT",))
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        result = pod.compute_signal("ETH/USDT", _make_bar_data(100, 101), ts)
        assert result is None
