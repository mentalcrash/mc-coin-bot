"""Tests for Pod MTM daily return recording + Orchestrator day boundary detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod, PodPosition
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1)."""

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
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
) -> PodConfig:
    return PodConfig(
        pod_id=pod_id,
        strategy_name="tsmom",
        symbols=symbols,
        initial_fraction=0.50,
        max_fraction=0.70,
        min_fraction=0.02,
    )


def _make_pod(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.50,
) -> StrategyPod:
    config = _make_pod_config(pod_id=pod_id, symbols=symbols)
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    pod._warmup = 3
    return pod


def _make_orchestrator_config(
    pod_configs: tuple[PodConfig, ...] | None = None,
) -> OrchestratorConfig:
    if pod_configs is None:
        pod_configs = (
            _make_pod_config("pod-a", ("BTC/USDT",)),
            _make_pod_config("pod-b", ("ETH/USDT",)),
        )
    return OrchestratorConfig(
        pods=pod_configs,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
    )


def _make_orchestrator(
    config: OrchestratorConfig | None = None,
    pods: list[StrategyPod] | None = None,
) -> StrategyOrchestrator:
    if config is None:
        config = _make_orchestrator_config()
    if pods is None:
        pods = [_make_pod(pod_id=pc.pod_id, symbols=pc.symbols) for pc in config.pods]
    allocator = CapitalAllocator(config)
    return StrategyOrchestrator(
        config=config,
        pods=pods,
        allocator=allocator,
        target_timeframe="1D",
    )


# ══════════════════════════════════════════════════════════════════
# 1. Pod MTM PnL / Equity
# ══════════════════════════════════════════════════════════════════


class TestPodComputeTotalPnl:
    def test_long_position_profit(self) -> None:
        """Long 포지션: (close - entry) * qty = 양수 PnL."""
        pod = _make_pod()
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,  # Long
            realized_pnl=0.0,
        )
        pnl = pod._compute_total_pnl({"BTC/USDT": 55000.0})
        expected = (55000.0 - 50000.0) * 0.1  # 500.0
        assert pnl == pytest.approx(expected)

    def test_short_position_profit(self) -> None:
        """Short 포지션: (close - entry) * qty(음수) = 양수 PnL when price drops."""
        pod = _make_pod()
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=-0.1,  # Short
            realized_pnl=0.0,
        )
        pnl = pod._compute_total_pnl({"BTC/USDT": 45000.0})
        # unrealized = (45000 - 50000) * -0.1 = 500
        assert pnl == pytest.approx(500.0)

    def test_includes_realized_pnl(self) -> None:
        """realized + unrealized 합산."""
        pod = _make_pod()
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,
            realized_pnl=200.0,
        )
        pnl = pod._compute_total_pnl({"BTC/USDT": 52000.0})
        unrealized = (52000.0 - 50000.0) * 0.1  # 200.0
        assert pnl == pytest.approx(200.0 + unrealized)

    def test_no_price_uses_realized_only(self) -> None:
        """close price가 없으면 realized_pnl만."""
        pod = _make_pod()
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,
            realized_pnl=100.0,
        )
        pnl = pod._compute_total_pnl({})  # no prices
        assert pnl == pytest.approx(100.0)


class TestPodComputeMtmEquity:
    def test_equity_equals_base_plus_pnl(self) -> None:
        pod = _make_pod()
        pod._base_equity = 10000.0
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,
            realized_pnl=0.0,
        )
        equity = pod.compute_mtm_equity({"BTC/USDT": 51000.0})
        expected = 10000.0 + (51000.0 - 50000.0) * 0.1  # 10100
        assert equity == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════════
# 2. Pod record_daily_return_mtm
# ══════════════════════════════════════════════════════════════════


class TestPodRecordDailyReturnMtm:
    def test_consecutive_days(self) -> None:
        """연속 2일 equity 변화 → daily return 정확."""
        pod = _make_pod()
        pod._base_equity = 10000.0
        pod._prev_equity = 10000.0

        # Day 1: BTC 50000→52000, 0.1 qty → PnL=200 → equity=10200
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,
            realized_pnl=0.0,
        )
        pod.record_daily_return_mtm({"BTC/USDT": 52000.0})
        assert len(pod.daily_returns) == 1
        assert pod.daily_returns[0] == pytest.approx(200.0 / 10000.0)  # 2%
        assert pod._prev_equity == pytest.approx(10200.0)

        # Day 2: BTC 52000→54000 → PnL=400 → equity=10400
        pod.record_daily_return_mtm({"BTC/USDT": 54000.0})
        assert len(pod.daily_returns) == 2
        assert pod.daily_returns[1] == pytest.approx(200.0 / 10200.0)  # ~1.96%

    def test_no_positions_return_zero(self) -> None:
        """포지션 없으면 PnL=0 → return=0."""
        pod = _make_pod()
        pod._base_equity = 10000.0
        pod._prev_equity = 10000.0

        pod.record_daily_return_mtm({"BTC/USDT": 50000.0})
        assert len(pod.daily_returns) == 1
        assert pod.daily_returns[0] == pytest.approx(0.0)

    def test_prev_equity_zero_skip(self) -> None:
        """미초기화 (prev_equity ≈ 0) 시 record 스킵."""
        pod = _make_pod()
        # prev_equity = 0.0 (default)
        pod.record_daily_return_mtm({"BTC/USDT": 50000.0})
        assert len(pod.daily_returns) == 0


# ══════════════════════════════════════════════════════════════════
# 3. Pod set_base_equity
# ══════════════════════════════════════════════════════════════════


class TestPodSetBaseEquity:
    def test_sets_initial(self) -> None:
        pod = _make_pod()
        pod.set_base_equity(5000.0)
        assert pod._base_equity == pytest.approx(5000.0)
        assert pod._prev_equity == pytest.approx(5000.0)

    def test_skip_if_already_restored(self) -> None:
        """이미 복원된 base_equity > 0이면 덮어쓰지 않음."""
        pod = _make_pod()
        pod._base_equity = 7000.0
        pod._prev_equity = 7500.0
        pod.set_base_equity(5000.0)  # should be skipped
        assert pod._base_equity == pytest.approx(7000.0)
        assert pod._prev_equity == pytest.approx(7500.0)


# ══════════════════════════════════════════════════════════════════
# 4. Pod adjust_base_equity_on_rebalance
# ══════════════════════════════════════════════════════════════════


class TestPodAdjustBaseEquityOnRebalance:
    def test_rebalance_equity_continuity(self) -> None:
        """리밸런스 후 return에 spike 없음."""
        pod = _make_pod()
        pod._base_equity = 10000.0
        pod._prev_equity = 10200.0  # equity 상승 후

        # PnL = 200 (unrealized)
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,
            realized_pnl=0.0,
        )
        close_prices = {"BTC/USDT": 52000.0}

        # Rebalance: 새 base = 12000
        pod.adjust_base_equity_on_rebalance(12000.0, close_prices)

        # new_base = 12000, pnl = 200 → prev_equity = 12200
        assert pod._base_equity == pytest.approx(12000.0)
        assert pod._prev_equity == pytest.approx(12200.0)

        # 가격 변화 없으면 return ≈ 0
        equity = pod.compute_mtm_equity(close_prices)
        daily_return = (equity - pod._prev_equity) / pod._prev_equity
        assert daily_return == pytest.approx(0.0, abs=1e-10)


# ══════════════════════════════════════════════════════════════════
# 5. Orchestrator Day Boundary
# ══════════════════════════════════════════════════════════════════


class TestOrchestratorDayBoundary:
    @pytest.mark.asyncio
    async def test_day_boundary_4h_bars(self) -> None:
        """4H bar 6개 (1일) → 다음 날 첫 bar에서 1회 기록."""
        config = _make_orchestrator_config(
            pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),),
        )
        pod = _make_pod(pod_id="pod-a", symbols=("BTC/USDT",))
        orch = _make_orchestrator(config=config, pods=[pod])
        orch._target_timeframe = "4h"
        orch.set_initial_capital(100000.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)

        base_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

        # Day 1: 6 bars (0h, 4h, 8h, 12h, 16h, 20h)
        for h in range(0, 24, 4):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="4h",
                open=50000.0,
                high=50500.0,
                low=49500.0,
                close=50100.0,
                volume=1000.0,
                bar_timestamp=base_ts + timedelta(hours=h),
            )
            await orch._on_bar(bar)

        # No daily return yet (first day)
        assert len(pod.daily_returns) == 0

        # Day 2 first bar → triggers daily return recording for day 1
        bar_day2 = BarEvent(
            symbol="BTC/USDT",
            timeframe="4h",
            open=50100.0,
            high=50600.0,
            low=49600.0,
            close=50200.0,
            volume=1000.0,
            bar_timestamp=base_ts + timedelta(days=1),
        )
        await orch._on_bar(bar_day2)

        assert len(pod.daily_returns) == 1
        assert pod.performance.live_days == 1

    @pytest.mark.asyncio
    async def test_day_boundary_1d_bars(self) -> None:
        """1D bar → 매일 1회 기록."""
        config = _make_orchestrator_config(
            pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),),
        )
        pod = _make_pod(pod_id="pod-a", symbols=("BTC/USDT",))
        orch = _make_orchestrator(config=config, pods=[pod])
        orch.set_initial_capital(100000.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)

        base_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

        # 3 daily bars → 2 day transitions (day2 records day1, day3 records day2)
        for d in range(3):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0 + d * 100,
                high=50500.0 + d * 100,
                low=49500.0 + d * 100,
                close=50100.0 + d * 100,
                volume=1000.0,
                bar_timestamp=base_ts + timedelta(days=d),
            )
            await orch._on_bar(bar)

        assert len(pod.daily_returns) == 2

    @pytest.mark.asyncio
    async def test_first_day_no_record(self) -> None:
        """첫날은 기록하지 않음 (이전 close가 없으므로)."""
        config = _make_orchestrator_config(
            pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),),
        )
        pod = _make_pod(pod_id="pod-a", symbols=("BTC/USDT",))
        orch = _make_orchestrator(config=config, pods=[pod])
        orch.set_initial_capital(100000.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)

        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50100.0,
            volume=1000.0,
            bar_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        await orch._on_bar(bar)
        assert len(pod.daily_returns) == 0

    @pytest.mark.asyncio
    async def test_close_price_ordering(self) -> None:
        """Day boundary 시 전일 close price 사용 확인 (당일 close X).

        Day 1 close=50100 → Day 2 bar arrives with close=51000
        MTM은 day 1 close (50100)로 계산해야 함.
        """
        config = _make_orchestrator_config(
            pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),),
        )
        pod = _make_pod(pod_id="pod-a", symbols=("BTC/USDT",))
        orch = _make_orchestrator(config=config, pods=[pod])
        orch.set_initial_capital(100000.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)

        # Day 1 bar
        bar_day1 = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50100.0,
            volume=1000.0,
            bar_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        await orch._on_bar(bar_day1)

        # last_close_prices should be day 1 close
        assert orch._last_close_prices["BTC/USDT"] == pytest.approx(50100.0)

        # Pod에 포지션 부여 (entry=50000, qty=0.1)
        pod._positions["BTC/USDT"] = PodPosition(
            pod_id="pod-a",
            symbol="BTC/USDT",
            avg_entry_price=50000.0,
            quantity=0.1,
            realized_pnl=0.0,
        )

        # Day 2 bar (close=51000 — 이 가격은 MTM 계산에 사용되지 않아야 함)
        bar_day2 = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50100.0,
            high=51500.0,
            low=50000.0,
            close=51000.0,
            volume=1000.0,
            bar_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        )
        await orch._on_bar(bar_day2)

        # daily return은 day1 close (50100)로 MTM 계산
        # PnL = (50100 - 50000) * 0.1 = 10
        # equity = 50000 + 10 = 50010, prev_equity = 50000
        # return = 10 / 50000 = 0.0002
        assert len(pod.daily_returns) == 1
        expected_return = 10.0 / 50000.0
        assert pod.daily_returns[0] == pytest.approx(expected_return)

    @pytest.mark.asyncio
    async def test_flush_daily_returns(self) -> None:
        """flush_daily_returns로 마지막 일 강제 기록."""
        config = _make_orchestrator_config(
            pod_configs=(_make_pod_config("pod-a", ("BTC/USDT",)),),
        )
        pod = _make_pod(pod_id="pod-a", symbols=("BTC/USDT",))
        orch = _make_orchestrator(config=config, pods=[pod])
        orch.set_initial_capital(100000.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)

        # Day 1 bar
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50100.0,
            volume=1000.0,
            bar_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        await orch._on_bar(bar)

        # 아직 day boundary 없으므로 기록 없음
        assert len(pod.daily_returns) == 0

        # Flush → 마지막 일 기록
        orch.flush_daily_returns()
        assert len(pod.daily_returns) == 1

    @pytest.mark.asyncio
    async def test_flush_no_op_when_no_data(self) -> None:
        """데이터 없을 때 flush는 no-op."""
        orch = _make_orchestrator()
        orch.flush_daily_returns()  # should not raise


# ══════════════════════════════════════════════════════════════════
# 6. State Persistence Round-trip
# ══════════════════════════════════════════════════════════════════


class TestStatePersistenceRoundTrip:
    def test_pod_base_prev_equity_serialized(self) -> None:
        """base_equity, prev_equity가 to_dict/restore_from_dict에 포함."""
        pod = _make_pod()
        pod._base_equity = 5000.0
        pod._prev_equity = 5200.0

        data = pod.to_dict()
        assert data["base_equity"] == pytest.approx(5000.0)
        assert data["prev_equity"] == pytest.approx(5200.0)

        pod2 = _make_pod()
        pod2.restore_from_dict(data)
        assert pod2._base_equity == pytest.approx(5000.0)
        assert pod2._prev_equity == pytest.approx(5200.0)

    def test_orchestrator_mtm_state_serialized(self) -> None:
        """initial_capital, last_day_date, last_close_prices 직렬화."""
        from datetime import date

        orch = _make_orchestrator()
        orch._initial_capital = 100000.0
        orch._last_day_date = date(2026, 2, 15)
        orch._last_close_prices = {"BTC/USDT": 50100.0, "ETH/USDT": 3200.0}

        data = orch.to_dict()
        assert data["initial_capital"] == pytest.approx(100000.0)
        assert data["last_day_date"] == "2026-02-15"
        assert data["last_close_prices"]["BTC/USDT"] == pytest.approx(50100.0)

        orch2 = _make_orchestrator()
        orch2.restore_from_dict(data)
        assert orch2._initial_capital == pytest.approx(100000.0)
        assert orch2._last_day_date == date(2026, 2, 15)
        assert orch2._last_close_prices["BTC/USDT"] == pytest.approx(50100.0)
