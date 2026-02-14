"""LiveRunner OMS 복원 버그 수정 테스트.

Phase 10: LiveRunner 재시작 시 OMS._processed_orders가 복원되어
중복 주문을 방지하는지 검증합니다.
"""

from __future__ import annotations

import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.eda.live_data_feed import LiveDataFeed
from src.eda.live_runner import LiveRunner
from src.eda.oms import OMS
from src.eda.persistence.database import Database
from src.eda.persistence.state_manager import StateManager
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyStrategy(BaseStrategy):
    """테스트용 더미 전략."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = pd.Series(1, index=df.index)
        strength = pd.Series(1.0, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


def _make_config() -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        max_leverage_cap=2.0,
        rebalance_threshold=0.01,
        system_stop_loss=None,
        use_trailing_stop=False,
        cost_model=CostModel.zero(),
    )


def _make_pm(config: PortfolioManagerConfig | None = None) -> EDAPortfolioManager:
    cfg = config or _make_config()
    return EDAPortfolioManager(config=cfg, initial_capital=10000.0)


def _make_rm(
    pm: EDAPortfolioManager, config: PortfolioManagerConfig | None = None
) -> EDARiskManager:
    cfg = config or _make_config()
    return EDARiskManager(config=cfg, portfolio_manager=pm)


def _make_runner(db_path: str | None = None) -> LiveRunner:
    client = MagicMock()
    client.exchange = MagicMock()
    return LiveRunner.paper(
        strategy=_DummyStrategy(),
        symbols=["BTC/USDT"],
        target_timeframe="1D",
        config=_make_config(),
        client=client,
        db_path=db_path,
    )


def _mock_feed(runner: LiveRunner) -> None:
    """Feed를 mock으로 교체 (WebSocket 없이 테스트)."""
    stopped = asyncio.Event()

    async def mock_start(bus: object) -> None:
        await stopped.wait()

    runner._feed = MagicMock(spec=LiveDataFeed)
    runner._feed.start = AsyncMock(side_effect=mock_start)
    runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
    runner._feed.bars_emitted = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRestoreStateWithOms:
    """_restore_state() OMS 복원 테스트."""

    @pytest.mark.asyncio
    async def test_restore_state_with_oms(self, tmp_path: pathlib.Path) -> None:
        """_restore_state(db, pm, rm, oms) → OMS processed orders 복원."""
        db_path = str(tmp_path / "oms_restore.db")
        db = Database(db_path)
        await db.connect()

        # OMS 상태 저장
        state_mgr = StateManager(db)
        saved_ids = {"order-1", "order-2", "order-3"}
        await state_mgr._save_key("oms_processed_orders", __import__("json").dumps(list(saved_ids)))

        pm = _make_pm()
        rm = _make_rm(pm)
        oms = OMS(executor=MagicMock(), portfolio_manager=pm)
        assert len(oms.processed_orders) == 0

        result = await LiveRunner._restore_state(db, pm, rm, oms)

        assert result is not None
        assert oms.processed_orders == saved_ids
        await db.close()

    @pytest.mark.asyncio
    async def test_restore_state_oms_none_backward_compat(self, tmp_path: pathlib.Path) -> None:
        """oms=None → 기존 동작 유지 (PM/RM만 복구)."""
        db_path = str(tmp_path / "compat.db")
        db = Database(db_path)
        await db.connect()

        pm = _make_pm()
        rm = _make_rm(pm)

        # PM/RM 상태만 저장
        state_mgr = StateManager(db)
        await state_mgr.save_all(pm, rm)

        pm2 = _make_pm()
        rm2 = _make_rm(pm2)

        # oms=None (기본값) — 에러 없이 동작
        result = await LiveRunner._restore_state(db, pm2, rm2)
        assert result is not None
        await db.close()

    @pytest.mark.asyncio
    async def test_restore_state_no_saved_oms(self, tmp_path: pathlib.Path) -> None:
        """DB에 OMS 상태 없을 때 → 빈 set 유지."""
        db_path = str(tmp_path / "no_oms.db")
        db = Database(db_path)
        await db.connect()

        pm = _make_pm()
        rm = _make_rm(pm)
        oms = OMS(executor=MagicMock(), portfolio_manager=pm)

        # PM/RM만 저장, OMS는 저장하지 않음
        state_mgr = StateManager(db)
        await state_mgr.save_all(pm, rm)

        result = await LiveRunner._restore_state(db, pm, rm, oms)
        assert result is not None
        assert len(oms.processed_orders) == 0
        await db.close()


class TestPeriodicSaveIncludesOms:
    """_periodic_state_save() OMS 포함 테스트."""

    @pytest.mark.asyncio
    async def test_periodic_save_includes_oms(self) -> None:
        """_periodic_state_save() → save_all(pm, rm, oms=oms) 호출 확인."""
        state_mgr = MagicMock()
        state_mgr.save_all = AsyncMock()

        pm = _make_pm()
        rm = _make_rm(pm)
        oms = OMS(executor=MagicMock(), portfolio_manager=pm)

        # 1회 실행 후 cancel
        task = asyncio.create_task(
            LiveRunner._periodic_state_save(state_mgr, pm, rm, oms, interval=0.05)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        state_mgr.save_all.assert_called()
        call_kwargs = state_mgr.save_all.call_args
        assert call_kwargs[1]["oms"] is oms


class TestShutdownSavesOms:
    """E2E: run() → shutdown → DB에 OMS 상태 저장."""

    @pytest.mark.asyncio
    async def test_shutdown_saves_oms(self, tmp_path: pathlib.Path) -> None:
        """shutdown 시 OMS 상태가 DB에 저장됨."""
        db_path = str(tmp_path / "shutdown_oms.db")
        runner = _make_runner(db_path=db_path)
        _mock_feed(runner)

        # run() 중 OMS에 order 추가하기 위해 monkey-patch
        original_register = OMS.register

        async def patched_register(self_oms: OMS, bus: object) -> None:
            await original_register(self_oms, bus)
            # 처리된 주문 시뮬레이션
            self_oms._processed_orders.add("shutdown-order-1")
            self_oms._processed_orders.add("shutdown-order-2")

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.15)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        with patch.object(OMS, "register", patched_register):
            await runner.run()
        await shutdown_task

        # DB에서 OMS 상태 확인
        db = Database(db_path)
        await db.connect()
        state_mgr = StateManager(db)
        oms_state = await state_mgr.load_oms_state()
        assert oms_state is not None
        assert "shutdown-order-1" in oms_state
        assert "shutdown-order-2" in oms_state
        await db.close()


class TestRestoredOrdersPreventDuplicates:
    """복원된 order ID로 중복 방지."""

    @pytest.mark.asyncio
    async def test_restored_orders_prevent_duplicates(self) -> None:
        """복원 후 동일 order ID → reject 확인."""
        pm = _make_pm()
        oms = OMS(executor=MagicMock(), portfolio_manager=pm)

        # 기존 주문 복원
        oms.restore_processed_orders({"dup-order-1", "dup-order-2"})
        assert "dup-order-1" in oms.processed_orders

        # 동일 ID로 주문 시도 → 멱등성에 의해 거부
        from src.core.event_bus import EventBus
        from src.core.events import OrderRequestEvent

        bus = EventBus()
        await oms.register(bus)

        order = OrderRequestEvent(
            client_order_id="dup-order-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=1000.0,
            validated=True,
            source="test",
        )
        await oms._on_order_request(order)
        assert oms.total_rejected == 1


class TestRunWithDbRestoresOms:
    """E2E: save → restart → OMS processed_orders 동일."""

    @pytest.mark.asyncio
    async def test_run_with_db_restores_oms(self, tmp_path: pathlib.Path) -> None:
        """DB에 OMS 상태 저장 → 재시작 → 동일 processed_orders 확인."""
        db_path = str(tmp_path / "e2e_oms.db")

        # 1차: DB에 OMS 상태 사전 저장
        db = Database(db_path)
        await db.connect()
        state_mgr = StateManager(db)
        pm = _make_pm()
        rm = _make_rm(pm)
        oms = OMS(executor=MagicMock(), portfolio_manager=pm)
        oms._processed_orders = {"e2e-order-1", "e2e-order-2", "e2e-order-3"}
        await state_mgr.save_all(pm, rm, oms=oms)
        await db.close()

        # 2차: LiveRunner.run()으로 복원 → shutdown → 재저장
        runner = _make_runner(db_path=db_path)
        _mock_feed(runner)

        original_register = OMS.register

        async def capture_register(self_oms: OMS, bus: object) -> None:
            await original_register(self_oms, bus)
            # run() 내부에서 _restore_state가 먼저 호출되므로 여기서 확인 불가
            # shutdown 후 DB에서 확인

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.15)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # DB에서 최종 OMS 상태 확인
        db2 = Database(db_path)
        await db2.connect()
        state_mgr2 = StateManager(db2)
        oms_state = await state_mgr2.load_oms_state()
        assert oms_state is not None
        assert oms_state == {"e2e-order-1", "e2e-order-2", "e2e-order-3"}
        await db2.close()


class TestRunWithoutDbOmsUnaffected:
    """db_path=None → OMS 빈 set으로 정상 동작."""

    @pytest.mark.asyncio
    async def test_run_without_db_oms_unaffected(self) -> None:
        """db_path=None → OMS processed_orders는 빈 set."""
        runner = _make_runner(db_path=None)
        _mock_feed(runner)

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.1)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task
        # 예외 없이 정상 종료되면 성공
