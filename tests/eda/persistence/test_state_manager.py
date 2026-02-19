"""StateManager 테스트 — PM/RM 상태 save/load round-trip."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from src.eda.persistence.database import Database
from src.eda.persistence.state_manager import StateManager
from src.eda.portfolio_manager import EDAPortfolioManager, Position
from src.eda.risk_manager import EDARiskManager
from src.models.types import Direction
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel


def _make_config() -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        max_leverage_cap=2.0,
        rebalance_threshold=0.01,
        system_stop_loss=0.1,
        use_trailing_stop=False,
        cost_model=CostModel.zero(),
    )


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    """인메모리 DB fixture."""
    database = Database(":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def state_mgr(db: Database) -> StateManager:
    """StateManager fixture."""
    return StateManager(db)


@pytest.fixture
def pm() -> EDAPortfolioManager:
    """PM fixture with a position."""
    config = _make_config()
    pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
    # 수동으로 포지션 세팅
    pos = Position(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        size=0.1,
        avg_entry_price=50000.0,
        realized_pnl=100.0,
        unrealized_pnl=200.0,
        current_weight=0.5,
        last_price=52000.0,
        peak_price_since_entry=53000.0,
        trough_price_since_entry=0.0,
        atr_values=[1000.0, 1100.0, 900.0],
    )
    pm._positions["BTC/USDT"] = pos
    pm._cash = 5000.0
    pm._order_counter = 42
    pm._last_target_weights = {"BTC/USDT": 0.5}
    pm._last_executed_targets = {"BTC/USDT": 0.5}
    return pm


@pytest.fixture
def rm(pm: EDAPortfolioManager) -> EDARiskManager:
    """RM fixture."""
    config = _make_config()
    rm = EDARiskManager(config=config, portfolio_manager=pm)
    rm._peak_equity = 12000.0
    rm._circuit_breaker_triggered = False
    return rm


class TestPMStateRoundTrip:
    """PM 상태 저장/복구 round-trip 테스트."""

    @pytest.mark.asyncio
    async def test_save_and_load_pm_state(
        self, state_mgr: StateManager, pm: EDAPortfolioManager
    ) -> None:
        """PM 상태 save → load round-trip."""
        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None
        assert loaded["cash"] == 5000.0
        assert loaded["order_counter"] == 42

        positions = loaded["positions"]
        assert isinstance(positions, dict)
        assert "BTC/USDT" in positions
        btc_pos = positions["BTC/USDT"]
        assert isinstance(btc_pos, dict)
        assert btc_pos["direction"] == Direction.LONG.value
        assert btc_pos["size"] == 0.1
        assert btc_pos["avg_entry_price"] == 50000.0

    @pytest.mark.asyncio
    async def test_pm_restore_positions_and_atr(
        self, state_mgr: StateManager, pm: EDAPortfolioManager
    ) -> None:
        """PM restore_state: positions + atr_values 정확 복원."""
        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None

        config = _make_config()
        new_pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        new_pm.restore_state(loaded)

        pos = new_pm.positions.get("BTC/USDT")
        assert pos is not None
        assert pos.direction == Direction.LONG
        assert pos.size == 0.1
        assert pos.avg_entry_price == 50000.0
        assert pos.atr_values == [1000.0, 1100.0, 900.0]
        assert pos.peak_price_since_entry == 53000.0

    @pytest.mark.asyncio
    async def test_pm_restore_cash_and_counter(
        self, state_mgr: StateManager, pm: EDAPortfolioManager
    ) -> None:
        """PM restore_state: cash, order_counter 복원."""
        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None

        config = _make_config()
        new_pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        new_pm.restore_state(loaded)

        assert new_pm.available_cash == 5000.0
        assert new_pm.order_counter == 42
        assert new_pm.last_target_weights == {"BTC/USDT": 0.5}
        assert new_pm.last_executed_targets == {"BTC/USDT": 0.5}


class TestRMStateRoundTrip:
    """RM 상태 저장/복구 round-trip 테스트."""

    @pytest.mark.asyncio
    async def test_save_and_load_rm_state(
        self, state_mgr: StateManager, rm: EDARiskManager
    ) -> None:
        """RM 상태 save → load round-trip."""
        await state_mgr.save_rm_state(rm)
        loaded = await state_mgr.load_rm_state()
        assert loaded is not None
        assert loaded["peak_equity"] == 12000.0
        assert loaded["circuit_breaker_triggered"] is False

    @pytest.mark.asyncio
    async def test_rm_restore_state(
        self, state_mgr: StateManager, pm: EDAPortfolioManager, rm: EDARiskManager
    ) -> None:
        """RM restore_state: peak_equity, circuit_breaker 복원."""
        rm._circuit_breaker_triggered = True
        await state_mgr.save_rm_state(rm)
        loaded = await state_mgr.load_rm_state()
        assert loaded is not None

        config = _make_config()
        new_rm = EDARiskManager(config=config, portfolio_manager=pm)
        new_rm.restore_state(loaded)

        assert new_rm.peak_equity == 12000.0
        assert new_rm.is_circuit_breaker_active is True


class TestEmptyState:
    """빈 DB에서 로드 테스트."""

    @pytest.mark.asyncio
    async def test_load_pm_returns_none(self, state_mgr: StateManager) -> None:
        """빈 DB에서 PM 로드 → None."""
        result = await state_mgr.load_pm_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_rm_returns_none(self, state_mgr: StateManager) -> None:
        """빈 DB에서 RM 로드 → None."""
        result = await state_mgr.load_rm_state()
        assert result is None


class TestSaveAll:
    """save_all + timestamp 테스트."""

    @pytest.mark.asyncio
    async def test_save_all_stores_timestamp(
        self, state_mgr: StateManager, pm: EDAPortfolioManager, rm: EDARiskManager
    ) -> None:
        """save_all → PM + RM + timestamp 저장."""
        await state_mgr.save_all(pm, rm)

        pm_state = await state_mgr.load_pm_state()
        rm_state = await state_mgr.load_rm_state()
        ts = await state_mgr.get_last_save_timestamp()

        assert pm_state is not None
        assert rm_state is not None
        assert ts is not None


class TestClearState:
    """clear_state 테스트."""

    @pytest.mark.asyncio
    async def test_clear_removes_all(
        self, state_mgr: StateManager, pm: EDAPortfolioManager, rm: EDARiskManager
    ) -> None:
        """clear_state → 모든 항목 삭제."""
        await state_mgr.save_all(pm, rm)
        await state_mgr.clear_state()

        assert await state_mgr.load_pm_state() is None
        assert await state_mgr.load_rm_state() is None
        assert await state_mgr.get_last_save_timestamp() is None


class TestEmptyPositions:
    """포지션 없을 때 복원 테스트."""

    @pytest.mark.asyncio
    async def test_restore_with_no_positions(self, state_mgr: StateManager) -> None:
        """빈 positions 복원 정상 동작."""
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None

        new_pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        new_pm.restore_state(loaded)
        assert len(new_pm.positions) == 0


class TestShortPositionRoundTrip:
    """SHORT 포지션 직렬화/역직렬화 round-trip."""

    @pytest.mark.asyncio
    async def test_short_position_restore(self, state_mgr: StateManager) -> None:
        """SHORT 포지션이 정확히 복원되는지 확인."""
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        short_pos = Position(
            symbol="ETH/USDT",
            direction=Direction.SHORT,
            size=5.0,
            avg_entry_price=3000.0,
            realized_pnl=-50.0,
            unrealized_pnl=100.0,
            current_weight=-0.3,
            last_price=2900.0,
            peak_price_since_entry=0.0,
            trough_price_since_entry=2800.0,
            atr_values=[80.0, 90.0],
        )
        pm._positions["ETH/USDT"] = short_pos
        pm._cash = 7000.0

        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None

        new_pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        new_pm.restore_state(loaded)

        pos = new_pm.positions["ETH/USDT"]
        assert pos.direction == Direction.SHORT
        assert pos.size == 5.0
        assert pos.avg_entry_price == 3000.0
        assert pos.trough_price_since_entry == 2800.0
        assert pos.current_weight == -0.3


class TestMultiPositionRoundTrip:
    """멀티에셋 포지션 round-trip."""

    @pytest.mark.asyncio
    async def test_multiple_positions_restore(self, state_mgr: StateManager) -> None:
        """LONG + SHORT 동시 포지션이 모두 복원되는지 확인."""
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=20000.0)

        pm._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.1,
            avg_entry_price=50000.0,
            last_price=52000.0,
        )
        pm._positions["ETH/USDT"] = Position(
            symbol="ETH/USDT",
            direction=Direction.SHORT,
            size=3.0,
            avg_entry_price=3200.0,
            last_price=3100.0,
            trough_price_since_entry=3000.0,
        )
        pm._cash = 5000.0
        pm._last_target_weights = {"BTC/USDT": 0.4, "ETH/USDT": -0.3}
        pm._last_executed_targets = {"BTC/USDT": 0.4, "ETH/USDT": -0.3}

        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None

        new_pm = EDAPortfolioManager(config=config, initial_capital=20000.0)
        new_pm.restore_state(loaded)

        assert len(new_pm.positions) == 2
        assert new_pm.positions["BTC/USDT"].direction == Direction.LONG
        assert new_pm.positions["ETH/USDT"].direction == Direction.SHORT
        assert new_pm.available_cash == 5000.0
        assert new_pm.last_target_weights == {"BTC/USDT": 0.4, "ETH/USDT": -0.3}


class TestStateOverwrite:
    """save → 수정 → save → load 시 최신 상태만 로드."""

    @pytest.mark.asyncio
    async def test_overwrite_preserves_latest(
        self, state_mgr: StateManager, pm: EDAPortfolioManager, rm: EDARiskManager
    ) -> None:
        """두 번 save 후 load하면 최신 상태만 반환."""
        await state_mgr.save_all(pm, rm)

        # PM 상태 변경
        pm._cash = 1234.0
        pm._order_counter = 99
        rm._peak_equity = 15000.0

        await state_mgr.save_all(pm, rm)

        pm_loaded = await state_mgr.load_pm_state()
        rm_loaded = await state_mgr.load_rm_state()

        assert pm_loaded is not None
        assert pm_loaded["cash"] == 1234.0
        assert pm_loaded["order_counter"] == 99

        assert rm_loaded is not None
        assert rm_loaded["peak_equity"] == 15000.0


class TestRestoreTotalEquityConsistency:
    """복원 후 total_equity 정합성 확인."""

    @pytest.mark.asyncio
    async def test_total_equity_after_restore(self, state_mgr: StateManager) -> None:
        """PM restore 후 total_equity가 cash + positions와 일치."""
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)

        pm._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.1,
            avg_entry_price=50000.0,
            last_price=52000.0,
        )
        pm._cash = 5000.0
        original_equity = pm.total_equity  # 5000 + 0.1*52000 = 10200

        await state_mgr.save_pm_state(pm)
        loaded = await state_mgr.load_pm_state()
        assert loaded is not None

        new_pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        new_pm.restore_state(loaded)

        assert new_pm.total_equity == original_equity
        assert new_pm.total_equity == 10200.0
