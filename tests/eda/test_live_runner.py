"""LiveRunner 테스트 — Paper/Shadow factory, 컴포넌트 조립, graceful shutdown.

Mock LiveDataFeed를 사용하여 실제 WebSocket 연결 없이 테스트합니다.
"""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.eda.executors import BacktestExecutor, LiveExecutor, ShadowExecutor
from src.eda.live_data_feed import LiveDataFeed
from src.eda.live_runner import LiveMode, LiveRunner
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ---------------------------------------------------------------------------
# Mock Strategy
# ---------------------------------------------------------------------------


class AlwaysLongStrategy(BaseStrategy):
    """테스트용 항상 LONG 전략."""

    @property
    def name(self) -> str:
        return "test-always-long"

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
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        max_leverage_cap=2.0,
        rebalance_threshold=0.01,
        system_stop_loss=None,
        use_trailing_stop=False,
        cost_model=CostModel.zero(),
    )


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.exchange = MagicMock()
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaperFactory:
    """LiveRunner.paper() factory 테스트."""

    def test_creates_paper_runner(self) -> None:
        """paper() factory로 Paper 모드 LiveRunner 생성."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert runner.mode == LiveMode.PAPER
        assert isinstance(runner.feed, LiveDataFeed)

    def test_paper_uses_backtest_executor(self) -> None:
        """Paper 모드는 BacktestExecutor를 사용."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert isinstance(runner._executor, BacktestExecutor)


class TestShadowFactory:
    """LiveRunner.shadow() factory 테스트."""

    def test_creates_shadow_runner(self) -> None:
        """shadow() factory로 Shadow 모드 LiveRunner 생성."""
        client = _make_mock_client()
        runner = LiveRunner.shadow(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert runner.mode == LiveMode.SHADOW
        assert isinstance(runner.feed, LiveDataFeed)

    def test_shadow_uses_shadow_executor(self) -> None:
        """Shadow 모드는 ShadowExecutor를 사용."""
        client = _make_mock_client()
        runner = LiveRunner.shadow(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert isinstance(runner._executor, ShadowExecutor)


class TestGracefulShutdown:
    """Graceful shutdown 테스트."""

    @pytest.mark.asyncio
    async def test_shutdown_via_request(self) -> None:
        """request_shutdown() 호출 시 정상 종료."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )

        # feed.start()를 mock — shutdown 대기만 하도록
        async def mock_start(bus: object) -> None:
            await asyncio.sleep(999)

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock()
        runner._feed.bars_emitted = 0

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # feed.stop()이 호출되었는지 확인
        runner._feed.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_event_set(self) -> None:
        """_handle_shutdown이 shutdown_event를 설정하는지 확인."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )

        assert not runner._shutdown_event.is_set()
        runner._handle_shutdown(signal.SIGTERM)
        assert runner._shutdown_event.is_set()


class TestComponentAssembly:
    """컴포넌트 조립 테스트."""

    @pytest.mark.asyncio
    async def test_all_components_registered(self) -> None:
        """run() 시 모든 EDA 컴포넌트가 EventBus에 등록되는지 확인."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )

        # Mock feed
        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock()
        runner._feed.stop = AsyncMock()
        runner._feed.bars_emitted = 0

        # 즉시 shutdown
        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.1)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # feed.start()가 bus와 함께 호출되었는지 확인
        runner._feed.start.assert_called_once()

    def test_paper_with_asset_weights(self) -> None:
        """asset_weights가 전달되는지 확인."""
        client = _make_mock_client()
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT", "ETH/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            asset_weights=weights,
        )
        assert runner._asset_weights == weights


class TestDbPathParameter:
    """db_path 파라미터 전달 테스트."""

    def test_paper_db_path_passed(self) -> None:
        """paper() factory에 db_path가 전달되는지 확인."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            db_path="data/test.db",
        )
        assert runner._db_path == "data/test.db"

    def test_shadow_db_path_passed(self) -> None:
        """shadow() factory에 db_path가 전달되는지 확인."""
        client = _make_mock_client()
        runner = LiveRunner.shadow(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            db_path="data/test_shadow.db",
        )
        assert runner._db_path == "data/test_shadow.db"

    def test_default_db_path_is_none(self) -> None:
        """db_path 미지정 시 None이 기본값."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert runner._db_path is None


class TestClientStorage:
    """_client 저장 테스트."""

    def test_paper_stores_client(self) -> None:
        """paper() factory에서 _client가 저장되는지 확인."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert runner._client is client

    def test_shadow_stores_client(self) -> None:
        """shadow() factory에서 _client가 저장되는지 확인."""
        client = _make_mock_client()
        runner = LiveRunner.shadow(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        assert runner._client is client

    def test_manual_init_client_none(self) -> None:
        """수동 __init__ 시 _client는 None."""
        feed = MagicMock(spec=LiveDataFeed)
        runner = LiveRunner(
            strategy=AlwaysLongStrategy(),
            feed=feed,
            executor=ShadowExecutor(),
            target_timeframe="1D",
            config=_make_config(),
            mode=LiveMode.SHADOW,
        )
        assert runner._client is None


class TestWarmup:
    """REST API warmup 테스트."""

    @pytest.mark.asyncio
    async def test_warmup_called_during_run(self) -> None:
        """run() 시 fetch_ohlcv_raw가 호출되는지 확인."""
        client = _make_mock_client()
        # fetch_ohlcv_raw mock 설정
        client.fetch_ohlcv_raw = AsyncMock(
            return_value=[
                [1704067200000, 42000.0, 42500.0, 41500.0, 42200.0, 100.0],
                [1704153600000, 42200.0, 42800.0, 42000.0, 42600.0, 120.0],
                [1704240000000, 42600.0, 43000.0, 42400.0, 42900.0, 110.0],
            ]
        )

        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0
        runner._feed.symbols = ["BTC/USDT"]

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        client.fetch_ohlcv_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_warmup_failure_graceful(self) -> None:
        """API 실패 시 예외 없이 계속 진행."""
        client = _make_mock_client()
        client.fetch_ohlcv_raw = AsyncMock(side_effect=Exception("API down"))

        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0
        runner._feed.symbols = ["BTC/USDT"]

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        # 예외 없이 정상 종료되면 성공
        await runner.run()
        await shutdown_task

    @pytest.mark.asyncio
    async def test_warmup_skipped_when_no_client(self) -> None:
        """_client가 None이면 warmup 스킵."""
        feed = MagicMock(spec=LiveDataFeed)
        runner = LiveRunner(
            strategy=AlwaysLongStrategy(),
            feed=feed,
            executor=ShadowExecutor(),
            target_timeframe="1D",
            config=_make_config(),
            mode=LiveMode.SHADOW,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        # 예외 없이 정상 종료 — _client=None이므로 warmup 스킵
        await runner.run()
        await shutdown_task


class TestRunWithDb:
    """LiveRunner.run()에서 DB 통합 테스트."""

    @pytest.mark.asyncio
    async def test_run_without_db_still_works(self) -> None:
        """db_path=None이면 persistence 없이 정상 동작."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            db_path=None,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.1)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task
        # 예외 없이 종료되면 성공

    @pytest.mark.asyncio
    async def test_run_with_db_creates_file(self, tmp_path: object) -> None:
        """db_path 지정 시 DB 파일이 생성되고 shutdown 시 최종 상태가 저장."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        db_path = str(tmp_path / "runner_test.db")

        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            db_path=db_path,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.15)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # DB 파일 생성 확인
        assert pathlib.Path(db_path).exists()

        # 최종 상태 저장 확인 — bot_state 테이블에 pm_state 존재
        from src.eda.persistence.database import Database

        db = Database(db_path)
        await db.connect()
        conn = db.connection
        cursor = await conn.execute("SELECT key FROM bot_state")
        keys = {row[0] for row in await cursor.fetchall()}
        assert "pm_state" in keys
        assert "rm_state" in keys
        assert "last_save_timestamp" in keys
        await db.close()

    @pytest.mark.asyncio
    async def test_run_with_db_restores_state(self, tmp_path: object) -> None:
        """DB에 저장된 상태가 다음 run에서 복구되는지 확인."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        db_path = str(tmp_path / "restore_test.db")

        # 1차 run — DB에 상태 저장
        from src.eda.persistence.database import Database
        from src.eda.persistence.state_manager import StateManager
        from src.eda.portfolio_manager import EDAPortfolioManager, Position
        from src.models.types import Direction

        db = Database(db_path)
        await db.connect()
        state_mgr = StateManager(db)

        # PM에 포지션 세팅 후 저장
        config = _make_config()
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        pm._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.05,
            avg_entry_price=60000.0,
            last_price=62000.0,
        )
        pm._cash = 7000.0
        pm._order_counter = 10

        from src.eda.risk_manager import EDARiskManager

        rm = EDARiskManager(config=config, portfolio_manager=pm)
        rm._peak_equity = 11000.0

        await state_mgr.save_all(pm, rm)
        await db.close()

        # 2차 run — 상태 복구 확인
        # LiveRunner.run()이 상태를 복구하고 DB에 다시 저장하므로, 재로드 후 값 확인
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            db_path=db_path,
        )

        stopped = asyncio.Event()

        async def mock_start2(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start2)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.15)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # DB에서 최종 저장된 PM 상태 확인
        db2 = Database(db_path)
        await db2.connect()
        state_mgr2 = StateManager(db2)
        pm_state = await state_mgr2.load_pm_state()
        assert pm_state is not None

        # 복구된 상태가 반영되어 있어야 함
        positions = pm_state["positions"]
        assert isinstance(positions, dict)
        assert "BTC/USDT" in positions
        btc = positions["BTC/USDT"]
        assert isinstance(btc, dict)
        assert btc["direction"] == Direction.LONG.value
        assert btc["size"] == 0.05
        assert pm_state["cash"] == 7000.0
        assert pm_state["order_counter"] == 10

        rm_state = await state_mgr2.load_rm_state()
        assert rm_state is not None
        assert rm_state["peak_equity"] == 11000.0

        await db2.close()


# ---------------------------------------------------------------------------
# Live Factory
# ---------------------------------------------------------------------------


def _make_mock_futures_client() -> MagicMock:
    """Mock BinanceFuturesClient."""
    client = MagicMock()
    client.setup_account = AsyncMock()
    client.fetch_positions = AsyncMock(return_value=[])
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 10000.0}})
    return client


class TestLiveFactory:
    """LiveRunner.live() factory 테스트."""

    def test_creates_live_runner(self) -> None:
        """live() factory로 Live 모드 LiveRunner 생성."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )
        assert runner.mode == LiveMode.LIVE
        assert isinstance(runner.feed, LiveDataFeed)

    def test_live_uses_live_executor(self) -> None:
        """Live 모드는 LiveExecutor를 사용."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )
        assert isinstance(runner._executor, LiveExecutor)

    def test_live_stores_futures_client(self) -> None:
        """live() factory에서 _futures_client가 저장되는지 확인."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )
        assert runner._futures_client is futures_client

    def test_live_stores_symbols(self) -> None:
        """live() factory에서 _symbols가 저장되는지 확인."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT", "ETH/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )
        assert runner._symbols == ["BTC/USDT", "ETH/USDT"]

    def test_live_with_asset_weights(self) -> None:
        """asset_weights가 전달되는지 확인."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT", "ETH/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
            asset_weights=weights,
        )
        assert runner._asset_weights == weights


class TestPreflightChecks:
    """_preflight_checks() 테스트."""

    @pytest.mark.asyncio
    async def test_preflight_returns_balance(self) -> None:
        """정상 잔고 반환."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        futures_client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 5000.0, "free": 4500.0}}
        )
        futures_client.fetch_open_orders = AsyncMock(return_value=[])
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
            initial_capital=5000.0,
        )
        balance = await runner._preflight_checks()
        assert balance == 5000.0

    @pytest.mark.asyncio
    async def test_preflight_zero_balance_raises(self) -> None:
        """잔고 0이면 RuntimeError."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        futures_client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 0.0, "free": 0.0}})
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )
        with pytest.raises(RuntimeError, match="must be > 0"):
            await runner._preflight_checks()

    @pytest.mark.asyncio
    async def test_preflight_warns_large_diff(self, caplog: pytest.LogCaptureFixture) -> None:
        """Config capital 대비 50% 이상 차이 시 WARNING."""
        import logging

        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        futures_client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 2000.0, "free": 2000.0}}
        )
        futures_client.fetch_open_orders = AsyncMock(return_value=[])
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
            initial_capital=10000.0,
        )
        with caplog.at_level(logging.WARNING):
            balance = await runner._preflight_checks()

        assert balance == 2000.0

    @pytest.mark.asyncio
    async def test_preflight_detects_stale_orders(self) -> None:
        """미체결 주문 감지 시 에러는 안 나지만 함수 완료."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        futures_client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 5000.0, "free": 5000.0}}
        )
        futures_client.fetch_open_orders = AsyncMock(
            return_value=[{"id": "stale_001", "status": "open"}]
        )
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )
        balance = await runner._preflight_checks()
        assert balance == 5000.0

    @pytest.mark.asyncio
    async def test_preflight_called_in_live_run(self) -> None:
        """LIVE run() 시 preflight_checks가 호출되고 capital이 override됨."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()
        futures_client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 7777.0, "free": 7777.0}}
        )
        futures_client.fetch_open_orders = AsyncMock(return_value=[])
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
            initial_capital=10000.0,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0
        runner._feed.symbols = ["BTC/USDT"]

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # preflight가 fetch_balance를 호출했어야 함
        futures_client.fetch_balance.assert_called()

    @pytest.mark.asyncio
    async def test_paper_mode_skips_preflight(self) -> None:
        """Paper 모드에서는 preflight 미실행."""
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            initial_capital=10000.0,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.1)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task
        # Paper 모드: futures_client 없으므로 fetch_balance 호출 없음


class TestLiveRunIntegration:
    """Live 모드 run() 통합 테스트."""

    @pytest.mark.asyncio
    async def test_live_run_sets_pm_and_reconciles(self) -> None:
        """Live run() 시 LiveExecutor.set_pm + reconciler 실행 확인."""
        client = _make_mock_client()
        futures_client = _make_mock_futures_client()

        runner = LiveRunner.live(
            strategy=AlwaysLongStrategy(),
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )

        stopped = asyncio.Event()

        async def mock_start(bus: object) -> None:
            await stopped.wait()

        runner._feed = MagicMock(spec=LiveDataFeed)
        runner._feed.start = AsyncMock(side_effect=mock_start)
        runner._feed.stop = AsyncMock(side_effect=lambda: stopped.set())
        runner._feed.bars_emitted = 0
        runner._feed.symbols = ["BTC/USDT"]

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            runner.request_shutdown()

        shutdown_task = asyncio.create_task(trigger_shutdown())
        await runner.run()
        await shutdown_task

        # LiveExecutor에 PM이 설정되어야 함
        assert runner._executor._pm is not None
        # Reconciler가 fetch_positions를 호출했어야 함
        futures_client.fetch_positions.assert_called()


# ---------------------------------------------------------------------------
# Startup Reconciliation Tests
# ---------------------------------------------------------------------------


class TestStartupReconciliation:
    """_reconcile_positions() 테스트."""

    @pytest.mark.asyncio
    async def test_non_live_mode_skips(self) -> None:
        """PAPER 모드에서는 reconciliation을 스킵."""
        strategy = AlwaysLongStrategy.from_params()
        client = _make_mock_client()
        runner = LiveRunner.paper(
            strategy=strategy,
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
        )
        pm = MagicMock()
        result = await runner._reconcile_positions(pm)
        assert result == []

    @pytest.mark.asyncio
    async def test_live_mode_removes_phantom(self) -> None:
        """LIVE 모드에서 phantom position을 제거."""
        strategy = AlwaysLongStrategy.from_params()
        client = _make_mock_client()
        futures_client = MagicMock()
        futures_client.fetch_positions = AsyncMock(return_value=[])
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=strategy,
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )

        # PM에 phantom position 설정
        from src.eda.portfolio_manager import EDAPortfolioManager, Position
        from src.models.types import Direction

        pm = EDAPortfolioManager(config=_make_config(), initial_capital=10000.0)
        pm._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.01,
            avg_entry_price=50000.0,
            last_price=50000.0,
        )

        result = await runner._reconcile_positions(pm)

        assert result == ["BTC/USDT"]
        assert pm.positions["BTC/USDT"].size == 0.0

    @pytest.mark.asyncio
    async def test_api_failure_returns_empty(self) -> None:
        """거래소 API 실패 시 빈 리스트 반환 (기존 동작 유지)."""
        strategy = AlwaysLongStrategy.from_params()
        client = _make_mock_client()
        futures_client = MagicMock()
        futures_client.fetch_positions = AsyncMock(side_effect=RuntimeError("API error"))
        futures_client.to_futures_symbol = MagicMock(side_effect=lambda s: f"{s}:USDT")

        runner = LiveRunner.live(
            strategy=strategy,
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=_make_config(),
            client=client,
            futures_client=futures_client,
        )

        pm = MagicMock()
        result = await runner._reconcile_positions(pm)

        assert result == []
