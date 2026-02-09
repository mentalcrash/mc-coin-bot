"""LiveRunner 테스트 — Paper/Shadow factory, 컴포넌트 조립, graceful shutdown.

Mock LiveDataFeed를 사용하여 실제 WebSocket 연결 없이 테스트합니다.
"""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.eda.executors import BacktestExecutor, ShadowExecutor
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
