"""Executor 테스트.

BacktestExecutor의 deferred execution, SL/TS 즉시 체결, 수수료 계산을 검증합니다.
LiveExecutor의 assert→if/raise 변환 검증을 포함합니다.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.events import BarEvent, OrderRequestEvent
from src.eda.executors import BacktestExecutor, LiveExecutor
from src.portfolio.cost_model import CostModel


def _make_bar(
    symbol: str = "BTC/USDT",
    open_price: float = 50000.0,
    close_price: float = 50500.0,
    ts: datetime | None = None,
    timeframe: str = "1D",
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=max(open_price, close_price) * 1.01,
        low=min(open_price, close_price) * 0.99,
        close=close_price,
        volume=1000.0,
        bar_timestamp=ts or datetime.now(UTC),
    )


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    notional_usd: float = 10000.0,
    price: float | None = None,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        target_weight=0.5,
        notional_usd=notional_usd,
        price=price,
        validated=True,
        correlation_id=uuid4(),
        source="test",
    )


class TestDeferredExecution:
    """Deferred execution (일반 주문 → 다음 bar open 체결)."""

    async def test_normal_order_deferred(self) -> None:
        """일반 주문(price=None)은 즉시 체결되지 않고 pending."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar(open_price=48000.0))

        fill = await executor.execute(_make_order())
        assert fill is None  # deferred
        assert executor.pending_count == 1

    async def test_deferred_fill_at_next_bar_open(self) -> None:
        """다음 bar 도착 시 open 가격으로 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        bar1 = _make_bar(open_price=48000.0, close_price=49000.0)
        executor.on_bar(bar1)

        await executor.execute(_make_order(notional_usd=9600.0))

        # 다음 bar 도착 → fill
        bar2 = _make_bar(open_price=50000.0, close_price=51000.0)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert len(fills) == 1
        assert fills[0].fill_price == 50000.0  # bar2 open
        assert abs(fills[0].fill_qty - 0.192) < 1e-10  # 9600 / 50000

    async def test_deferred_fill_timestamp(self) -> None:
        """Deferred fill의 timestamp는 체결 bar의 timestamp."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        ts1 = datetime(2024, 1, 1, tzinfo=UTC)
        ts2 = datetime(2024, 1, 2, tzinfo=UTC)

        executor.on_bar(_make_bar(ts=ts1))
        await executor.execute(_make_order())

        executor.on_bar(_make_bar(ts=ts2))
        executor.fill_pending(_make_bar(ts=ts2))

        fills = executor.drain_fills()
        assert fills[0].fill_timestamp == ts2  # bar2 timestamp

    async def test_pending_count_decreases_after_fill(self) -> None:
        """fill_pending 후 pending_count 감소."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        await executor.execute(_make_order())
        await executor.execute(_make_order())
        assert executor.pending_count == 2

        executor.on_bar(_make_bar())
        executor.fill_pending(_make_bar())

        assert executor.pending_count == 0
        fills = executor.drain_fills()
        assert len(fills) == 2

    async def test_drain_fills_clears_buffer(self) -> None:
        """drain_fills() 호출 후 비워짐."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())
        await executor.execute(_make_order())

        bar2 = _make_bar()
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills1 = executor.drain_fills()
        fills2 = executor.drain_fills()
        assert len(fills1) == 1
        assert len(fills2) == 0  # 이미 비워짐

    async def test_multi_symbol_deferred_fills(self) -> None:
        """멀티 심볼: 해당 심볼 bar 도착 시에만 fill."""
        executor = BacktestExecutor(cost_model=CostModel.zero())

        # BTC, ETH 각각 주문
        executor.on_bar(_make_bar(symbol="BTC/USDT", open_price=50000.0))
        executor.on_bar(_make_bar(symbol="ETH/USDT", open_price=3000.0))

        await executor.execute(_make_order(symbol="BTC/USDT", notional_usd=10000.0))
        await executor.execute(_make_order(symbol="ETH/USDT", notional_usd=3000.0))
        assert executor.pending_count == 2

        # BTC bar만 도착 → BTC만 fill
        btc_bar2 = _make_bar(symbol="BTC/USDT", open_price=51000.0)
        executor.on_bar(btc_bar2)
        executor.fill_pending(btc_bar2)

        assert executor.pending_count == 1  # ETH 아직 pending
        fills = executor.drain_fills()
        assert len(fills) == 1
        assert fills[0].symbol == "BTC/USDT"
        assert fills[0].fill_price == 51000.0

    async def test_deferred_preserves_correlation_id(self) -> None:
        """Deferred fill이 주문의 correlation_id를 보존."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        order = _make_order()
        await executor.execute(order)

        bar2 = _make_bar()
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert fills[0].correlation_id == order.correlation_id

    async def test_deferred_preserves_side(self) -> None:
        """SELL 주문은 SELL fill."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        await executor.execute(_make_order(side="SELL"))

        bar2 = _make_bar()
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert fills[0].side == "SELL"

    async def test_no_next_bar_stays_pending(self) -> None:
        """다음 bar 없이 종료 시 pending 유지."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        await executor.execute(_make_order())
        assert executor.pending_count == 1
        # drain_fills 호출해도 체결 안 됨
        assert len(executor.drain_fills()) == 0


class TestImmediateExecution:
    """SL/TS 즉시 체결 (order.price 설정)."""

    async def test_sl_order_fills_immediately(self) -> None:
        """SL 주문(price 설정)은 즉시 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        fill = await executor.execute(_make_order(price=47000.0, notional_usd=9400.0))
        assert fill is not None
        assert fill.fill_price == 47000.0
        assert executor.pending_count == 0

    async def test_ts_order_fills_immediately(self) -> None:
        """TS 주문(price 설정)은 즉시 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        bar_ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        executor.on_bar(_make_bar(ts=bar_ts))

        fill = await executor.execute(_make_order(price=52000.0, notional_usd=5200.0))
        assert fill is not None
        assert fill.fill_price == 52000.0
        assert fill.fill_timestamp == bar_ts

    async def test_mixed_immediate_and_deferred(self) -> None:
        """SL 즉시 + 일반 deferred 혼합."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        # SL → 즉시
        sl_fill = await executor.execute(_make_order(price=47000.0))
        assert sl_fill is not None

        # 일반 → deferred
        normal_fill = await executor.execute(_make_order())
        assert normal_fill is None
        assert executor.pending_count == 1


class TestFeeCalculation:
    """CostModel 수수료 적용 테스트."""

    async def test_fee_on_immediate_fill(self) -> None:
        """SL 즉시 체결 시 수수료 적용."""
        cost_model = CostModel.binance_futures()
        executor = BacktestExecutor(cost_model=cost_model)
        executor.on_bar(_make_bar())

        fill = await executor.execute(_make_order(price=50000.0, notional_usd=10000.0))
        assert fill is not None
        expected_fee = 10000.0 * cost_model.total_fee_rate
        assert abs(fill.fee - expected_fee) < 0.01

    async def test_fee_on_deferred_fill(self) -> None:
        """Deferred 체결 시 수수료 적용."""
        cost_model = CostModel.binance_futures()
        executor = BacktestExecutor(cost_model=cost_model)
        executor.on_bar(_make_bar())

        await executor.execute(_make_order(notional_usd=10000.0))

        bar2 = _make_bar(open_price=51000.0)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        expected_fee = 10000.0 * cost_model.total_fee_rate
        assert abs(fills[0].fee - expected_fee) < 0.01


class TestEdgeCases:
    """엣지 케이스 테스트."""

    async def test_fill_at_zero_price_fails(self) -> None:
        """가격 0인 bar로 fill 시 실패."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())
        await executor.execute(_make_order())

        bar2 = _make_bar(open_price=0.0)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert len(fills) == 0  # invalid price → skip
        assert executor.pending_count == 0  # order consumed even if fill fails

    async def test_fill_timestamp_updates_per_bar(self) -> None:
        """각 bar마다 fill_timestamp 업데이트."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        ts1 = datetime(2024, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(days=1)
        ts3 = ts2 + timedelta(days=1)

        # 첫 주문
        executor.on_bar(_make_bar(ts=ts1))
        await executor.execute(_make_order())

        # 두 번째 bar에서 fill
        bar2 = _make_bar(ts=ts2)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)
        fills1 = executor.drain_fills()
        assert fills1[0].fill_timestamp == ts2

        # 다시 주문 + 세 번째 bar에서 fill
        await executor.execute(_make_order())
        bar3 = _make_bar(ts=ts3)
        executor.on_bar(bar3)
        executor.fill_pending(bar3)
        fills2 = executor.drain_fills()
        assert fills2[0].fill_timestamp == ts3


class TestLiveExecutorSafetyChecks:
    """LiveExecutor: assert→if/raise 변환 검증."""

    def _make_live_executor(self) -> LiveExecutor:
        """Mock BinanceFuturesClient로 LiveExecutor 생성."""
        mock_client = MagicMock()
        mock_client.is_api_healthy = True
        mock_client.consecutive_failures = 0
        mock_client.to_futures_symbol = MagicMock(return_value="BTC/USDT:USDT")
        return LiveExecutor(futures_client=mock_client)

    async def test_execute_without_pm_returns_none(self) -> None:
        """PM 미설정 시 execute()는 None을 반환 (assert 대신)."""
        executor = self._make_live_executor()
        # PM 설정하지 않음
        order = _make_order()
        fill = await executor.execute(order)
        assert fill is None

    async def test_resolve_reduce_only_without_pm_raises(self) -> None:
        """PM 미설정 시 _resolve_reduce_only()는 RuntimeError 발생."""
        executor = self._make_live_executor()
        # PM 설정하지 않음
        order = _make_order()
        with pytest.raises(RuntimeError, match="PM set"):
            executor._resolve_reduce_only(order)

    async def test_execute_single_without_pm_raises(self) -> None:
        """PM 미설정 시 _execute_single()은 RuntimeError 발생."""
        executor = self._make_live_executor()
        # PM 설정하지 않음
        order = _make_order()
        with pytest.raises(RuntimeError, match="PM set"):
            await executor._execute_single(
                order=order,
                futures_symbol="BTC/USDT:USDT",
                reduce_only=False,
            )

    async def test_execute_with_pm_set_proceeds(self) -> None:
        """PM 설정 시 정상 실행 (API unhealthy 케이스에서 None 반환)."""
        executor = self._make_live_executor()
        executor._client.is_api_healthy = False
        mock_pm = MagicMock()
        executor.set_pm(mock_pm)
        order = _make_order()
        fill = await executor.execute(order)
        assert fill is None  # API unhealthy → None

    async def test_parse_fill_static_method(self) -> None:
        """_parse_fill은 정상 CCXT 응답에서 FillEvent 생성."""
        order = _make_order()
        result: dict[str, object] = {
            "average": 50000.0,
            "filled": 0.2,
            "status": "closed",
            "fee": {"cost": 5.0, "currency": "USDT"},
        }
        fill = LiveExecutor._parse_fill(order, result, requested_amount=0.2)
        assert fill is not None
        assert fill.fill_price == 50000.0
        assert fill.fill_qty == 0.2
        assert fill.fee == 5.0

    async def test_parse_fill_invalid_price_returns_none(self) -> None:
        """_parse_fill: price=0이면 None."""
        order = _make_order()
        result: dict[str, object] = {"average": 0, "filled": 0.2}
        fill = LiveExecutor._parse_fill(order, result)
        assert fill is None

    async def test_confirm_order_closed_status_returns_immediately(self) -> None:
        """_confirm_order: status=closed이면 재확인 없이 즉시 반환."""
        executor = self._make_live_executor()
        result: dict[str, object] = {"status": "closed", "id": "123"}
        confirmed = await executor._confirm_order(result, "BTC/USDT:USDT")
        assert confirmed["status"] == "closed"

    async def test_execute_with_pm_and_healthy_api(self) -> None:
        """PM 설정 + API healthy 시 _resolve_reduce_only 호출."""
        executor = self._make_live_executor()
        mock_pm = MagicMock()
        mock_pm.positions = {}
        executor.set_pm(mock_pm)

        # _execute_single을 mock하여 호출 여부 확인
        executor._execute_single = AsyncMock(return_value=None)  # type: ignore[method-assign]

        order = _make_order()
        await executor.execute(order)
        executor._execute_single.assert_called_once()


class TestLiveExecutorMetricsCallback:
    """LiveExecutor metrics 콜백 주입 테스트."""

    def _make_live_executor_with_metrics(self) -> tuple[LiveExecutor, MagicMock]:
        """Mock BinanceFuturesClient + Mock metrics로 LiveExecutor 생성."""
        mock_client = MagicMock()
        mock_client.is_api_healthy = True
        mock_client.consecutive_failures = 0
        mock_client.to_futures_symbol = MagicMock(return_value="BTC/USDT:USDT")
        mock_metrics = MagicMock()
        executor = LiveExecutor(futures_client=mock_client, metrics=mock_metrics)
        return executor, mock_metrics

    async def test_api_blocked_calls_metrics(self) -> None:
        """API unhealthy 시 on_api_blocked 호출."""
        executor, metrics = self._make_live_executor_with_metrics()
        executor._client.is_api_healthy = False
        mock_pm = MagicMock()
        executor.set_pm(mock_pm)

        order = _make_order()
        result = await executor.execute(order)

        assert result is None
        metrics.on_api_blocked.assert_called_once_with(order.symbol)

    async def test_fill_parse_failure_calls_metrics(self) -> None:
        """Fill 파싱 실패 시 on_fill_parse_failure 호출."""
        _, metrics = self._make_live_executor_with_metrics()

        order = _make_order()
        result = LiveExecutor._parse_fill(
            order, {"average": 0, "filled": 0}, metrics=metrics
        )

        assert result is None
        metrics.on_fill_parse_failure.assert_called_once_with(order.symbol)

    async def test_partial_fill_calls_metrics(self) -> None:
        """Partial fill 시 on_partial_fill 호출."""
        _, metrics = self._make_live_executor_with_metrics()

        order = _make_order()
        result = LiveExecutor._parse_fill(
            order,
            {"average": 50000.0, "filled": 0.05, "fee": {"cost": 1.0}},
            requested_amount=0.1,
            metrics=metrics,
        )

        assert result is not None  # fill은 성공
        metrics.on_partial_fill.assert_called_once_with(order.symbol)

    async def test_no_metrics_no_error(self) -> None:
        """metrics=None이어도 에러 없이 동작."""
        mock_client = MagicMock()
        mock_client.is_api_healthy = False
        mock_client.consecutive_failures = 3
        mock_client.to_futures_symbol = MagicMock(return_value="BTC/USDT:USDT")
        executor = LiveExecutor(futures_client=mock_client)
        mock_pm = MagicMock()
        executor.set_pm(mock_pm)

        order = _make_order()
        result = await executor.execute(order)
        assert result is None  # no error, just returns None
