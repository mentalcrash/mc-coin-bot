"""LiveExecutor 단위 테스트.

Mock BinanceFuturesClient + PM으로 테스트합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import FillEvent, OrderRequestEvent
from src.eda.executors import LiveExecutor
from src.models.types import Direction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakePosition:
    """테스트용 간이 Position."""

    symbol: str
    direction: Direction = Direction.NEUTRAL
    size: float = 0.0
    avg_entry_price: float = 0.0
    last_price: float = 0.0
    atr_values: list[float] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.size > 0.0

    @property
    def notional(self) -> float:
        return self.size * self.last_price


def _make_mock_client() -> MagicMock:
    """Mock BinanceFuturesClient."""
    client = MagicMock()
    client.create_order = AsyncMock(
        return_value={
            "id": "ord_001",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.001,
            "filled": 0.001,
            "average": 50000.0,
            "price": 50000.0,
            "cost": 50.0,
            "fee": {"cost": 0.02, "currency": "USDT"},
            "status": "closed",
        }
    )
    client.to_futures_symbol = MagicMock(
        side_effect=lambda s: f"{s}:USDT" if ":USDT" not in s else s
    )
    return client


def _make_mock_pm(positions: dict[str, FakePosition] | None = None) -> MagicMock:
    """Mock EDAPortfolioManager."""
    pm = MagicMock()
    pm.positions = positions or {}
    return pm


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    target_weight: float = 1.0,
    notional_usd: float = 50.0,
    price: float | None = None,
    client_order_id: str = "test_001",
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        target_weight=target_weight,
        notional_usd=notional_usd,
        price=price,
        validated=True,
    )


# ---------------------------------------------------------------------------
# ExecutorPort Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    """ExecutorPort Protocol 만족 확인."""

    def test_satisfies_executor_port(self) -> None:
        from src.eda.ports import ExecutorPort

        client = _make_mock_client()
        executor = LiveExecutor(client)
        assert isinstance(executor, ExecutorPort)


# ---------------------------------------------------------------------------
# positionSide 매핑
# ---------------------------------------------------------------------------


class TestPositionSideMapping:
    """_resolve_position_side() 테스트."""

    def test_long_entry_no_position(self) -> None:
        """포지션 없을 때 LONG entry."""
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm())

        order = _make_order(target_weight=1.0, side="BUY")
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "LONG"
        assert reduce_only is False
        assert flip is False

    def test_short_entry_no_position(self) -> None:
        """포지션 없을 때 SHORT entry."""
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm())

        order = _make_order(target_weight=-1.0, side="SELL")
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "SHORT"
        assert reduce_only is False
        assert flip is False

    def test_close_long_position(self) -> None:
        """LONG 포지션 청산 (target_weight=0)."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=0, side="SELL")
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "LONG"
        assert reduce_only is True
        assert flip is False

    def test_close_short_position(self) -> None:
        """SHORT 포지션 청산 (target_weight=0)."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.SHORT, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=0, side="BUY")
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "SHORT"
        assert reduce_only is True
        assert flip is False

    def test_sl_exit_long(self) -> None:
        """SL/TS exit (price 설정) — LONG 포지션."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=0, side="SELL", price=48000.0)
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "LONG"
        assert reduce_only is True
        assert flip is False

    def test_sl_exit_short(self) -> None:
        """SL/TS exit (price 설정) — SHORT 포지션."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.SHORT, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=0, side="BUY", price=52000.0)
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "SHORT"
        assert reduce_only is True
        assert flip is False

    def test_direction_flip_long_to_short(self) -> None:
        """LONG→SHORT 전환 시 close LONG만."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=-1.0, side="SELL")
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "LONG"
        assert reduce_only is True
        assert flip is True

    def test_direction_flip_short_to_long(self) -> None:
        """SHORT→LONG 전환 시 close SHORT만."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.SHORT, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=1.0, side="BUY")
        ps, reduce_only, flip = executor._resolve_position_side(order)
        assert ps == "SHORT"
        assert reduce_only is True
        assert flip is True


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestExecute:
    """execute() 통합 테스트."""

    @pytest.mark.asyncio
    async def test_long_entry_success(self) -> None:
        """LONG entry 성공 → FillEvent 반환."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=1.0, side="BUY", notional_usd=50.0)
        fill = await executor.execute(order)

        assert fill is not None
        assert isinstance(fill, FillEvent)
        assert fill.symbol == "BTC/USDT"
        assert fill.side == "BUY"
        assert fill.fill_price == 50000.0
        assert fill.fill_qty == 0.001
        assert fill.source == "LiveExecutor"

    @pytest.mark.asyncio
    async def test_close_uses_position_size(self) -> None:
        """Close 주문 시 포지션 size를 amount로 사용."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.005, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=0, side="SELL", notional_usd=250.0)
        await executor.execute(order)

        call_args = client.create_order.call_args
        assert call_args.kwargs["amount"] == 0.005
        assert call_args.kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_pm_not_set_returns_none(self) -> None:
        """PM이 설정되지 않으면 None."""
        client = _make_mock_client()
        executor = LiveExecutor(client)

        order = _make_order()
        fill = await executor.execute(order)
        assert fill is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self) -> None:
        """거래소 에러 시 None 반환."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        client.create_order = AsyncMock(side_effect=Exception("API error"))

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order()
        fill = await executor.execute(order)
        assert fill is None

    @pytest.mark.asyncio
    async def test_no_price_estimate_returns_none(self) -> None:
        """가격 추정 불가 시 None."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=0.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=1.0, side="BUY")
        fill = await executor.execute(order)
        assert fill is None


# ---------------------------------------------------------------------------
# _parse_fill
# ---------------------------------------------------------------------------


class TestParseFill:
    """_parse_fill() 테스트."""

    def test_parse_normal(self) -> None:
        """정상 CCXT 응답 파싱."""
        order = _make_order()
        result = {
            "average": 50000.0,
            "filled": 0.001,
            "fee": {"cost": 0.025, "currency": "USDT"},
        }
        fill = LiveExecutor._parse_fill(order, result)

        assert fill is not None
        assert fill.fill_price == 50000.0
        assert fill.fill_qty == 0.001
        assert fill.fee == 0.025
        assert fill.client_order_id == "test_001"
        assert fill.source == "LiveExecutor"

    def test_parse_zero_price(self) -> None:
        """가격 0이면 None."""
        order = _make_order()
        result = {"average": 0, "filled": 0.001, "fee": None}
        assert LiveExecutor._parse_fill(order, result) is None

    def test_parse_zero_filled(self) -> None:
        """체결 수량 0이면 None."""
        order = _make_order()
        result = {"average": 50000.0, "filled": 0, "fee": None}
        assert LiveExecutor._parse_fill(order, result) is None

    def test_parse_no_fee(self) -> None:
        """fee 없으면 0."""
        order = _make_order()
        result = {"average": 50000.0, "filled": 0.001, "fee": None}
        fill = LiveExecutor._parse_fill(order, result)
        assert fill is not None
        assert fill.fee == 0.0

    def test_parse_fallback_price_field(self) -> None:
        """average 없으면 price 필드 사용."""
        order = _make_order()
        result = {"average": None, "price": 49000.0, "filled": 0.002, "fee": None}
        fill = LiveExecutor._parse_fill(order, result)
        assert fill is not None
        assert fill.fill_price == 49000.0
