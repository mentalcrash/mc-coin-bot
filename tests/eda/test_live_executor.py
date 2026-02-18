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
    client.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    client.fetch_order = AsyncMock(
        return_value={"id": "ord_001", "status": "closed", "filled": 0.001, "average": 50000.0}
    )
    client.validate_min_notional = MagicMock(return_value=True)
    client.get_min_notional = MagicMock(return_value=5.0)
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
# reduceOnly 매핑
# ---------------------------------------------------------------------------


class TestReduceOnlyMapping:
    """_resolve_reduce_only() 테스트."""

    def test_long_entry_no_position(self) -> None:
        """포지션 없을 때 LONG entry."""
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm())

        order = _make_order(target_weight=1.0, side="BUY")
        reduce_only, flip = executor._resolve_reduce_only(order)
        assert reduce_only is False
        assert flip is False

    def test_short_entry_no_position(self) -> None:
        """포지션 없을 때 SHORT entry."""
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm())

        order = _make_order(target_weight=-1.0, side="SELL")
        reduce_only, flip = executor._resolve_reduce_only(order)
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
        reduce_only, flip = executor._resolve_reduce_only(order)
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
        reduce_only, flip = executor._resolve_reduce_only(order)
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
        reduce_only, flip = executor._resolve_reduce_only(order)
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
        reduce_only, flip = executor._resolve_reduce_only(order)
        assert reduce_only is True
        assert flip is False

    def test_direction_flip_long_to_short(self) -> None:
        """LONG→SHORT 전환 시 close만."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=-1.0, side="SELL")
        reduce_only, flip = executor._resolve_reduce_only(order)
        assert reduce_only is True
        assert flip is True

    def test_direction_flip_short_to_long(self) -> None:
        """SHORT→LONG 전환 시 close만."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.SHORT, size=0.01, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=1.0, side="BUY")
        reduce_only, flip = executor._resolve_reduce_only(order)
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
        """가격 추정 불가 시 None (fetch_ticker 실패 + PM price=0)."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=0.0
            )
        }
        client = _make_mock_client()
        client.fetch_ticker = AsyncMock(return_value={"last": 0})
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

    def test_partial_fill_detected(self) -> None:
        """partial fill (filled < requested * 0.99) 시 fill은 반환하되 경고."""
        order = _make_order()
        result = {"average": 50000.0, "filled": 0.0005, "fee": None}
        # requested 0.001, filled 0.0005 → 50% partial
        fill = LiveExecutor._parse_fill(order, result, requested_amount=0.001)
        assert fill is not None
        assert fill.fill_qty == 0.0005


# ---------------------------------------------------------------------------
# Fresh price fetch
# ---------------------------------------------------------------------------


class TestFreshPriceFetch:
    """실시간 가격 조회 + fallback 테스트."""

    @pytest.mark.asyncio
    async def test_uses_fresh_price_for_entry(self) -> None:
        """Entry 주문 시 fetch_ticker로 실시간 가격 사용."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=45000.0
            )
        }
        client = _make_mock_client()
        client.fetch_ticker = AsyncMock(return_value={"last": 52000.0})

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=520.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        # fetch_ticker가 호출되었어야 함
        client.fetch_ticker.assert_called_once()
        assert fill is not None

    @pytest.mark.asyncio
    async def test_falls_back_to_pm_price_on_ticker_failure(self) -> None:
        """fetch_ticker 실패 시 PM last_price로 fallback."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=48000.0
            )
        }
        client = _make_mock_client()
        client.fetch_ticker = AsyncMock(side_effect=Exception("API error"))

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=480.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        # fallback으로 PM price 사용 → 주문 성공
        assert fill is not None

    @pytest.mark.asyncio
    async def test_reduce_only_skips_ticker(self) -> None:
        """청산 주문(reduce_only)은 fetch_ticker 미호출."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.001, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(target_weight=0, side="SELL", notional_usd=50.0)
        fill = await executor.execute(order)

        # reduce_only → fetch_ticker 미호출
        client.fetch_ticker.assert_not_called()
        assert fill is not None


# ---------------------------------------------------------------------------
# MIN_NOTIONAL 검증
# ---------------------------------------------------------------------------


class TestMinNotionalCheck:
    """MIN_NOTIONAL 사전 검증 테스트."""

    @pytest.mark.asyncio
    async def test_rejects_below_min_notional(self) -> None:
        """notional < MIN_NOTIONAL 시 거래소 호출 없이 None 반환."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        client.validate_min_notional = MagicMock(return_value=False)
        client.get_min_notional = MagicMock(return_value=10.0)

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=3.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        assert fill is None
        # create_order가 호출되면 안됨
        client.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_above_min_notional(self) -> None:
        """notional >= MIN_NOTIONAL 시 정상 진행."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        client.validate_min_notional = MagicMock(return_value=True)

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=100.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        assert fill is not None
        client.create_order.assert_called_once()


# ---------------------------------------------------------------------------
# Order confirmation
# ---------------------------------------------------------------------------


class TestOrderConfirmation:
    """주문 상태 확인 테스트."""

    @pytest.mark.asyncio
    async def test_confirms_non_closed_order(self) -> None:
        """status != 'closed' 시 fetch_order로 재확인."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        client.create_order = AsyncMock(
            return_value={
                "id": "ord_002",
                "status": "open",
                "filled": 0,
                "average": 0,
            }
        )
        client.fetch_order = AsyncMock(
            return_value={
                "id": "ord_002",
                "status": "closed",
                "filled": 0.001,
                "average": 50000.0,
                "fee": {"cost": 0.02, "currency": "USDT"},
            }
        )

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=50.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        # fetch_order가 확인 호출
        client.fetch_order.assert_called_once_with("ord_002", "BTC/USDT:USDT")
        assert fill is not None
        assert fill.fill_price == 50000.0

    @pytest.mark.asyncio
    async def test_skips_confirmation_when_closed(self) -> None:
        """status == 'closed' 시 fetch_order 미호출."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=50.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        client.fetch_order.assert_not_called()
        assert fill is not None

    @pytest.mark.asyncio
    async def test_confirmation_failure_uses_original(self) -> None:
        """fetch_order 실패 시 원본 결과 사용."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.NEUTRAL, size=0.0, last_price=50000.0
            )
        }
        client = _make_mock_client()
        client.create_order = AsyncMock(
            return_value={
                "id": "ord_003",
                "status": "open",
                "filled": 0.001,
                "average": 50000.0,
                "fee": {"cost": 0.02, "currency": "USDT"},
            }
        )
        client.fetch_order = AsyncMock(side_effect=Exception("API error"))

        executor = LiveExecutor(client)
        executor.set_pm(_make_mock_pm(positions))

        order = _make_order(notional_usd=50.0, target_weight=1.0, side="BUY")
        fill = await executor.execute(order)

        # 원본 결과로 fill 생성
        assert fill is not None
        assert fill.fill_price == 50000.0
