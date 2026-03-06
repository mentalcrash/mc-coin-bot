"""SpotExecutor 단위 테스트.

Mock BinanceSpotClient로 Spot 주문 실행 검증.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock
from uuid import uuid4

import pytest

from src.core.events import FillEvent, OrderRequestEvent


@pytest.fixture()
def mock_spot_client() -> MagicMock:
    """Mock BinanceSpotClient."""
    client = MagicMock()
    type(client).is_api_healthy = PropertyMock(return_value=True)
    type(client).consecutive_failures = PropertyMock(return_value=0)
    client.create_market_buy = AsyncMock()
    client.create_market_sell = AsyncMock()
    client.fetch_order = AsyncMock()
    client.get_min_notional = MagicMock(return_value=10.0)
    client.validate_min_notional = MagicMock(return_value=True)
    return client


@pytest.fixture()
def mock_pm() -> MagicMock:
    """Mock EDAPortfolioManager."""
    pm = MagicMock()
    pm.positions = {}
    return pm


@pytest.fixture()
def spot_executor(mock_spot_client: MagicMock, mock_pm: MagicMock) -> Any:
    """SpotExecutor with mocked client and PM."""
    from src.eda.spot_executor import SpotExecutor

    executor = SpotExecutor(mock_spot_client)
    executor.set_pm(mock_pm)
    return executor


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    notional: float = 100.0,
    target_weight: float = 0.5,
    price: float | None = None,
) -> OrderRequestEvent:
    """Helper: OrderRequestEvent 생성."""
    return OrderRequestEvent(
        symbol=symbol,
        side=side,
        notional_usd=notional,
        target_weight=target_weight,
        client_order_id=f"test_{uuid4().hex[:8]}",
        price=price,
    )


# =========================================================================
# BUY
# =========================================================================


@pytest.mark.asyncio()
async def test_buy_calls_create_market_buy(spot_executor: Any, mock_spot_client: MagicMock) -> None:
    """BUY: create_market_buy에 quoteOrderQty 전달."""
    mock_spot_client.create_market_buy.return_value = {
        "id": "buy_1",
        "status": "closed",
        "average": 50000.0,
        "filled": 0.002,
        "fee": {"cost": 0.1},
    }

    order = _make_order(side="BUY", notional=100.0)
    fill = await spot_executor.execute(order)

    assert fill is not None
    assert isinstance(fill, FillEvent)
    assert fill.side == "BUY"
    assert fill.fill_price == 50000.0
    assert fill.fill_qty == 0.002
    assert fill.source == "SpotExecutor"

    mock_spot_client.create_market_buy.assert_awaited_once()
    call_kwargs = mock_spot_client.create_market_buy.call_args[1]
    assert call_kwargs["quote_amount"] == 100.0


# =========================================================================
# SELL
# =========================================================================


@pytest.mark.asyncio()
async def test_sell_full_close(
    spot_executor: Any, mock_spot_client: MagicMock, mock_pm: MagicMock
) -> None:
    """SELL (target_weight=0): 전량 매도."""
    # Setup position
    pos = MagicMock()
    pos.is_open = True
    pos.size = 0.05
    pos.last_price = 50000.0
    mock_pm.positions = {"BTC/USDT": pos}

    mock_spot_client.create_market_sell.return_value = {
        "id": "sell_1",
        "status": "closed",
        "average": 50000.0,
        "filled": 0.05,
        "fee": {"cost": 2.5},
    }

    order = _make_order(side="SELL", target_weight=0.0, notional=2500.0)
    fill = await spot_executor.execute(order)

    assert fill is not None
    assert fill.fill_qty == 0.05

    call_kwargs = mock_spot_client.create_market_sell.call_args[1]
    assert call_kwargs["base_amount"] == 0.05  # 전량


@pytest.mark.asyncio()
async def test_sell_partial(
    spot_executor: Any, mock_spot_client: MagicMock, mock_pm: MagicMock
) -> None:
    """SELL (target_weight > 0): 부분 매도."""
    pos = MagicMock()
    pos.is_open = True
    pos.size = 1.0
    pos.last_price = 3000.0
    mock_pm.positions = {"ETH/USDT": pos}

    mock_spot_client.create_market_sell.return_value = {
        "id": "sell_2",
        "status": "closed",
        "average": 3000.0,
        "filled": 0.5,
        "fee": {"cost": 1.5},
    }

    order = _make_order(symbol="ETH/USDT", side="SELL", target_weight=0.3, notional=1500.0)
    fill = await spot_executor.execute(order)

    assert fill is not None
    # base_amount = 1500 / 3000 = 0.5
    call_kwargs = mock_spot_client.create_market_sell.call_args[1]
    assert call_kwargs["base_amount"] == pytest.approx(0.5, abs=1e-6)


# =========================================================================
# SHORT 차단
# =========================================================================


@pytest.mark.asyncio()
async def test_short_order_rejected(spot_executor: Any) -> None:
    """target_weight < 0: Long-Only 위반 → None."""
    order = _make_order(side="SELL", target_weight=-0.5)
    fill = await spot_executor.execute(order)

    assert fill is None


# =========================================================================
# API unhealthy
# =========================================================================


@pytest.mark.asyncio()
async def test_api_unhealthy_blocks_order(spot_executor: Any, mock_spot_client: MagicMock) -> None:
    """API unhealthy → None 반환."""
    type(mock_spot_client).is_api_healthy = PropertyMock(return_value=False)
    type(mock_spot_client).consecutive_failures = PropertyMock(return_value=5)

    order = _make_order()
    fill = await spot_executor.execute(order)

    assert fill is None
    mock_spot_client.create_market_buy.assert_not_awaited()


# =========================================================================
# PM not set
# =========================================================================


@pytest.mark.asyncio()
async def test_pm_not_set_returns_none(mock_spot_client: MagicMock) -> None:
    """PM 미설정 → None."""
    from src.eda.spot_executor import SpotExecutor

    executor = SpotExecutor(mock_spot_client)
    order = _make_order()
    fill = await executor.execute(order)

    assert fill is None


# =========================================================================
# Partial fill 감지
# =========================================================================


@pytest.mark.asyncio()
async def test_partial_fill_detected(
    spot_executor: Any, mock_spot_client: MagicMock, mock_pm: MagicMock
) -> None:
    """Partial fill 시 FillEvent에 실제 체결량 반영."""
    pos = MagicMock()
    pos.is_open = True
    pos.size = 1.0
    pos.last_price = 100.0
    mock_pm.positions = {"SOL/USDT": pos}

    mock_spot_client.create_market_sell.return_value = {
        "id": "partial_1",
        "status": "closed",
        "average": 100.0,
        "filled": 0.3,  # 요청 1.0 중 0.3만 체결
        "fee": {"cost": 0.03},
    }

    order = _make_order(symbol="SOL/USDT", side="SELL", target_weight=0.0, notional=100.0)
    fill = await spot_executor.execute(order)

    assert fill is not None
    assert fill.fill_qty == 0.3


# =========================================================================
# MIN_NOTIONAL 미달
# =========================================================================


@pytest.mark.asyncio()
async def test_min_notional_skip(spot_executor: Any, mock_spot_client: MagicMock) -> None:
    """MIN_NOTIONAL 미달 → skip."""
    mock_spot_client.validate_min_notional.return_value = False

    order = _make_order(side="BUY", notional=5.0)
    fill = await spot_executor.execute(order)

    assert fill is None
    mock_spot_client.create_market_buy.assert_not_awaited()


# =========================================================================
# No position to sell
# =========================================================================


@pytest.mark.asyncio()
async def test_sell_no_position_returns_none(spot_executor: Any, mock_pm: MagicMock) -> None:
    """포지션 없이 SELL → None."""
    mock_pm.positions = {}

    order = _make_order(side="SELL", target_weight=0.0)
    fill = await spot_executor.execute(order)

    assert fill is None
