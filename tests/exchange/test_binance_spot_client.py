"""BinanceSpotClient 단위 테스트.

Mock ccxt exchange로 Spot 주문 파라미터 검증.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    AuthenticationError,
    InsufficientFundsError,
    OrderExecutionError,
)


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Mock IngestionSettings."""
    settings = MagicMock()
    settings.rate_limit_per_minute = 1200
    settings.request_timeout = 30
    settings.has_api_credentials.return_value = True
    settings.binance_api_key.get_secret_value.return_value = "test_key"
    settings.binance_secret.get_secret_value.return_value = "test_secret"
    return settings


@pytest.fixture()
def mock_exchange() -> MagicMock:
    """Mock ccxt.pro.binance exchange instance."""
    exchange = MagicMock()
    exchange.load_markets = AsyncMock()
    exchange.close = AsyncMock()
    exchange.markets = {"BTC/USDT": {}}
    exchange.create_order = AsyncMock()
    exchange.fetch_balance = AsyncMock()
    exchange.fetch_ticker = AsyncMock()
    exchange.fetch_open_orders = AsyncMock()
    exchange.fetch_order = AsyncMock()
    exchange.cancel_order = AsyncMock()
    exchange.market = MagicMock(return_value={"limits": {"cost": {"min": 10.0}}})

    # precision methods
    exchange.cost_to_precision = MagicMock(side_effect=lambda sym, v: f"{float(v):.2f}")
    exchange.amount_to_precision = MagicMock(side_effect=lambda sym, v: f"{float(v):.6f}")
    exchange.price_to_precision = MagicMock(side_effect=lambda sym, v: f"{float(v):.2f}")

    return exchange


@pytest.fixture()
async def spot_client(mock_settings: MagicMock, mock_exchange: MagicMock) -> Any:
    """Initialize BinanceSpotClient with mocked exchange."""
    from src.exchange.binance_spot_client import BinanceSpotClient

    client = BinanceSpotClient(settings=mock_settings)
    with patch("src.exchange.binance_spot_client.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = mock_exchange
        await client._initialize()

    return client


# =========================================================================
# create_market_buy
# =========================================================================


@pytest.mark.asyncio()
async def test_create_market_buy_params(spot_client: Any, mock_exchange: MagicMock) -> None:
    """BUY: quoteOrderQty가 CCXT params로 전달되는지 검증."""
    mock_exchange.create_order.return_value = {
        "id": "12345",
        "status": "closed",
        "average": 50000.0,
        "filled": 0.002,
        "fee": {"cost": 0.1},
    }

    result = await spot_client.create_market_buy(
        symbol="BTC/USDT",
        quote_amount=100.0,
        client_order_id="spot_BTC_USDT_123",
    )

    mock_exchange.create_order.assert_awaited_once()
    call_args = mock_exchange.create_order.call_args
    assert call_args[0][0] == "BTC/USDT"  # symbol
    assert call_args[0][1] == "market"  # type
    assert call_args[0][2] == "buy"  # side
    assert call_args[0][3] is None  # amount (ignored for quoteOrderQty)

    params = call_args[0][5]
    assert "quoteOrderQty" in params
    assert params["newClientOrderId"] == "spot_BTC_USDT_123"
    assert result["id"] == "12345"


@pytest.mark.asyncio()
async def test_create_market_buy_precision(spot_client: Any, mock_exchange: MagicMock) -> None:
    """BUY: cost_to_precision이 호출되는지 검증."""
    mock_exchange.create_order.return_value = {
        "id": "1",
        "status": "closed",
        "average": 100,
        "filled": 1.0,
    }

    await spot_client.create_market_buy(symbol="BTC/USDT", quote_amount=99.999)

    mock_exchange.cost_to_precision.assert_called_once_with("BTC/USDT", 99.999)


# =========================================================================
# create_market_sell
# =========================================================================


@pytest.mark.asyncio()
async def test_create_market_sell_params(spot_client: Any, mock_exchange: MagicMock) -> None:
    """SELL: base amount가 정확히 전달되는지 검증."""
    mock_exchange.create_order.return_value = {
        "id": "67890",
        "status": "closed",
        "average": 50000.0,
        "filled": 0.001,
        "fee": {"cost": 0.05},
    }

    result = await spot_client.create_market_sell(
        symbol="BTC/USDT",
        base_amount=0.001,
        client_order_id="spot_BTC_sell_123",
    )

    call_args = mock_exchange.create_order.call_args
    assert call_args[0][2] == "sell"  # side
    # amount should be precision-formatted
    assert call_args[0][3] == "0.001000"
    assert result["id"] == "67890"


@pytest.mark.asyncio()
async def test_create_market_sell_amount_precision(
    spot_client: Any, mock_exchange: MagicMock
) -> None:
    """SELL: amount_to_precision이 호출되는지 검증."""
    mock_exchange.create_order.return_value = {
        "id": "1",
        "status": "closed",
        "average": 100,
        "filled": 1.0,
    }

    await spot_client.create_market_sell(symbol="ETH/USDT", base_amount=1.23456789)

    mock_exchange.amount_to_precision.assert_called_once_with("ETH/USDT", 1.23456789)


# =========================================================================
# create_stop_limit_sell
# =========================================================================


@pytest.mark.asyncio()
async def test_stop_limit_sell_params(spot_client: Any, mock_exchange: MagicMock) -> None:
    """STOP_LOSS_LIMIT: stopPrice와 limit price가 전달되는지 검증."""
    mock_exchange.create_order.return_value = {
        "id": "stop_1",
        "status": "open",
        "type": "STOP_LOSS_LIMIT",
    }

    result = await spot_client.create_stop_limit_sell(
        symbol="BTC/USDT",
        base_amount=0.01,
        stop_price=48000.0,
        limit_price=47760.0,
        client_order_id="spot-stop-BTC-USDT_abc",
    )

    call_args = mock_exchange.create_order.call_args
    assert call_args[0][1] == "STOP_LOSS_LIMIT"  # type
    assert call_args[0][2] == "sell"  # side
    assert call_args[0][3] == "0.010000"  # amount

    # limit price as positional arg
    limit_arg = call_args[0][4]
    assert limit_arg == "47760.00"

    # params
    params = call_args[0][5]
    assert params["stopPrice"] == "48000.00"
    assert params["newClientOrderId"] == "spot-stop-BTC-USDT_abc"
    assert result["id"] == "stop_1"


# =========================================================================
# Circuit Breaker
# =========================================================================


@pytest.mark.asyncio()
async def test_circuit_breaker_5_failures(spot_client: Any, mock_exchange: MagicMock) -> None:
    """5연속 실패 → is_api_healthy = False."""
    import ccxt as ccxt_sync

    assert spot_client.is_api_healthy is True

    mock_exchange.create_order.side_effect = ccxt_sync.ExchangeError("fail")

    for _ in range(5):
        with pytest.raises(OrderExecutionError):
            await spot_client.create_market_sell(symbol="BTC/USDT", base_amount=0.001)

    assert spot_client.is_api_healthy is False
    assert spot_client.consecutive_failures == 5


@pytest.mark.asyncio()
async def test_circuit_breaker_reset_on_success(spot_client: Any, mock_exchange: MagicMock) -> None:
    """성공 시 실패 카운터 리셋."""
    import ccxt as ccxt_sync

    mock_exchange.create_order.side_effect = ccxt_sync.ExchangeError("fail")
    for _ in range(3):
        with pytest.raises(OrderExecutionError):
            await spot_client.create_market_sell(symbol="BTC/USDT", base_amount=0.001)

    assert spot_client.consecutive_failures == 3

    # 성공
    mock_exchange.create_order.side_effect = None
    mock_exchange.create_order.return_value = {
        "id": "1",
        "status": "closed",
        "average": 100,
        "filled": 1.0,
    }
    await spot_client.create_market_sell(symbol="BTC/USDT", base_amount=0.001)

    assert spot_client.consecutive_failures == 0
    assert spot_client.is_api_healthy is True


# =========================================================================
# Retry with backoff
# =========================================================================


@pytest.mark.asyncio()
async def test_retry_with_backoff_success_after_retry(spot_client: Any) -> None:
    """재시도 후 성공."""
    import ccxt as ccxt_sync

    call_count = 0

    async def flaky_func() -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ccxt_sync.NetworkError("timeout")
        return {"ok": "true"}

    result = await spot_client._retry_with_backoff(flaky_func, max_retries=3, base_backoff=0.01)

    assert result == {"ok": "true"}
    assert call_count == 3


# =========================================================================
# Query Methods
# =========================================================================


@pytest.mark.asyncio()
async def test_fetch_balance(spot_client: Any, mock_exchange: MagicMock) -> None:
    """fetch_balance 정상 호출."""
    mock_exchange.fetch_balance.return_value = {
        "USDT": {"total": 1000.0, "free": 800.0},
        "BTC": {"total": 0.05, "free": 0.05},
    }

    balance = await spot_client.fetch_balance()
    assert balance["USDT"]["total"] == 1000.0
    assert balance["BTC"]["total"] == 0.05


@pytest.mark.asyncio()
async def test_fetch_ticker(spot_client: Any, mock_exchange: MagicMock) -> None:
    """fetch_ticker 정상 호출."""
    mock_exchange.fetch_ticker.return_value = {"last": 50000.0, "bid": 49999.0}

    ticker = await spot_client.fetch_ticker("BTC/USDT")
    assert ticker["last"] == 50000.0


# =========================================================================
# Utility Methods
# =========================================================================


def test_get_min_notional(spot_client: Any, mock_exchange: MagicMock) -> None:
    """MIN_NOTIONAL 조회."""
    assert spot_client.get_min_notional("BTC/USDT") == 10.0


def test_validate_min_notional(spot_client: Any) -> None:
    """MIN_NOTIONAL 검증."""
    assert spot_client.validate_min_notional("BTC/USDT", 15.0) is True
    assert spot_client.validate_min_notional("BTC/USDT", 5.0) is False


# =========================================================================
# InsufficientFunds
# =========================================================================


@pytest.mark.asyncio()
async def test_insufficient_funds_raises(spot_client: Any, mock_exchange: MagicMock) -> None:
    """InsufficientFunds → InsufficientFundsError."""
    import ccxt as ccxt_sync

    mock_exchange.create_order.side_effect = ccxt_sync.InsufficientFunds("not enough")

    with pytest.raises(InsufficientFundsError):
        await spot_client.create_market_buy(symbol="BTC/USDT", quote_amount=100.0)


# =========================================================================
# Authentication
# =========================================================================


@pytest.mark.asyncio()
async def test_no_credentials_raises() -> None:
    """API credentials 없으면 AuthenticationError."""
    settings = MagicMock()
    settings.has_api_credentials.return_value = False

    from src.exchange.binance_spot_client import BinanceSpotClient

    client = BinanceSpotClient(settings=settings)
    with pytest.raises(AuthenticationError):
        await client._initialize()
