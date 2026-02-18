"""BinanceFuturesClient STOP_MARKET / cancel_all 단위 테스트.

거래소 안전망(Exchange Safety Net) 관련 메서드를 테스트합니다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import ccxt.pro as ccxt
import pytest

from src.core.exceptions import NetworkError, OrderExecutionError
from src.exchange.binance_futures_client import BinanceFuturesClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.rate_limit_per_minute = 1200
    settings.request_timeout = 30
    api_key = MagicMock()
    api_key.get_secret_value.return_value = "test_key"
    secret = MagicMock()
    secret.get_secret_value.return_value = "test_secret"
    settings.binance_api_key = api_key
    settings.binance_secret = secret
    settings.has_api_credentials.return_value = True
    return settings


def _make_mock_exchange() -> MagicMock:
    exchange = MagicMock()
    exchange.load_markets = AsyncMock()
    exchange.close = AsyncMock()
    exchange.markets = {"BTC/USDT:USDT": {}}
    exchange.create_order = AsyncMock(
        return_value={
            "id": "stop_123",
            "symbol": "BTC/USDT:USDT",
            "type": "STOP_MARKET",
            "side": "sell",
            "status": "open",
            "info": {"closePosition": "true"},
        }
    )
    exchange.cancel_all_orders = AsyncMock(return_value={"code": 200, "msg": "success"})
    exchange.amount_to_precision = MagicMock(side_effect=lambda s, a: str(a))
    exchange.price_to_precision = MagicMock(side_effect=lambda s, p: str(p))
    return exchange


async def _make_client() -> tuple[BinanceFuturesClient, MagicMock]:
    """초기화된 client + mock exchange 반환."""
    settings = _make_settings()
    client = BinanceFuturesClient(settings=settings)
    mock_exchange = _make_mock_exchange()
    client._exchange = mock_exchange  # type: ignore[assignment]
    client._initialized = True
    return client, mock_exchange


# ---------------------------------------------------------------------------
# create_stop_market_order
# ---------------------------------------------------------------------------


class TestCreateStopMarketOrder:
    """create_stop_market_order 테스트."""

    @pytest.mark.asyncio
    async def test_basic_long_stop(self) -> None:
        """LONG 포지션 안전망 STOP_MARKET 배치."""
        client, mock = await _make_client()

        result = await client.create_stop_market_order(
            symbol="BTC/USDT:USDT",
            side="sell",
            stop_price=44000.0,
            client_order_id="safety-stop-BTC",
        )

        assert result["id"] == "stop_123"
        assert result["type"] == "STOP_MARKET"

        # create_order 호출 인자 검증
        call_args = mock.create_order.call_args
        assert call_args[0][0] == "BTC/USDT:USDT"  # symbol
        assert call_args[0][1] == "STOP_MARKET"  # type
        assert call_args[0][2] == "sell"  # side

        params = call_args[0][5]
        assert "positionSide" not in params
        assert params["stopPrice"] == "44000.0"
        assert params["closePosition"] == "true"
        assert params["newClientOrderId"] == "safety-stop-BTC"

    @pytest.mark.asyncio
    async def test_short_stop(self) -> None:
        """SHORT 포지션 안전망 STOP_MARKET 배치."""
        client, mock = await _make_client()

        await client.create_stop_market_order(
            symbol="ETH/USDT:USDT",
            side="buy",
            stop_price=3500.0,
        )

        call_args = mock.create_order.call_args
        params = call_args[0][5]
        assert "positionSide" not in params
        assert params["stopPrice"] == "3500.0"

    @pytest.mark.asyncio
    async def test_invalid_order_error(self) -> None:
        """InvalidOrder → OrderExecutionError 변환."""
        client, mock = await _make_client()
        mock.create_order = AsyncMock(side_effect=ccxt.InvalidOrder("Bad stop price"))

        with pytest.raises(OrderExecutionError, match="Invalid STOP_MARKET"):
            await client.create_stop_market_order(
                symbol="BTC/USDT:USDT",
                side="sell",
                stop_price=0.0,
            )

    @pytest.mark.asyncio
    async def test_exchange_error(self) -> None:
        """ExchangeError → OrderExecutionError + failure counter."""
        client, mock = await _make_client()
        mock.create_order = AsyncMock(side_effect=ccxt.ExchangeError("Server error"))

        with pytest.raises(OrderExecutionError, match="STOP_MARKET order execution failed"):
            await client.create_stop_market_order(
                symbol="BTC/USDT:USDT",
                side="sell",
                stop_price=44000.0,
            )

        assert client.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NetworkError → 재시도 후 NetworkError raise."""
        monkeypatch.setattr("asyncio.sleep", AsyncMock())

        client, mock = await _make_client()
        mock.create_order = AsyncMock(side_effect=ccxt.NetworkError("timeout"))

        with pytest.raises(NetworkError):
            await client.create_stop_market_order(
                symbol="BTC/USDT:USDT",
                side="sell",
                stop_price=44000.0,
            )

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self) -> None:
        """성공 시 failure counter 리셋."""
        client, _mock = await _make_client()
        client._consecutive_failures = 3

        await client.create_stop_market_order(
            symbol="BTC/USDT:USDT",
            side="sell",
            stop_price=44000.0,
        )

        assert client.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_metrics_callback_called(self) -> None:
        """메트릭 콜백 호출 검증."""
        client, _mock = await _make_client()
        callback = MagicMock()
        client.set_metrics_callback(callback)

        await client.create_stop_market_order(
            symbol="BTC/USDT:USDT",
            side="sell",
            stop_price=44000.0,
        )

        callback.on_api_call.assert_called_once()
        call_args = callback.on_api_call.call_args
        assert call_args[0][0] == "create_stop_market"
        assert call_args[0][2] == "success"


# ---------------------------------------------------------------------------
# cancel_all_symbol_orders
# ---------------------------------------------------------------------------


class TestCancelAllSymbolOrders:
    """cancel_all_symbol_orders 테스트."""

    @pytest.mark.asyncio
    async def test_basic_cancel(self) -> None:
        """정상 취소."""
        client, mock = await _make_client()

        result = await client.cancel_all_symbol_orders("BTC/USDT:USDT")

        mock.cancel_all_orders.assert_called_once_with("BTC/USDT:USDT")
        assert result["code"] == 200

    @pytest.mark.asyncio
    async def test_cancel_error_propagation(self) -> None:
        """취소 실패 시 예외 전파."""
        client, mock = await _make_client()
        mock.cancel_all_orders = AsyncMock(side_effect=ccxt.ExchangeError("No orders"))

        with pytest.raises(Exception, match="No orders"):
            await client.cancel_all_symbol_orders("BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_metrics_on_cancel(self) -> None:
        """취소 메트릭 콜백 호출."""
        client, _mock = await _make_client()
        callback = MagicMock()
        client.set_metrics_callback(callback)

        await client.cancel_all_symbol_orders("BTC/USDT:USDT")

        callback.on_api_call.assert_called_once()
        assert callback.on_api_call.call_args[0][0] == "cancel_all_orders"
