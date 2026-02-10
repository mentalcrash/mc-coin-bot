"""BinanceFuturesClient 단위 테스트.

CCXT exchange mock으로 실제 API 호출 없이 테스트합니다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    AuthenticationError,
    InsufficientFundsError,
    NetworkError,
    OrderExecutionError,
    RateLimitError,
)
from src.exchange.binance_futures_client import BinanceFuturesClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(has_creds: bool = True) -> MagicMock:
    """Mock IngestionSettings."""
    settings = MagicMock()
    settings.rate_limit_per_minute = 1200
    settings.request_timeout = 30
    api_key = MagicMock()
    api_key.get_secret_value.return_value = "test_key" if has_creds else ""
    secret = MagicMock()
    secret.get_secret_value.return_value = "test_secret" if has_creds else ""
    settings.binance_api_key = api_key
    settings.binance_secret = secret
    settings.has_api_credentials.return_value = has_creds
    return settings


def _make_mock_exchange() -> MagicMock:
    """Mock CCXT exchange."""
    exchange = MagicMock()
    exchange.load_markets = AsyncMock()
    exchange.close = AsyncMock()
    exchange.markets = {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}
    exchange.set_position_mode = AsyncMock()
    exchange.set_leverage = AsyncMock()
    exchange.set_margin_mode = AsyncMock()
    exchange.create_order = AsyncMock(
        return_value={
            "id": "123456",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000.0,
            "filled": 0.001,
            "average": 50000.0,
            "cost": 50.0,
            "fee": {"cost": 0.02, "currency": "USDT"},
            "status": "closed",
        }
    )
    exchange.fetch_positions = AsyncMock(
        return_value=[
            {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long", "notional": 500.0},
            {"symbol": "ETH/USDT:USDT", "contracts": 0, "side": "long", "notional": 0},
        ]
    )
    exchange.fetch_balance = AsyncMock(
        return_value={"USDT": {"total": 10000.0, "free": 9500.0, "used": 500.0}}
    )
    exchange.cancel_order = AsyncMock(return_value={"id": "123456", "status": "canceled"})
    exchange.amount_to_precision = MagicMock(side_effect=lambda s, a: str(a))
    exchange.price_to_precision = MagicMock(side_effect=lambda s, p: str(p))
    return exchange


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------


class TestToFuturesSymbol:
    """to_futures_symbol() 테스트."""

    def test_spot_to_futures(self) -> None:
        assert BinanceFuturesClient.to_futures_symbol("BTC/USDT") == "BTC/USDT:USDT"

    def test_already_futures(self) -> None:
        assert BinanceFuturesClient.to_futures_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_eth(self) -> None:
        assert BinanceFuturesClient.to_futures_symbol("ETH/USDT") == "ETH/USDT:USDT"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """__aenter__ / __aexit__ 테스트."""

    @pytest.mark.asyncio
    async def test_init_success(self) -> None:
        """정상 초기화."""
        settings = _make_settings()
        client = BinanceFuturesClient(settings)

        mock_exchange = _make_mock_exchange()
        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            await client._initialize()

        assert client._initialized is True
        mock_exchange.load_markets.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_no_credentials(self) -> None:
        """API 키 없으면 AuthenticationError."""
        settings = _make_settings(has_creds=False)
        client = BinanceFuturesClient(settings)

        with pytest.raises(AuthenticationError, match="credentials required"):
            await client._initialize()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """async with 패턴 동작 확인."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                assert client._initialized is True

        mock_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_exchange_property_before_init(self) -> None:
        """초기화 전 exchange 접근 시 RuntimeError."""
        settings = _make_settings()
        client = BinanceFuturesClient(settings)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = client.exchange


# ---------------------------------------------------------------------------
# Account setup
# ---------------------------------------------------------------------------


class TestSetupAccount:
    """setup_account() 테스트."""

    @pytest.mark.asyncio
    async def test_setup_account_success(self) -> None:
        """Hedge Mode + Leverage + Margin Mode 설정."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                await client.setup_account(["BTC/USDT"])

        mock_exchange.set_position_mode.assert_called_once()
        mock_exchange.set_leverage.assert_called_once_with(1, "BTC/USDT:USDT")
        mock_exchange.set_margin_mode.assert_called_once_with("cross", "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_setup_account_custom_leverage(self) -> None:
        """커스텀 leverage 파라미터 전달."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                await client.setup_account(["BTC/USDT"], leverage=3)

        mock_exchange.set_leverage.assert_called_once_with(3, "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_setup_account_already_set(self) -> None:
        """이미 설정된 경우 무시 (idempotent)."""
        import ccxt as ccxt_sync

        settings = _make_settings()
        mock_exchange = _make_mock_exchange()
        mock_exchange.set_position_mode = AsyncMock(
            side_effect=ccxt_sync.ExchangeError("No need to change position side")
        )
        mock_exchange.set_leverage = AsyncMock(
            side_effect=ccxt_sync.ExchangeError("No need to change leverage")
        )
        mock_exchange.set_margin_mode = AsyncMock(
            side_effect=ccxt_sync.ExchangeError("No need to change margin type")
        )

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                # 예외 없이 성공
                await client.setup_account(["BTC/USDT"])


# ---------------------------------------------------------------------------
# create_order
# ---------------------------------------------------------------------------


class TestCreateOrder:
    """create_order() 테스트."""

    @pytest.mark.asyncio
    async def test_market_order(self) -> None:
        """시장가 주문 성공."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                result = await client.create_order(
                    symbol="BTC/USDT:USDT",
                    side="buy",
                    amount=0.001,
                    position_side="LONG",
                    client_order_id="test_001",
                )

        assert result["id"] == "123456"
        mock_exchange.create_order.assert_called_once()
        call_args = mock_exchange.create_order.call_args
        assert call_args[0][0] == "BTC/USDT:USDT"  # symbol
        assert call_args[0][1] == "market"  # type
        assert call_args[0][2] == "buy"  # side

    @pytest.mark.asyncio
    async def test_limit_order(self) -> None:
        """지정가 주문 시 price_to_precision 사용."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                await client.create_order(
                    symbol="BTC/USDT:USDT",
                    side="sell",
                    amount=0.002,
                    position_side="SHORT",
                    price=50000.0,
                    reduce_only=True,
                )

        call_args = mock_exchange.create_order.call_args
        assert call_args[0][1] == "limit"  # type
        params = call_args[0][5]
        assert params["positionSide"] == "SHORT"
        assert params["reduceOnly"] is True

    @pytest.mark.asyncio
    async def test_insufficient_funds(self) -> None:
        """잔고 부족 시 InsufficientFundsError."""
        import ccxt as ccxt_sync

        settings = _make_settings()
        mock_exchange = _make_mock_exchange()
        mock_exchange.create_order = AsyncMock(
            side_effect=ccxt_sync.InsufficientFunds("Not enough USDT")
        )

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                with pytest.raises(InsufficientFundsError, match="Insufficient funds"):
                    await client.create_order(
                        symbol="BTC/USDT:USDT",
                        side="buy",
                        amount=100.0,
                        position_side="LONG",
                    )

    @pytest.mark.asyncio
    async def test_invalid_order(self) -> None:
        """잘못된 파라미터 시 OrderExecutionError."""
        import ccxt as ccxt_sync

        settings = _make_settings()
        mock_exchange = _make_mock_exchange()
        mock_exchange.create_order = AsyncMock(side_effect=ccxt_sync.InvalidOrder("MIN_NOTIONAL"))

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                with pytest.raises(OrderExecutionError, match="Invalid order"):
                    await client.create_order(
                        symbol="BTC/USDT:USDT",
                        side="buy",
                        amount=0.0001,
                        position_side="LONG",
                    )

    @pytest.mark.asyncio
    async def test_precision_applied(self) -> None:
        """amount_to_precision이 호출되는지 확인."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                await client.create_order(
                    symbol="BTC/USDT:USDT",
                    side="buy",
                    amount=0.0015,
                    position_side="LONG",
                )

        mock_exchange.amount_to_precision.assert_called_once_with("BTC/USDT:USDT", 0.0015)


# ---------------------------------------------------------------------------
# fetch_positions
# ---------------------------------------------------------------------------


class TestFetchPositions:
    """fetch_positions() 테스트."""

    @pytest.mark.asyncio
    async def test_filters_zero_contracts(self) -> None:
        """contracts=0인 포지션은 필터링."""
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                positions = await client.fetch_positions()

        # BTC만 (ETH contracts=0)
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC/USDT:USDT"


# ---------------------------------------------------------------------------
# fetch_balance
# ---------------------------------------------------------------------------


class TestFetchBalance:
    """fetch_balance() 테스트."""

    @pytest.mark.asyncio
    async def test_returns_balance(self) -> None:
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                balance = await client.fetch_balance()

        assert balance["USDT"]["total"] == 10000.0


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    """cancel_order() 테스트."""

    @pytest.mark.asyncio
    async def test_cancel_success(self) -> None:
        settings = _make_settings()
        mock_exchange = _make_mock_exchange()

        with patch("src.exchange.binance_futures_client.ccxt.binance", return_value=mock_exchange):
            async with BinanceFuturesClient(settings) as client:
                result = await client.cancel_order("123456", "BTC/USDT:USDT")

        assert result["status"] == "canceled"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetry:
    """_retry_with_backoff() 테스트."""

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self) -> None:
        """NetworkError 시 재시도 후 성공."""
        import ccxt as ccxt_sync

        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ccxt_sync.NetworkError("timeout")
            return "success"

        result = await BinanceFuturesClient._retry_with_backoff(
            flaky_func, max_retries=3, base_backoff=0.01
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_network_error(self) -> None:
        """재시도 초과 시 NetworkError."""
        import ccxt as ccxt_sync

        async def always_fail() -> None:
            raise ccxt_sync.NetworkError("always fails")

        with pytest.raises(NetworkError, match="after retries"):
            await BinanceFuturesClient._retry_with_backoff(
                always_fail, max_retries=1, base_backoff=0.01
            )

    @pytest.mark.asyncio
    async def test_retry_rate_limit_raises(self) -> None:
        """Rate limit 재시도 초과 시 RateLimitError."""
        import ccxt as ccxt_sync

        async def rate_limited() -> None:
            raise ccxt_sync.RateLimitExceeded("429")

        with pytest.raises(RateLimitError, match="Rate limit"):
            await BinanceFuturesClient._retry_with_backoff(
                rate_limited, max_retries=1, base_backoff=0.01
            )

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        """재시도 불가 에러는 즉시 전파."""
        import ccxt as ccxt_sync

        call_count = 0

        async def invalid() -> None:
            nonlocal call_count
            call_count += 1
            raise ccxt_sync.InsufficientFunds("no funds")

        with pytest.raises(ccxt_sync.InsufficientFunds):
            await BinanceFuturesClient._retry_with_backoff(
                invalid, max_retries=3, base_backoff=0.01
            )
        assert call_count == 1  # 재시도 없이 즉시 실패
