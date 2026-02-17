"""ExchangeStopManager 단위 테스트.

Stop price 계산, lifecycle, throttle, ratchet, error handling, state 검증.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.events import BarEvent, EventType, FillEvent
from src.eda.exchange_stop_manager import (
    ExchangeStopManager,
    StopOrderState,
)
from src.eda.portfolio_manager import Position
from src.models.types import Direction
from src.portfolio.config import PortfolioManagerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    system_stop_loss: float | None = 0.10,
    use_trailing_stop: bool = False,
    trailing_stop_atr_multiplier: float = 3.0,
    exchange_safety_margin: float = 0.02,
    exchange_trailing_safety_margin: float = 0.005,
    exchange_stop_update_threshold: float = 0.005,
) -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        system_stop_loss=system_stop_loss,
        use_trailing_stop=use_trailing_stop,
        trailing_stop_atr_multiplier=trailing_stop_atr_multiplier,
        use_exchange_safety_stop=True,
        exchange_safety_margin=exchange_safety_margin,
        exchange_trailing_safety_margin=exchange_trailing_safety_margin,
        exchange_stop_update_threshold=exchange_stop_update_threshold,
    )


def _make_position(
    symbol: str = "BTC/USDT",
    direction: Direction = Direction.LONG,
    size: float = 0.01,
    avg_entry_price: float = 50000.0,
    peak_price: float = 55000.0,
    trough_price: float = 48000.0,
    atr_values: list[float] | None = None,
) -> Position:
    return Position(
        symbol=symbol,
        direction=direction,
        size=size,
        avg_entry_price=avg_entry_price,
        last_price=avg_entry_price,
        peak_price_since_entry=peak_price,
        trough_price_since_entry=trough_price,
        atr_values=atr_values or [],
    )


def _make_futures_client() -> MagicMock:
    client = MagicMock()
    client.create_stop_market_order = AsyncMock(return_value={"id": "stop_order_123"})
    client.cancel_order = AsyncMock(return_value={"id": "stop_order_123", "status": "canceled"})
    client.cancel_all_symbol_orders = AsyncMock()
    client.to_futures_symbol = MagicMock(
        side_effect=lambda s: f"{s}:USDT" if ":USDT" not in s else s
    )
    return client


def _make_pm(positions: dict[str, Position] | None = None) -> MagicMock:
    pm = MagicMock()
    pm.positions = positions or {}
    return pm


def _make_fill(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    fill_price: float = 50000.0,
    fill_qty: float = 0.01,
) -> FillEvent:
    return FillEvent(
        client_order_id="test_order_1",
        symbol=symbol,
        side=side,
        fill_price=fill_price,
        fill_qty=fill_qty,
        fill_timestamp=datetime.now(UTC),
    )


def _make_bar(symbol: str = "BTC/USDT", close: float = 51000.0) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe="1m",
        open=close - 100,
        high=close + 100,
        low=close - 200,
        close=close,
        volume=1000.0,
        bar_timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Stop Price 계산
# ---------------------------------------------------------------------------


class TestStopPriceCalculation:
    """_calculate_stop_price 관련 테스트."""

    def test_long_sl_only(self) -> None:
        """LONG SL: entry * (1 - sl - margin) = 50000 * (1 - 0.10 - 0.02) = 44000."""
        config = _make_config()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(direction=Direction.LONG, avg_entry_price=50000.0)

        result = mgr._calculate_stop_price(pos)

        assert result is not None
        assert result == pytest.approx(44000.0)

    def test_short_sl_only(self) -> None:
        """SHORT SL: entry * (1 + sl + margin) = 50000 * (1 + 0.10 + 0.02) = 56000."""
        config = _make_config()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(direction=Direction.SHORT, avg_entry_price=50000.0)

        result = mgr._calculate_stop_price(pos)

        assert result is not None
        assert result == pytest.approx(56000.0)

    def test_long_ts_only(self) -> None:
        """LONG TS: (peak - atr * mult) * (1 - ts_margin)."""
        atr_values = [500.0] * 14  # ATR = 500
        config = _make_config(
            system_stop_loss=None,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=3.0,
            exchange_trailing_safety_margin=0.005,
        )
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(
            direction=Direction.LONG,
            peak_price=55000.0,
            atr_values=atr_values,
        )

        result = mgr._calculate_stop_price(pos)

        # (55000 - 500*3) * (1 - 0.005) = 53500 * 0.995 = 53232.5
        assert result is not None
        assert result == pytest.approx(53232.5)

    def test_short_ts_only(self) -> None:
        """SHORT TS: (trough + atr * mult) * (1 + ts_margin)."""
        atr_values = [500.0] * 14
        config = _make_config(
            system_stop_loss=None,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=3.0,
            exchange_trailing_safety_margin=0.005,
        )
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(
            direction=Direction.SHORT,
            trough_price=48000.0,
            atr_values=atr_values,
        )

        result = mgr._calculate_stop_price(pos)

        # (48000 + 500*3) * (1 + 0.005) = 49500 * 1.005 = 49747.5
        assert result is not None
        assert result == pytest.approx(49747.5)

    def test_long_combined_takes_min(self) -> None:
        """LONG SL + TS: min(sl_stop, ts_stop) — 더 넓은 쪽."""
        atr_values = [500.0] * 14
        config = _make_config(
            system_stop_loss=0.10,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=3.0,
        )
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(
            direction=Direction.LONG,
            avg_entry_price=50000.0,
            peak_price=55000.0,
            atr_values=atr_values,
        )

        sl_stop = mgr._calculate_sl_stop(pos)
        ts_stop = mgr._calculate_ts_stop(pos)
        result = mgr._calculate_stop_price(pos)

        assert sl_stop is not None and ts_stop is not None
        assert result == min(sl_stop, ts_stop)

    def test_short_combined_takes_max(self) -> None:
        """SHORT SL + TS: max(sl_stop, ts_stop) — 더 넓은 쪽."""
        atr_values = [500.0] * 14
        config = _make_config(
            system_stop_loss=0.10,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=3.0,
        )
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(
            direction=Direction.SHORT,
            avg_entry_price=50000.0,
            trough_price=48000.0,
            atr_values=atr_values,
        )

        sl_stop = mgr._calculate_sl_stop(pos)
        ts_stop = mgr._calculate_ts_stop(pos)
        result = mgr._calculate_stop_price(pos)

        assert sl_stop is not None and ts_stop is not None
        assert result == max(sl_stop, ts_stop)

    def test_no_sl_no_ts_returns_none(self) -> None:
        """SL=None, TS=False → None."""
        config = _make_config(system_stop_loss=None, use_trailing_stop=False)
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position()

        assert mgr._calculate_stop_price(pos) is None

    def test_atr_immature_ts_returns_none(self) -> None:
        """ATR 14봉 미달 → TS stop = None."""
        config = _make_config(system_stop_loss=None, use_trailing_stop=True)
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(atr_values=[500.0] * 10)  # < 14

        assert mgr._calculate_ts_stop(pos) is None

    def test_neutral_direction_returns_none(self) -> None:
        """NEUTRAL 방향 → None."""
        config = _make_config()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, _make_futures_client(), pm)
        pos = _make_position(direction=Direction.NEUTRAL)

        assert mgr._calculate_sl_stop(pos) is None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Entry fill → stop placed, exit fill → stop cancelled."""

    @pytest.mark.asyncio
    async def test_entry_fill_places_stop(self) -> None:
        """진입 fill → 거래소 stop 배치."""
        config = _make_config()
        client = _make_futures_client()
        pos = _make_position()
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        fill = _make_fill(symbol="BTC/USDT")
        await mgr._on_fill(fill)

        client.create_stop_market_order.assert_called_once()
        assert "BTC/USDT" in mgr.active_stops

    @pytest.mark.asyncio
    async def test_exit_fill_cancels_stop(self) -> None:
        """청산 fill → 거래소 stop 취소."""
        config = _make_config()
        client = _make_futures_client()
        # 포지션이 이미 청산된 상태
        pos = _make_position(size=0.0)
        pos_dict: dict[str, Position] = {"BTC/USDT": pos}
        pm = _make_pm(pos_dict)
        mgr = ExchangeStopManager(config, client, pm)

        # 먼저 stop을 수동으로 추가
        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="stop_123",
            client_order_id="safety-stop-test",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        fill = _make_fill(symbol="BTC/USDT")
        await mgr._on_fill(fill)

        client.cancel_order.assert_called_once()
        assert "BTC/USDT" not in mgr.active_stops

    @pytest.mark.asyncio
    async def test_no_duplicate_stop_on_additional_fill(self) -> None:
        """이미 stop이 있으면 중복 배치 안 함."""
        config = _make_config()
        client = _make_futures_client()
        pos = _make_position()
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        # 이미 stop 존재
        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="stop_123",
            client_order_id="safety-stop-test",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        fill = _make_fill(symbol="BTC/USDT")
        await mgr._on_fill(fill)

        client.create_stop_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_all_stops(self) -> None:
        """cancel_all_stops 동작 확인."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )
        mgr._stops["ETH/USDT"] = StopOrderState(
            symbol="ETH/USDT",
            exchange_order_id="s2",
            client_order_id="c2",
            stop_price=3000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr.cancel_all_stops()

        assert len(mgr.active_stops) == 0
        assert client.cancel_order.call_count == 2


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


class TestThrottle:
    """0.5% 미만 변동 → skip, 이상 → update."""

    @pytest.mark.asyncio
    async def test_small_change_skipped(self) -> None:
        """0.3% 변동 → 업데이트 skip."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        # 0.3% 변동: 44000 → 44132 (0.3%)
        await mgr._update_stop_if_needed("BTC/USDT", 44132.0)

        client.cancel_order.assert_not_called()
        client.create_stop_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_change_updates(self) -> None:
        """0.6% 변동 → 업데이트 실행."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        # 0.6% 변동: 44000 → 44264 (LONG → up, ratchet OK)
        await mgr._update_stop_if_needed("BTC/USDT", 44264.0)

        client.cancel_order.assert_called_once()
        client.create_stop_market_order.assert_called_once()
        assert mgr._stops["BTC/USDT"].stop_price == pytest.approx(44264.0)


# ---------------------------------------------------------------------------
# Ratchet
# ---------------------------------------------------------------------------


class TestRatchet:
    """LONG stop은 올림만, SHORT stop은 내림만."""

    @pytest.mark.asyncio
    async def test_long_ratchet_up_allowed(self) -> None:
        """LONG: stop price 올리기 → 허용."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr._update_stop_if_needed("BTC/USDT", 45000.0)  # +2.3%

        client.create_stop_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_ratchet_down_blocked(self) -> None:
        """LONG: stop price 내리기 → 차단."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr._update_stop_if_needed("BTC/USDT", 43000.0)  # -2.3%

        client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_ratchet_down_allowed(self) -> None:
        """SHORT: stop price 내리기 → 허용."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["ETH/USDT"] = StopOrderState(
            symbol="ETH/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=56000.0,
            position_side="SHORT",
            close_side="buy",
        )

        await mgr._update_stop_if_needed("ETH/USDT", 55000.0)  # -1.8%

        client.create_stop_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_ratchet_up_blocked(self) -> None:
        """SHORT: stop price 올리기 → 차단."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["ETH/USDT"] = StopOrderState(
            symbol="ETH/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=56000.0,
            position_side="SHORT",
            close_side="buy",
        )

        await mgr._update_stop_if_needed("ETH/USDT", 57000.0)  # +1.8%

        client.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Placement 실패 → retry, 5회 연속 → CRITICAL log."""

    @pytest.mark.asyncio
    async def test_placement_failure_increments_counter(self) -> None:
        """배치 실패 시 failure counter 증가."""
        config = _make_config()
        client = _make_futures_client()
        client.create_stop_market_order = AsyncMock(side_effect=Exception("API error"))
        pos = _make_position()
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        # First placement attempt fails
        await mgr._place_safety_stop("BTC/USDT", pos)

        # stop이 등록되지 않아야 함
        assert "BTC/USDT" not in mgr.active_stops

    @pytest.mark.asyncio
    async def test_update_failure_increments_counter(self) -> None:
        """업데이트 실패 시 failure counter 증가."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        client.cancel_order = AsyncMock(return_value={})
        client.create_stop_market_order = AsyncMock(side_effect=Exception("API error"))
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr._update_stop_if_needed("BTC/USDT", 45000.0)

        assert mgr._stops["BTC/USDT"].placement_failures == 1

    @pytest.mark.asyncio
    async def test_cancel_failure_silent(self) -> None:
        """취소 실패 시 예외 없이 진행."""
        config = _make_config()
        client = _make_futures_client()
        client.cancel_order = AsyncMock(side_effect=Exception("Already cancelled"))
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        # 예외 없이 진행
        await mgr._cancel_safety_stop("BTC/USDT")

        assert "BTC/USDT" not in mgr.active_stops

    @pytest.mark.asyncio
    async def test_bar_with_closed_position_cancels_stop(self) -> None:
        """Bar 시 포지션이 없으면 stop 취소."""
        config = _make_config()
        client = _make_futures_client()
        # 포지션 없음
        pm = _make_pm({})
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="s1",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        bar = _make_bar(symbol="BTC/USDT")
        await mgr._on_bar(bar)

        assert "BTC/USDT" not in mgr.active_stops


# ---------------------------------------------------------------------------
# State save/restore
# ---------------------------------------------------------------------------


class TestState:
    """State save/restore round-trip."""

    def test_save_restore_roundtrip(self) -> None:
        """get_state → restore_state round trip."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="stop_abc",
            client_order_id="safety-stop-btc-123",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
            placement_failures=2,
        )
        mgr._stops["ETH/USDT"] = StopOrderState(
            symbol="ETH/USDT",
            exchange_order_id="stop_def",
            client_order_id="safety-stop-eth-456",
            stop_price=3000.0,
            position_side="SHORT",
            close_side="buy",
        )

        state = mgr.get_state()

        # 새 인스턴스에서 복원
        mgr2 = ExchangeStopManager(config, client, pm)
        mgr2.restore_state(state)

        assert len(mgr2.active_stops) == 2
        assert mgr2.active_stops["BTC/USDT"].stop_price == 44000.0
        assert mgr2.active_stops["BTC/USDT"].position_side == "LONG"
        assert mgr2.active_stops["BTC/USDT"].placement_failures == 2
        assert mgr2.active_stops["ETH/USDT"].position_side == "SHORT"

    def test_empty_state_roundtrip(self) -> None:
        """빈 상태 round trip."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        state = mgr.get_state()
        mgr.restore_state(state)

        assert len(mgr.active_stops) == 0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestUtility:
    """유틸리티 메서드 테스트."""

    def test_is_safety_stop_order(self) -> None:
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        assert mgr.is_safety_stop_order("safety-stop-BTC-abc123")
        assert not mgr.is_safety_stop_order("normal-order-123")

    @pytest.mark.asyncio
    async def test_register(self) -> None:
        """EventBus 등록."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        bus = MagicMock()
        await mgr.register(bus)

        assert bus.subscribe.call_count == 2
        call_types = [call[0][0] for call in bus.subscribe.call_args_list]
        assert EventType.FILL in call_types
        assert EventType.BAR in call_types

    @pytest.mark.asyncio
    async def test_system_stop_loss_none_skips_placement(self) -> None:
        """system_stop_loss=None → 배치 안 함."""
        config = _make_config(system_stop_loss=None, use_trailing_stop=False)
        client = _make_futures_client()
        pos = _make_position()
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        await mgr._place_safety_stop("BTC/USDT", pos)

        client.create_stop_market_order.assert_not_called()


# ---------------------------------------------------------------------------
# Direction Flip (HIGH-1)
# ---------------------------------------------------------------------------


class TestDirectionFlip:
    """Direction flip(LONG→SHORT) 시 기존 stop 교체."""

    @pytest.mark.asyncio
    async def test_long_to_short_flip_replaces_stop(self) -> None:
        """LONG stop → SHORT 포지션 → cancel old + place new."""
        config = _make_config()
        client = _make_futures_client()
        # 현재 포지션: SHORT
        pos = _make_position(direction=Direction.SHORT)
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        # 기존 LONG stop
        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="old_stop",
            client_order_id="safety-stop-old",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        fill = _make_fill(symbol="BTC/USDT", side="SELL")
        await mgr._on_fill(fill)

        # old stop cancelled + new stop placed
        client.cancel_order.assert_called_once_with("old_stop", "BTC/USDT:USDT")
        client.create_stop_market_order.assert_called_once()
        state = mgr.active_stops.get("BTC/USDT")
        assert state is not None
        assert state.position_side == "SHORT"

    @pytest.mark.asyncio
    async def test_short_to_long_flip_replaces_stop(self) -> None:
        """SHORT stop → LONG 포지션 → cancel old + place new."""
        config = _make_config()
        client = _make_futures_client()
        pos = _make_position(direction=Direction.LONG)
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="old_stop",
            client_order_id="safety-stop-old",
            stop_price=56000.0,
            position_side="SHORT",
            close_side="buy",
        )

        fill = _make_fill(symbol="BTC/USDT", side="BUY")
        await mgr._on_fill(fill)

        client.cancel_order.assert_called_once()
        client.create_stop_market_order.assert_called_once()
        state = mgr.active_stops.get("BTC/USDT")
        assert state is not None
        assert state.position_side == "LONG"

    @pytest.mark.asyncio
    async def test_same_direction_no_replace(self) -> None:
        """동일 방향 → 기존 stop 유지, 추가 배치 없음."""
        config = _make_config()
        client = _make_futures_client()
        pos = _make_position(direction=Direction.LONG)
        pm = _make_pm({"BTC/USDT": pos})
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="existing_stop",
            client_order_id="safety-stop-existing",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        fill = _make_fill(symbol="BTC/USDT", side="BUY")
        await mgr._on_fill(fill)

        client.cancel_order.assert_not_called()
        client.create_stop_market_order.assert_not_called()
        assert mgr.active_stops["BTC/USDT"].exchange_order_id == "existing_stop"


# ---------------------------------------------------------------------------
# Atomicity Gap (HIGH-2)
# ---------------------------------------------------------------------------


class TestAtomicityGap:
    """Cancel 후 Create 실패 시 exchange_order_id가 None으로 초기화."""

    @pytest.mark.asyncio
    async def test_create_failure_clears_exchange_order_id(self) -> None:
        """Cancel 성공 + Create 실패 → exchange_order_id=None."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        client.cancel_order = AsyncMock(return_value={})
        client.create_stop_market_order = AsyncMock(side_effect=Exception("API error"))
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="old_order_id",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr._update_stop_if_needed("BTC/USDT", 45000.0)

        # exchange_order_id가 None으로 초기화되어야 함
        assert mgr._stops["BTC/USDT"].exchange_order_id is None
        assert mgr._stops["BTC/USDT"].placement_failures == 1

    @pytest.mark.asyncio
    async def test_next_update_skips_cancel_when_id_none(self) -> None:
        """exchange_order_id=None → cancel 시도 없이 create만 시도."""
        config = _make_config(exchange_stop_update_threshold=0.005)
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id=None,  # 이전 create 실패로 None
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr._update_stop_if_needed("BTC/USDT", 45000.0)

        # cancel 호출 없이 create만 호출
        client.cancel_order.assert_not_called()
        client.create_stop_market_order.assert_called_once()
        assert mgr._stops["BTC/USDT"].exchange_order_id == "stop_order_123"


# ---------------------------------------------------------------------------
# Discord Alert (HIGH-3)
# ---------------------------------------------------------------------------


class TestDiscordAlert:
    """5회 연속 실패 → Discord CRITICAL 알림."""

    @pytest.mark.asyncio
    async def test_critical_enqueues_notification(self) -> None:
        """5회 연속 실패 시 notification_queue.enqueue 호출."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()

        mock_queue = AsyncMock()
        mgr = ExchangeStopManager(config, client, pm, notification_queue=mock_queue)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id=None,
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
            placement_failures=5,
        )

        await mgr._check_failure_threshold("BTC/USDT")

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity.value == "critical"

    @pytest.mark.asyncio
    async def test_no_queue_no_error(self) -> None:
        """queue=None → 에러 없이 로그만."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)  # no queue

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id=None,
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
            placement_failures=5,
        )

        # 에러 없이 실행
        await mgr._check_failure_threshold("BTC/USDT")


# ---------------------------------------------------------------------------
# Verify Exchange Stops (MED-1)
# ---------------------------------------------------------------------------


class TestVerifyExchangeStops:
    """재시작 후 거래소 실제 주문 존재 여부 검증."""

    @pytest.mark.asyncio
    async def test_valid_order_retained(self) -> None:
        """거래소에 존재하는 주문 → 유지."""
        config = _make_config()
        client = _make_futures_client()
        client.fetch_open_orders = AsyncMock(return_value=[{"id": "stop_123"}])
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="stop_123",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr.verify_exchange_stops()

        assert "BTC/USDT" in mgr._stops

    @pytest.mark.asyncio
    async def test_missing_order_removed(self) -> None:
        """거래소에 없는 주문 → 제거."""
        config = _make_config()
        client = _make_futures_client()
        client.fetch_open_orders = AsyncMock(return_value=[])  # 빈 목록
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="stop_123",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr.verify_exchange_stops()

        assert "BTC/USDT" not in mgr._stops

    @pytest.mark.asyncio
    async def test_api_failure_retains(self) -> None:
        """API 에러 → 보수적으로 유지."""
        config = _make_config()
        client = _make_futures_client()
        client.fetch_open_orders = AsyncMock(side_effect=Exception("Network error"))
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id="stop_123",
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr.verify_exchange_stops()

        assert "BTC/USDT" in mgr._stops

    @pytest.mark.asyncio
    async def test_none_id_removed(self) -> None:
        """exchange_order_id=None → 제거."""
        config = _make_config()
        client = _make_futures_client()
        pm = _make_pm()
        mgr = ExchangeStopManager(config, client, pm)

        mgr._stops["BTC/USDT"] = StopOrderState(
            symbol="BTC/USDT",
            exchange_order_id=None,
            client_order_id="c1",
            stop_price=44000.0,
            position_side="LONG",
            close_side="sell",
        )

        await mgr.verify_exchange_stops()

        assert "BTC/USDT" not in mgr._stops
