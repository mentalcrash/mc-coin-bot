"""SpotStopManager 단위 테스트.

Mock BinanceSpotClient로 Spot STOP_LOSS_LIMIT lifecycle 검증.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent, FillEvent
from src.models.types import Direction


@pytest.fixture()
def mock_spot_client() -> MagicMock:
    """Mock BinanceSpotClient."""
    client = MagicMock()
    client.create_stop_limit_sell = AsyncMock(return_value={"id": "stop_order_1"})
    client.cancel_order = AsyncMock(return_value={"status": "cancelled"})
    client.fetch_open_orders = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def mock_pm() -> MagicMock:
    """Mock EDAPortfolioManager."""
    pm = MagicMock()
    pm.positions = {}
    return pm


@pytest.fixture()
def mock_config() -> MagicMock:
    """Mock PortfolioManagerConfig."""
    config = MagicMock()
    config.system_stop_loss = 0.05  # 5%
    config.exchange_safety_margin = 0.01  # 1%
    config.use_trailing_stop = True
    config.trailing_stop_atr_multiplier = 3.0
    config.exchange_trailing_safety_margin = 0.005
    config.exchange_stop_update_threshold = 0.005  # 0.5%
    config.use_exchange_safety_stop = True
    config.cancel_stops_on_shutdown = True
    return config


def _make_position(
    entry: float = 50000.0,
    size: float = 0.01,
    peak: float = 55000.0,
    atr_values: list[float] | None = None,
) -> MagicMock:
    """Mock Position (LONG only for Spot)."""
    pos = MagicMock()
    pos.is_open = True
    pos.direction = Direction.LONG
    pos.avg_entry_price = entry
    pos.size = size
    pos.peak_price_since_entry = peak
    pos.last_price = peak
    pos.atr_values = atr_values or [1000.0] * 14
    return pos


# =========================================================================
# FILL → stop 배치
# =========================================================================


@pytest.mark.asyncio()
async def test_fill_event_places_stop(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """FILL 이벤트 → stop 배치 검증."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position()
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    bus = EventBus()
    await mgr.register(bus)

    fill = FillEvent(
        client_order_id="test_fill",
        symbol="BTC/USDT",
        side="BUY",
        fill_price=50000.0,
        fill_qty=0.01,
        fee=0.05,
        fill_timestamp=datetime.now(UTC),
        source="test",
    )

    await mgr._on_fill(fill)

    assert "BTC/USDT" in mgr.active_stops
    state = mgr.active_stops["BTC/USDT"]
    assert state.base_amount == 0.01
    assert state.exchange_order_id == "stop_order_1"
    mock_spot_client.create_stop_limit_sell.assert_awaited_once()


# =========================================================================
# BAR → ratchet up
# =========================================================================


@pytest.mark.asyncio()
async def test_bar_event_ratchets_stop_up(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """BAR 이벤트 → ratchet up (stop 상승) 검증.

    SL stop = entry*(1-0.06) = 47000 (고정, entry 기준).
    TS stop = (peak - 3*ATR)*(1-0.005). peak 55000→TS=51850, peak 65000→TS=56730.
    stop = min(SL, TS) → 47000 vs 51850 → 47000 (SL 우세).
    SL 없애면 TS만 테스트 가능.
    """
    from src.eda.spot_stop_manager import SpotStopManager

    # SL 비활성 → TS만 테스트
    mock_config.system_stop_loss = None
    pos = _make_position(peak=55000.0)
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)

    # 초기 stop 배치
    await mgr._place_safety_stop("BTC/USDT", pos)
    initial_stop = mgr.active_stops["BTC/USDT"].stop_price

    # Peak 상승 → stop도 올라야 함
    pos.peak_price_since_entry = 65000.0
    mock_spot_client.create_stop_limit_sell.return_value = {"id": "stop_order_2"}

    bar = BarEvent(
        symbol="BTC/USDT",
        timeframe="12h",
        open=64000.0,
        high=65000.0,
        low=63000.0,
        close=64500.0,
        volume=1000.0,
        bar_timestamp=datetime.now(UTC),
    )

    await mgr._on_bar(bar)

    new_stop = mgr.active_stops["BTC/USDT"].stop_price
    assert new_stop > initial_stop


# =========================================================================
# BAR → ratchet 하향 시 유지
# =========================================================================


@pytest.mark.asyncio()
async def test_bar_event_no_ratchet_down(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """BAR 이벤트: peak 하락 시 stop 유지 (하향 불가)."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position(peak=55000.0)
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)
    initial_stop = mgr.active_stops["BTC/USDT"].stop_price

    # Peak 하락 → stop 유지
    pos.peak_price_since_entry = 50000.0

    bar = BarEvent(
        symbol="BTC/USDT",
        timeframe="12h",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50000.0,
        volume=1000.0,
        bar_timestamp=datetime.now(UTC),
    )

    await mgr._on_bar(bar)

    assert mgr.active_stops["BTC/USDT"].stop_price == initial_stop


# =========================================================================
# Cancel + replace 시퀀스
# =========================================================================


@pytest.mark.asyncio()
async def test_cancel_and_replace(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Update: cancel old + place new. SL 비활성 → TS만 테스트."""
    from src.eda.spot_stop_manager import SpotStopManager

    mock_config.system_stop_loss = None  # SL 비활성 → TS만 사용
    pos = _make_position(peak=55000.0)
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)

    old_order_id = mgr.active_stops["BTC/USDT"].exchange_order_id
    old_stop = mgr.active_stops["BTC/USDT"].stop_price

    # Ratchet up: peak 크게 상승
    pos.peak_price_since_entry = 70000.0
    mock_spot_client.create_stop_limit_sell.return_value = {"id": "stop_order_new"}

    new_stop = mgr._calculate_stop_price(pos)
    assert new_stop is not None and new_stop > old_stop

    await mgr._update_stop_if_needed("BTC/USDT", new_stop, pos.size)

    mock_spot_client.cancel_order.assert_awaited_once_with(old_order_id, "BTC/USDT")
    assert mgr.active_stops["BTC/USDT"].exchange_order_id == "stop_order_new"


# =========================================================================
# limit_price 계산
# =========================================================================


@pytest.mark.asyncio()
async def test_limit_price_calculation(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """limit_price = stop * (1 - 0.005) 검증."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position()
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)

    state = mgr.active_stops["BTC/USDT"]
    expected_limit = state.stop_price * (1.0 - 0.005)
    assert state.limit_price == pytest.approx(expected_limit, rel=1e-6)


# =========================================================================
# verify_exchange_stops 복구 로직
# =========================================================================


@pytest.mark.asyncio()
async def test_verify_exchange_stops_removes_stale(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """verify_exchange_stops: 거래소에 없는 주문 → 제거."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position()
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)

    # 거래소에 해당 주문 없음
    mock_spot_client.fetch_open_orders.return_value = []

    await mgr.verify_exchange_stops()

    assert "BTC/USDT" not in mgr.active_stops


@pytest.mark.asyncio()
async def test_verify_exchange_stops_retains_valid(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """verify_exchange_stops: 거래소에 있는 주문 → 유지."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position()
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)

    order_id = mgr.active_stops["BTC/USDT"].exchange_order_id
    mock_spot_client.fetch_open_orders.return_value = [{"id": order_id}]

    await mgr.verify_exchange_stops()

    assert "BTC/USDT" in mgr.active_stops


# =========================================================================
# State persistence
# =========================================================================


@pytest.mark.asyncio()
async def test_state_save_and_restore(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """get_state / restore_state 라운드트립."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position()
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)

    state = mgr.get_state()
    assert "stops" in state

    # 새 mgr에서 복원
    mgr2 = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    mgr2.restore_state(state)

    assert "BTC/USDT" in mgr2.active_stops
    assert mgr2.active_stops["BTC/USDT"].stop_price == mgr.active_stops["BTC/USDT"].stop_price


# =========================================================================
# cancel_all_stops
# =========================================================================


@pytest.mark.asyncio()
async def test_cancel_all_stops(
    mock_spot_client: MagicMock,
    mock_pm: MagicMock,
    mock_config: MagicMock,
) -> None:
    """cancel_all_stops: 모든 stop 취소."""
    from src.eda.spot_stop_manager import SpotStopManager

    pos = _make_position()
    mock_pm.positions = {"BTC/USDT": pos}

    mgr = SpotStopManager(mock_config, mock_spot_client, mock_pm)
    await mgr._place_safety_stop("BTC/USDT", pos)
    assert len(mgr.active_stops) == 1

    await mgr.cancel_all_stops()
    assert len(mgr.active_stops) == 0


# =========================================================================
# SL stop 계산 (TS 없을 때)
# =========================================================================


def test_calculate_sl_stop_only(mock_config: MagicMock) -> None:
    """system_stop_loss=5% + margin=1% → entry * (1 - 0.06)."""
    from src.eda.spot_stop_manager import SpotStopManager

    mock_config.use_trailing_stop = False
    pos = _make_position(entry=50000.0, atr_values=[])
    mgr = SpotStopManager(mock_config, MagicMock(), MagicMock())

    stop = mgr._calculate_stop_price(pos)
    expected = 50000.0 * (1.0 - 0.05 - 0.01)
    assert stop == pytest.approx(expected, rel=1e-6)


# =========================================================================
# is_safety_stop_order
# =========================================================================


def test_is_safety_stop_order() -> None:
    """prefix 매칭 검증."""
    from src.eda.spot_stop_manager import SpotStopManager

    mgr = SpotStopManager(MagicMock(), MagicMock(), MagicMock())
    assert mgr.is_safety_stop_order("spot-stop-BTC-USDT_abc123") is True
    assert mgr.is_safety_stop_order("safety-stop-BTC-USDT_abc123") is False
    assert mgr.is_safety_stop_order("random_order_id") is False
