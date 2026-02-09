"""Notification 테스트 공통 fixture."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.events import (
    BalanceUpdateEvent,
    CircuitBreakerEvent,
    FillEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
)
from src.models.types import Direction


def _ts() -> datetime:
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_fill() -> FillEvent:
    return FillEvent(
        client_order_id="order-001",
        symbol="BTC/USDT",
        side="BUY",
        fill_price=50000.0,
        fill_qty=0.1,
        fee=5.0,
        fill_timestamp=_ts(),
    )


@pytest.fixture
def sample_fill_sell() -> FillEvent:
    return FillEvent(
        client_order_id="order-002",
        symbol="ETH/USDT",
        side="SELL",
        fill_price=3000.0,
        fill_qty=1.0,
        fee=3.0,
        fill_timestamp=_ts(),
    )


@pytest.fixture
def sample_circuit_breaker() -> CircuitBreakerEvent:
    return CircuitBreakerEvent(
        reason="System drawdown exceeded 10%",
        close_all_positions=True,
    )


@pytest.fixture
def sample_risk_alert_warning() -> RiskAlertEvent:
    return RiskAlertEvent(
        alert_level="WARNING",
        message="Drawdown approaching threshold: 7.5%",
    )


@pytest.fixture
def sample_risk_alert_critical() -> RiskAlertEvent:
    return RiskAlertEvent(
        alert_level="CRITICAL",
        message="Leverage exceeded maximum: 3.2x",
    )


@pytest.fixture
def sample_balance_update() -> BalanceUpdateEvent:
    return BalanceUpdateEvent(
        total_equity=10500.0,
        available_cash=8000.0,
        total_margin_used=2500.0,
    )


@pytest.fixture
def sample_position_update() -> PositionUpdateEvent:
    return PositionUpdateEvent(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        size=0.1,
        avg_entry_price=50000.0,
        unrealized_pnl=250.0,
        realized_pnl=100.0,
    )
