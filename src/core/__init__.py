"""Core module - Single Source of Truth for shared components."""

from src.core.events import Event, EventBus, EventType
from src.core.models import (
    Candle,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
)
from src.core.strategy import BaseStrategy

__all__ = [
    # Events
    "Event",
    "EventBus",
    "EventType",
    # Models
    "Candle",
    "Fill",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "Signal",
    "SignalType",
    # Strategy
    "BaseStrategy",
]
