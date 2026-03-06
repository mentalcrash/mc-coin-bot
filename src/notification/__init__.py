"""Notification module for trading alerts.

This module provides a unified notification interface for the trading bot,
supporting Discord Bot (slash commands + channel alerts) and webhook fallback.

Rules Applied:
    - #22 Notification Standards: Discord Bot + webhooks, Rich embeds
    - #10 Python Standards: Async patterns, type hints
"""

from src.notification.bot import DiscordBotService, TradingContext
from src.notification.config import DiscordBotConfig
from src.notification.discord import (
    DiscordColor,
    DiscordNotifier,
    send_discord_alert,
    send_error_alert,
    send_trade_alert,
)
from src.notification.engine import NotificationEngine
from src.notification.models import ChannelRoute, NotificationItem, Severity
from src.notification.queue import NotificationQueue, SpamGuard

__all__ = [
    "ChannelRoute",
    "DiscordBotConfig",
    "DiscordBotService",
    "DiscordColor",
    "DiscordNotifier",
    "NotificationEngine",
    "NotificationItem",
    "NotificationQueue",
    "Severity",
    "SpamGuard",
    "TradingContext",
    "send_discord_alert",
    "send_error_alert",
    "send_trade_alert",
]
