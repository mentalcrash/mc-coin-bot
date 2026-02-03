"""Notification module for trading alerts.

This module provides a unified notification interface for the trading bot,
supporting multiple channels (Discord, Telegram) with Rich Embed formatting.

Rules Applied:
    - #22 Notification Standards: Discord webhooks, Rich embeds, aiohttp
    - #10 Python Standards: Async patterns, type hints
"""

from src.notification.discord import (
    DiscordColor,
    DiscordNotifier,
    send_discord_alert,
    send_error_alert,
    send_trade_alert,
)

__all__ = [
    "DiscordColor",
    "DiscordNotifier",
    "send_discord_alert",
    "send_error_alert",
    "send_trade_alert",
]
