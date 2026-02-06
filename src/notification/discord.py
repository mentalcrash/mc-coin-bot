"""Discord notification service with Rich Embeds.

This module provides a complete Discord notification service for the trading
bot, supporting trade alerts, error notifications, and daily reports with
visually rich embed formatting.

Rules Applied:
    - #22 Notification Standards: aiohttp, Rich Embeds, channel segmentation
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import aiohttp

from src.logging.config import DiscordChannelConfig, get_discord_config

if TYPE_CHECKING:
    from decimal import Decimal

# Discord API limits and status codes
DISCORD_RATE_LIMIT_STATUS = 429
DISCORD_SUCCESS_STATUSES = (200, 204)
CONTEXT_TRUNCATE_LIMIT = 1000


class DiscordColor(IntEnum):
    """Discord Embed color codes (decimal format).

    Standard color palette following Rule #22.
    """

    GREEN = 5763719  # Success/Buy - #57F287
    RED = 15548997  # Error/Sell - #ED4245
    BLUE = 3447003  # Info - #3498DB
    YELLOW = 16776960  # Warning - #FFFF00
    ORANGE = 15105570  # Critical - #E67E22
    PURPLE = 10181046  # Special - #9B59B6


class DiscordNotifier:
    """Discord notification service with Rich Embed support.

    Provides methods for sending various types of trading alerts
    to Discord channels via webhooks. Supports channel segmentation
    for different alert types.

    Args:
        config: Discord channel configuration (loads from env if None)
        session: Optional aiohttp session (creates new if None)

    Example:
        >>> notifier = DiscordNotifier()
        >>> await notifier.send_trade_alert(
        ...     side="BUY",
        ...     symbol="BTC/USDT",
        ...     price=Decimal("50000.00"),
        ...     amount=Decimal("0.1"),
        ... )
    """

    def __init__(
        self,
        config: DiscordChannelConfig | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize Discord notifier.

        Args:
            config: Channel configuration (loads from env if None)
            session: Existing aiohttp session (optional)
        """
        self._config = config or get_discord_config()
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session if we own it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def send_trade_alert(
        self,
        side: str,
        symbol: str,
        price: Decimal,
        amount: Decimal,
        *,
        strategy: str | None = None,
        pnl: Decimal | None = None,
        pnl_percent: float | None = None,
    ) -> bool:
        """Send a trade execution alert.

        Args:
            side: Trade side ("BUY" or "SELL")
            symbol: Trading symbol
            price: Execution price
            amount: Trade amount
            strategy: Strategy name (optional)
            pnl: Profit/Loss in quote currency (optional)
            pnl_percent: Profit/Loss percentage (optional)

        Returns:
            True if sent successfully
        """
        if not self._config.trade_webhook_url:
            return False

        is_buy = side.upper() == "BUY"
        emoji = "🚀" if is_buy else "💰"
        color = DiscordColor.GREEN if is_buy else DiscordColor.RED
        value = price * amount

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Side", "value": side.upper(), "inline": True},
            {"name": "Price", "value": f"${price:,.4f}", "inline": True},
            {"name": "Amount", "value": f"{amount:,.6f}", "inline": True},
            {"name": "Value", "value": f"${value:,.2f}", "inline": True},
        ]

        if strategy:
            fields.append({"name": "Strategy", "value": strategy, "inline": True})

        if pnl is not None:
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_str = f"{pnl_emoji} ${pnl:+,.2f}"
            if pnl_percent is not None:
                pnl_str += f" ({pnl_percent:+.2f}%)"
            fields.append({"name": "PnL", "value": pnl_str, "inline": True})

        embed = self._create_embed(
            title=f"{emoji} {side.upper()} EXECUTED: {symbol}",
            color=color,
            fields=fields,
        )

        return await self._send(self._config.trade_webhook_url, embed)

    async def send_error_alert(
        self,
        error_type: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        traceback: str | None = None,
    ) -> bool:
        """Send an error alert.

        Args:
            error_type: Type of error (e.g., "NetworkError", "CriticalError")
            message: Error message
            context: Additional context (optional)
            traceback: Stack trace (optional, truncated to 1000 chars)

        Returns:
            True if sent successfully
        """
        if not self._config.error_webhook_url:
            return False

        # Determine color based on error type
        is_critical = "critical" in error_type.lower()
        color = DiscordColor.ORANGE if is_critical else DiscordColor.RED
        emoji = "🚨" if is_critical else "❌"

        fields = [
            {"name": "Error Type", "value": f"`{error_type}`", "inline": True},
        ]

        if context:
            ctx_str = "\n".join(f"**{k}:** {v}" for k, v in context.items())
            if len(ctx_str) > CONTEXT_TRUNCATE_LIMIT:
                ctx_str = ctx_str[: CONTEXT_TRUNCATE_LIMIT - 3] + "..."
            fields.append({"name": "Context", "value": ctx_str, "inline": False})

        description = message
        if traceback:
            tb_truncated = (
                traceback[:CONTEXT_TRUNCATE_LIMIT] + "..."
                if len(traceback) > CONTEXT_TRUNCATE_LIMIT
                else traceback
            )
            description += f"\n\n```python\n{tb_truncated}\n```"

        embed = self._create_embed(
            title=f"{emoji} {error_type}",
            description=description,
            color=color,
            fields=fields,
        )

        return await self._send(self._config.error_webhook_url, embed)

    async def send_daily_report(
        self,
        date: str,
        total_trades: int,
        win_rate: float,
        total_pnl: Decimal,
        total_pnl_percent: float,
        *,
        best_trade: dict[str, Any] | None = None,
        worst_trade: dict[str, Any] | None = None,
    ) -> bool:
        """Send a daily trading report.

        Args:
            date: Report date (YYYY-MM-DD)
            total_trades: Total number of trades
            win_rate: Win rate percentage
            total_pnl: Total PnL in quote currency
            total_pnl_percent: Total PnL percentage
            best_trade: Best trade of the day (optional)
            worst_trade: Worst trade of the day (optional)

        Returns:
            True if sent successfully
        """
        if not self._config.report_webhook_url:
            return False

        # Color based on PnL
        if total_pnl > 0:
            color = DiscordColor.GREEN
            emoji = "📈"
        elif total_pnl < 0:
            color = DiscordColor.RED
            emoji = "📉"
        else:
            color = DiscordColor.BLUE
            emoji = "-"  # Neutral/unchanged

        fields = [
            {"name": "Date", "value": date, "inline": True},
            {"name": "Total Trades", "value": str(total_trades), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "Total PnL", "value": f"${total_pnl:+,.2f}", "inline": True},
            {"name": "PnL %", "value": f"{total_pnl_percent:+.2f}%", "inline": True},
        ]

        if best_trade:
            best_str = f"{best_trade.get('symbol', 'N/A')}: +${best_trade.get('pnl', 0):,.2f}"
            fields.append({"name": "Best Trade 🏆", "value": best_str, "inline": True})

        if worst_trade:
            worst_str = f"{worst_trade.get('symbol', 'N/A')}: ${worst_trade.get('pnl', 0):,.2f}"
            fields.append({"name": "Worst Trade 💀", "value": worst_str, "inline": True})

        embed = self._create_embed(
            title=f"{emoji} Daily Report: {date}",
            color=color,
            fields=fields,
        )

        return await self._send(self._config.report_webhook_url, embed)

    async def send_custom_alert(
        self,
        webhook_url: str,
        title: str,
        *,
        description: str | None = None,
        color: DiscordColor = DiscordColor.BLUE,
        fields: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send a custom alert to any webhook.

        Args:
            webhook_url: Destination webhook URL
            title: Embed title
            description: Embed description (optional)
            color: Embed color
            fields: Embed fields (optional)

        Returns:
            True if sent successfully
        """
        embed = self._create_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
        )
        return await self._send(webhook_url, embed)

    def _create_embed(
        self,
        title: str,
        color: DiscordColor,
        *,
        description: str | None = None,
        fields: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a Discord embed object.

        Args:
            title: Embed title
            color: Embed color
            description: Embed description
            fields: Embed fields

        Returns:
            Discord embed dictionary
        """
        embed: dict[str, Any] = {
            "title": title,
            "color": int(color),
            "timestamp": datetime.now(UTC).isoformat(),
            "footer": {"text": "MC-Coin-Bot • 2026 Edition"},
        }

        if description:
            embed["description"] = description

        if fields:
            embed["fields"] = fields

        return embed

    async def _send(self, webhook_url: str, embed: dict[str, Any]) -> bool:
        """Send embed to webhook.

        Args:
            webhook_url: Destination URL
            embed: Embed to send

        Returns:
            True if sent successfully
        """
        payload = {
            "username": "MC Coin Bot",
            "embeds": [embed],
        }

        try:
            session = await self._get_session()
            async with session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == DISCORD_RATE_LIMIT_STATUS:
                    # Rate limited - log and skip
                    return False
                return response.status in DISCORD_SUCCESS_STATUSES
        except (aiohttp.ClientError, TimeoutError):
            return False


# =============================================================================
# Convenience Functions
# =============================================================================


async def send_discord_alert(
    webhook_url: str,
    title: str,
    fields: list[dict[str, Any]],
    color: DiscordColor,
) -> bool:
    """Send a Discord alert (convenience function).

    Args:
        webhook_url: Discord webhook URL
        title: Embed title
        fields: Embed fields
        color: Embed color

    Returns:
        True if sent successfully

    Example:
        >>> await send_discord_alert(
        ...     webhook_url="https://discord.com/api/webhooks/...",
        ...     title="Alert",
        ...     fields=[{"name": "Status", "value": "OK", "inline": True}],
        ...     color=DiscordColor.GREEN,
        ... )
    """
    notifier = DiscordNotifier()
    try:
        return await notifier.send_custom_alert(
            webhook_url=webhook_url,
            title=title,
            color=color,
            fields=fields,
        )
    finally:
        await notifier.close()


async def send_trade_alert(
    side: str,
    symbol: str,
    price: Decimal,
    amount: Decimal,
    **kwargs: Any,
) -> bool:
    """Send a trade alert (convenience function).

    Uses configuration from environment variables.

    Args:
        side: Trade side
        symbol: Trading symbol
        price: Execution price
        amount: Trade amount
        **kwargs: Additional arguments for send_trade_alert

    Returns:
        True if sent successfully
    """
    notifier = DiscordNotifier()
    try:
        return await notifier.send_trade_alert(
            side=side,
            symbol=symbol,
            price=price,
            amount=amount,
            **kwargs,
        )
    finally:
        await notifier.close()


async def send_error_alert(
    error_type: str,
    message: str,
    **kwargs: Any,
) -> bool:
    """Send an error alert (convenience function).

    Uses configuration from environment variables.

    Args:
        error_type: Type of error
        message: Error message
        **kwargs: Additional arguments for send_error_alert

    Returns:
        True if sent successfully
    """
    notifier = DiscordNotifier()
    try:
        return await notifier.send_error_alert(
            error_type=error_type,
            message=message,
            **kwargs,
        )
    finally:
        await notifier.close()
