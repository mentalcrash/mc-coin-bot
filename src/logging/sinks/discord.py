"""Discord Webhook Sink for loguru.

This module provides a non-blocking Discord webhook sink that sends
ERROR and CRITICAL level logs as Rich Embed messages. It integrates
with the trading bot's alerting pipeline for immediate notification.

Rules Applied:
    - #22 Notification Standards: Discord webhooks, Rich Embeds, aiohttp
    - #15 Logging Standards: Non-blocking I/O
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

import asyncio
import atexit
import queue
import re
import sys
import threading
import time
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

import aiohttp

# Discord API limits
DISCORD_MSG_LIMIT = 2000
DISCORD_RATE_LIMIT_STATUS = 429
DISCORD_SUCCESS_STATUSES = (200, 204)


class DiscordColor(IntEnum):
    """Discord Embed color codes (decimal format).

    Standard color palette following Rule #22.
    """

    GREEN = 5763719  # Success/Buy - #57F287
    RED = 15548997  # Error/Sell - #ED4245
    BLUE = 3447003  # Info - #3498DB
    YELLOW = 16776960  # Warning - #FFFF00
    ORANGE = 15105570  # Critical - #E67E22


class DiscordWebhookSink:
    """Non-blocking Discord webhook sink for loguru.

    Sends log messages to Discord as Rich Embed messages. Uses a background
    thread with its own event loop to avoid blocking the trading bot's
    main event loop.

    Args:
        webhook_url: Discord webhook URL
        min_level: Minimum log level to send (default: ERROR)
        username: Bot username shown in Discord
        avatar_url: Bot avatar URL
        footer_text: Footer text for embeds
        rate_limit_per_second: Max messages per second (Discord limit is 5)

    Example:
        >>> from loguru import logger
        >>> sink = DiscordWebhookSink(
        ...     webhook_url="https://discord.com/api/webhooks/...",
        ...     min_level="ERROR"
        ... )
        >>> logger.add(sink.write, level="ERROR", format="{message}")
    """

    # Level name to color mapping
    LEVEL_COLORS: dict[str, DiscordColor] = {
        "TRACE": DiscordColor.BLUE,
        "DEBUG": DiscordColor.BLUE,
        "INFO": DiscordColor.BLUE,
        "SUCCESS": DiscordColor.GREEN,
        "WARNING": DiscordColor.YELLOW,
        "ERROR": DiscordColor.RED,
        "CRITICAL": DiscordColor.ORANGE,
    }

    def __init__(
        self,
        webhook_url: str,
        min_level: str = "ERROR",
        username: str = "MC Coin Bot",
        avatar_url: str | None = None,
        footer_text: str = "MC-Coin-Bot • 2026 Edition",
        rate_limit_per_second: float = 4.0,
    ) -> None:
        """Initialize Discord webhook sink.

        Args:
            webhook_url: Discord webhook URL
            min_level: Minimum level to forward (default ERROR)
            username: Display name in Discord
            avatar_url: Avatar image URL (optional)
            footer_text: Footer text for embeds
            rate_limit_per_second: Rate limit (default 4/sec, Discord max is 5)
        """
        self._url = webhook_url
        self._min_level = min_level.upper()
        self._username = username
        self._avatar_url = avatar_url
        self._footer_text = footer_text
        self._rate_limit_interval = 1.0 / rate_limit_per_second

        # Message queue for async processing
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=100)
        self._running = True
        self._last_send_time = 0.0

        # Start background worker thread with its own event loop
        self._worker = threading.Thread(
            target=self._run_async_loop,
            name="DiscordWebhookWorker",
            daemon=True,
        )
        self._worker.start()

        atexit.register(self.stop)

    def write(self, message: str) -> None:
        """Write a log message to the Discord queue.

        This method is called by loguru. It extracts metadata from the
        message and enqueues it for async sending.

        Args:
            message: Formatted log message from loguru
        """
        if not self._running:
            return

        # Extract level from message (loguru format: "LEVEL | message")
        level = self._extract_level(message)
        color = self.LEVEL_COLORS.get(level, DiscordColor.RED)

        embed = self._create_embed(
            title=f"[{level}] Trading Bot Alert",
            description=self._sanitize_message(message),
            color=color,
        )

        payload = {
            "username": self._username,
            "embeds": [embed],
        }
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Queue full - drop oldest and add new
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except queue.Empty:
                pass

    def _extract_level(self, message: str) -> str:
        """Extract log level from formatted message.

        Args:
            message: Loguru formatted message

        Returns:
            Uppercase level name (default: ERROR)
        """
        # Try to match common loguru formats
        patterns = [
            r"\|\s*(TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\s*\|",
            r"<level>(TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)</level>",
            r"^\s*(TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\s+",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return "ERROR"

    def _sanitize_message(self, message: str) -> str:
        """Sanitize message for Discord (remove ANSI codes, limit length).

        Args:
            message: Raw log message

        Returns:
            Cleaned message (max 2000 chars for Discord limit)
        """
        # Remove ANSI color codes
        ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
        cleaned = ansi_pattern.sub("", message)

        # Remove loguru markup tags
        markup_pattern = re.compile(r"<[^>]+>")
        cleaned = markup_pattern.sub("", cleaned)

        # Truncate to Discord limit
        if len(cleaned) > DISCORD_MSG_LIMIT:
            cleaned = cleaned[: DISCORD_MSG_LIMIT - 3] + "..."

        return cleaned.strip()

    def _create_embed(
        self,
        title: str,
        description: str,
        color: DiscordColor,
    ) -> dict[str, Any]:
        """Create a Discord embed object.

        Args:
            title: Embed title
            description: Embed description (the log message)
            color: Embed color

        Returns:
            Discord embed dictionary
        """
        return {
            "title": title,
            "description": f"```\n{description}\n```",
            "color": int(color),
            "timestamp": datetime.now(UTC).isoformat(),
            "footer": {"text": self._footer_text},
        }

    def _run_async_loop(self) -> None:
        """Run the async event loop in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._consume())
        finally:
            loop.close()

    async def _consume(self) -> None:
        """Consume messages from queue and send to Discord."""
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    # Non-blocking get with timeout
                    payload = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._queue.get(timeout=1.0),
                    )
                    if payload is None:
                        break

                    # Rate limiting
                    await self._apply_rate_limit()

                    # Send to Discord
                    await self._send(session, payload)

                except queue.Empty:
                    continue
                except Exception as e:
                    # Log to stderr, don't crash the worker
                    print(f"[DiscordSink] Error: {e}", file=sys.stderr)

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to respect Discord's limits."""
        elapsed = time.time() - self._last_send_time
        if elapsed < self._rate_limit_interval:
            await asyncio.sleep(self._rate_limit_interval - elapsed)
        self._last_send_time = time.time()

    async def _send(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> None:
        """Send payload to Discord webhook.

        Args:
            session: aiohttp session
            payload: Discord webhook payload
        """
        try:
            async with session.post(
                self._url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == DISCORD_RATE_LIMIT_STATUS:
                    # Rate limited - wait and retry
                    retry_after = float(response.headers.get("Retry-After", "5"))
                    await asyncio.sleep(retry_after)
                    await self._send(session, payload)
                elif response.status not in DISCORD_SUCCESS_STATUSES:
                    text = await response.text()
                    print(
                        f"[DiscordSink] HTTP {response.status}: {text}",
                        file=sys.stderr,
                    )
        except TimeoutError:
            pass  # Silent timeout - don't block trading
        except aiohttp.ClientError:
            pass  # Network error - don't block trading

    def stop(self) -> None:
        """Gracefully stop the worker thread."""
        if not self._running:
            return
        self._running = False
        self._queue.put(None)  # Sentinel
        self._worker.join(timeout=5.0)

    @property
    def queue_size(self) -> int:
        """Current number of pending messages."""
        return self._queue.qsize()
