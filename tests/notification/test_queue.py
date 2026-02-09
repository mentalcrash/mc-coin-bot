"""SpamGuard + NotificationQueue 테스트."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from src.notification.models import ChannelRoute, NotificationItem, Severity
from src.notification.queue import NotificationQueue, SpamGuard

# 테스트용 빠른 backoff
_FAST_BACKOFF = 0.01


def _make_item(
    channel: ChannelRoute = ChannelRoute.TRADE_LOG,
    spam_key: str | None = None,
) -> NotificationItem:
    return NotificationItem(
        severity=Severity.INFO,
        channel=channel,
        embed={"title": "Test"},
        spam_key=spam_key,
    )


class TestSpamGuard:
    def test_first_send_allowed(self) -> None:
        guard = SpamGuard(cooldown_seconds=300.0)
        assert guard.should_send("key1") is True

    def test_second_send_blocked(self) -> None:
        guard = SpamGuard(cooldown_seconds=300.0)
        assert guard.should_send("key1") is True
        assert guard.should_send("key1") is False

    def test_different_keys_independent(self) -> None:
        guard = SpamGuard(cooldown_seconds=300.0)
        assert guard.should_send("key1") is True
        assert guard.should_send("key2") is True

    def test_cooldown_expired(self) -> None:
        guard = SpamGuard(cooldown_seconds=0.01)
        assert guard.should_send("key1") is True
        time.sleep(0.02)
        assert guard.should_send("key1") is True


class TestNotificationQueue:
    async def test_enqueue_and_send(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=True)
        queue = NotificationQueue(sender, queue_size=10, base_backoff=_FAST_BACKOFF)

        item = _make_item()
        await queue.enqueue(item)

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.05)
        await queue.stop()
        await worker

        sender.send_embed.assert_called_once_with(ChannelRoute.TRADE_LOG, item.embed)

    async def test_spam_key_throttled(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=True)
        queue = NotificationQueue(
            sender, queue_size=10, cooldown_seconds=300.0, base_backoff=_FAST_BACKOFF
        )

        item1 = _make_item(spam_key="same_key")
        item2 = _make_item(spam_key="same_key")
        await queue.enqueue(item1)
        await queue.enqueue(item2)  # should be dropped by SpamGuard

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.05)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 1

    async def test_no_spam_key_always_sent(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=True)
        queue = NotificationQueue(sender, queue_size=10, base_backoff=_FAST_BACKOFF)

        await queue.enqueue(_make_item())
        await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.05)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 2

    async def test_queue_full_drops(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=True)
        queue = NotificationQueue(sender, queue_size=2, base_backoff=_FAST_BACKOFF)

        await queue.enqueue(_make_item())
        await queue.enqueue(_make_item())
        await queue.enqueue(_make_item())  # should be dropped (full)

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.05)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 2

    async def test_retry_on_failure(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(side_effect=[False, False, True])
        queue = NotificationQueue(
            sender, max_retries=3, queue_size=10, base_backoff=_FAST_BACKOFF
        )

        await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.1)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 3

    async def test_retry_exhausted(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=False)
        queue = NotificationQueue(
            sender, max_retries=2, queue_size=10, base_backoff=_FAST_BACKOFF
        )

        await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.1)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 2

    async def test_drain_on_stop(self) -> None:
        """stop() 호출 후 큐 잔여 아이템을 drain."""
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=True)
        queue = NotificationQueue(sender, queue_size=10, base_backoff=_FAST_BACKOFF)

        # worker 시작 → 아이템 2개 enqueue → 즉시 stop
        worker = asyncio.create_task(queue.start())

        await queue.enqueue(_make_item())
        await queue.enqueue(_make_item())

        # worker가 큐를 처리할 시간
        await asyncio.sleep(0.05)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 2
