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

        sender.send_embed.assert_called_once_with(ChannelRoute.TRADE_LOG, item.embed, ())

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
        queue = NotificationQueue(sender, max_retries=3, queue_size=10, base_backoff=_FAST_BACKOFF)

        await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.1)
        await queue.stop()
        await worker

        assert sender.send_embed.call_count == 3

    async def test_retry_exhausted(self) -> None:
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=False)
        queue = NotificationQueue(sender, max_retries=2, queue_size=10, base_backoff=_FAST_BACKOFF)

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


class TestNotificationQueueDegradation:
    """NotificationQueue graceful degradation 테스트."""

    async def test_degraded_mode_after_consecutive_failures(self) -> None:
        """연속 실패 시 degraded 모드 진입."""
        sender = AsyncMock()
        sender.send_embed = AsyncMock(return_value=False)
        queue = NotificationQueue(
            sender, max_retries=1, queue_size=20, base_backoff=_FAST_BACKOFF
        )
        assert queue.is_degraded is False

        # 5개 이상 연속 실패 시 degraded 진입
        for _ in range(6):
            await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.2)
        await queue.stop()
        await worker

        assert queue.is_degraded is True
        assert queue.total_dropped == 6

    async def test_recovery_from_degraded_mode(self) -> None:
        """전송 성공 시 degraded 모드 복구."""
        sender = AsyncMock()
        results = [False] * 5 + [True]
        sender.send_embed = AsyncMock(side_effect=results)
        queue = NotificationQueue(
            sender, max_retries=1, queue_size=20, base_backoff=_FAST_BACKOFF
        )

        for _ in range(6):
            await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.2)
        await queue.stop()
        await worker

        # 마지막 전송 성공 → degraded 해제
        assert queue.is_degraded is False
        assert queue.total_dropped == 5  # 처음 5개만 드롭

    async def test_consecutive_failures_reset_on_success(self) -> None:
        """전송 성공 시 연속 실패 카운터 리셋."""
        sender = AsyncMock()
        sender.send_embed = AsyncMock(side_effect=[False, True, False])
        queue = NotificationQueue(
            sender, max_retries=1, queue_size=10, base_backoff=_FAST_BACKOFF
        )

        for _ in range(3):
            await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.15)
        await queue.stop()
        await worker

        # 실패→성공→실패 = consecutive=1 (리셋 후 1)
        assert queue._consecutive_failures == 1
        assert queue.total_dropped == 2

    async def test_total_dropped_persists(self) -> None:
        """total_dropped는 성공 후에도 유지."""
        sender = AsyncMock()
        sender.send_embed = AsyncMock(side_effect=[False, False, True])
        queue = NotificationQueue(
            sender, max_retries=1, queue_size=10, base_backoff=_FAST_BACKOFF
        )

        for _ in range(3):
            await queue.enqueue(_make_item())

        worker = asyncio.create_task(queue.start())
        await asyncio.sleep(0.15)
        await queue.stop()
        await worker

        assert queue.total_dropped == 2  # 리셋 후에도 총 드롭 수 유지
