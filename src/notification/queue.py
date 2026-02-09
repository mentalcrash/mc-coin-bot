"""NotificationQueue -- 비동기 알림 큐.

거래 로직을 block하지 않도록 fire-and-forget으로 알림을 전송합니다.
SpamGuard로 동일 알림 반복 전송을 방지합니다.

Rules Applied:
    - #10 Python Standards: asyncio, type hints
    - EDA 패턴: bounded queue, graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Protocol

from loguru import logger

if TYPE_CHECKING:
    from src.notification.models import ChannelRoute, NotificationItem

# 큐 기본 크기
_DEFAULT_QUEUE_SIZE = 500
# 최대 재시도 횟수
_DEFAULT_MAX_RETRIES = 3
# SpamGuard 기본 cooldown (초)
_DEFAULT_COOLDOWN = 300.0
# 재시도 백오프 기본값 (초)
_BASE_BACKOFF = 1.0


class NotificationSender(Protocol):
    """알림 전송 프로토콜 (DiscordBotService가 구현)."""

    async def send_embed(self, channel: ChannelRoute, embed_dict: dict[str, object]) -> bool: ...


class SpamGuard:
    """동일 알림 반복 전송 방지 (cooldown 기반).

    동일한 spam_key를 가진 알림은 cooldown_seconds 간격으로만 전송됩니다.

    Args:
        cooldown_seconds: 동일 키 재전송 최소 간격 (초)
    """

    def __init__(self, cooldown_seconds: float = _DEFAULT_COOLDOWN) -> None:
        self._cooldown = cooldown_seconds
        self._last_sent: dict[str, float] = {}

    def should_send(self, key: str) -> bool:
        """해당 키의 알림을 전송해도 되는지 확인.

        Args:
            key: 스팸 체크 키

        Returns:
            True면 전송 허용
        """
        now = time.monotonic()
        last = self._last_sent.get(key)
        if last is not None and (now - last) < self._cooldown:
            return False
        self._last_sent[key] = now
        return True


class NotificationQueue:
    """비동기 알림 큐 -- 거래 로직 block 방지.

    Args:
        sender: 알림 전송자 (NotificationSender 프로토콜)
        max_retries: 전송 실패 시 최대 재시도 횟수
        queue_size: asyncio.Queue 크기
        cooldown_seconds: SpamGuard cooldown
    """

    def __init__(
        self,
        sender: NotificationSender,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        cooldown_seconds: float = _DEFAULT_COOLDOWN,
        base_backoff: float = _BASE_BACKOFF,
    ) -> None:
        self._sender = sender
        self._max_retries = max_retries
        self._queue: asyncio.Queue[NotificationItem] = asyncio.Queue(maxsize=queue_size)
        self._spam_guard = SpamGuard(cooldown_seconds=cooldown_seconds)
        self._base_backoff = base_backoff
        self._running = False

    async def enqueue(self, item: NotificationItem) -> None:
        """알림을 큐에 추가 (non-blocking, full이면 drop).

        Args:
            item: 전송할 알림 아이템
        """
        # SpamGuard 체크
        if item.spam_key is not None and not self._spam_guard.should_send(item.spam_key):
            return

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("Notification queue full, dropping: {}", item.channel)

    async def start(self) -> None:
        """Worker 루프 시작. stop() 호출 전까지 큐에서 아이템을 꺼내 전송."""
        self._running = True
        logger.info("NotificationQueue worker started")
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            await self._send_with_retry(item)
            self._queue.task_done()

        # drain 잔여 아이템
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                await self._send_with_retry(item)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        logger.info("NotificationQueue worker stopped")

    async def stop(self) -> None:
        """Worker 루프 중지 요청."""
        self._running = False

    async def _send_with_retry(self, item: NotificationItem) -> None:
        """지수 백오프로 재시도하며 전송.

        Args:
            item: 전송할 알림 아이템
        """
        for attempt in range(self._max_retries):
            try:
                success = await self._sender.send_embed(item.channel, item.embed)
                if success:
                    return
                logger.warning(
                    "Notification send returned False (attempt {}/{})",
                    attempt + 1,
                    self._max_retries,
                )
            except Exception:
                logger.exception(
                    "Notification send error (attempt {}/{})",
                    attempt + 1,
                    self._max_retries,
                )

            if attempt < self._max_retries - 1:
                backoff = self._base_backoff * (2**attempt)
                await asyncio.sleep(backoff)

        logger.error("Notification dropped after {} retries: {}", self._max_retries, item.channel)
