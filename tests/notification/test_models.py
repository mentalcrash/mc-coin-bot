"""NotificationItem, Severity, ChannelRoute 테스트."""

from __future__ import annotations

import pytest

from src.notification.models import ChannelRoute, NotificationItem, Severity


class TestSeverity:
    def test_values(self) -> None:
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.CRITICAL == "critical"
        assert Severity.EMERGENCY == "emergency"


class TestChannelRoute:
    def test_values(self) -> None:
        assert ChannelRoute.TRADE_LOG == "trade_log"
        assert ChannelRoute.ALERTS == "alerts"
        assert ChannelRoute.DAILY_REPORT == "daily_report"


class TestNotificationItem:
    def test_create(self) -> None:
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.TRADE_LOG,
            embed={"title": "Test", "color": 0x57F287},
        )
        assert item.severity == Severity.INFO
        assert item.channel == ChannelRoute.TRADE_LOG
        assert item.embed["title"] == "Test"
        assert item.spam_key is None

    def test_with_spam_key(self) -> None:
        item = NotificationItem(
            severity=Severity.WARNING,
            channel=ChannelRoute.ALERTS,
            embed={"title": "Alert"},
            spam_key="risk_alert:WARNING",
        )
        assert item.spam_key == "risk_alert:WARNING"

    def test_frozen(self) -> None:
        import pydantic

        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.TRADE_LOG,
            embed={"title": "Test"},
        )
        with pytest.raises(pydantic.ValidationError):
            item.severity = Severity.CRITICAL  # type: ignore[misc]

    def test_serialization(self) -> None:
        item = NotificationItem(
            severity=Severity.CRITICAL,
            channel=ChannelRoute.ALERTS,
            embed={"title": "CB", "color": 0xE67E22},
            spam_key="cb:test",
        )
        data = item.model_dump()
        assert data["severity"] == "critical"
        assert data["channel"] == "alerts"
        assert data["spam_key"] == "cb:test"
        restored = NotificationItem.model_validate(data)
        assert restored == item
