"""Tests for src/pipeline/audit_models.py."""

from __future__ import annotations

from datetime import date

import pytest

from src.pipeline.audit_models import (
    ActionItem,
    ActionPriority,
    ActionStatus,
    AuditCategory,
    AuditGrades,
    AuditSeverity,
    AuditSnapshot,
    Finding,
    FindingStatus,
    HealthLevel,
    MetricsSnapshot,
    ModuleHealth,
    StrategySummary,
)


class TestEnums:
    def test_severity_values(self) -> None:
        assert AuditSeverity.CRITICAL == "critical"
        assert AuditSeverity.HIGH == "high"
        assert AuditSeverity.MEDIUM == "medium"
        assert AuditSeverity.LOW == "low"
        assert len(AuditSeverity) == 4

    def test_category_values(self) -> None:
        assert AuditCategory.ARCHITECTURE == "architecture"
        assert AuditCategory.RISK_SAFETY == "risk-safety"
        assert AuditCategory.CODE_QUALITY == "code-quality"
        assert AuditCategory.DATA_PIPELINE == "data-pipeline"
        assert AuditCategory.TESTING_OPS == "testing-ops"
        assert AuditCategory.PERFORMANCE == "performance"
        assert len(AuditCategory) == 6

    def test_finding_status_values(self) -> None:
        assert FindingStatus.OPEN == "open"
        assert FindingStatus.IN_PROGRESS == "in_progress"
        assert FindingStatus.RESOLVED == "resolved"
        assert FindingStatus.WONT_FIX == "wont_fix"
        assert FindingStatus.DEFERRED == "deferred"
        assert len(FindingStatus) == 5

    def test_action_priority_values(self) -> None:
        assert ActionPriority.P0 == "P0"
        assert ActionPriority.P1 == "P1"
        assert ActionPriority.P2 == "P2"
        assert ActionPriority.P3 == "P3"
        assert len(ActionPriority) == 4

    def test_action_status_values(self) -> None:
        assert ActionStatus.PENDING == "pending"
        assert ActionStatus.IN_PROGRESS == "in_progress"
        assert ActionStatus.COMPLETED == "completed"
        assert ActionStatus.CANCELLED == "cancelled"
        assert len(ActionStatus) == 4

    def test_health_level_values(self) -> None:
        assert HealthLevel.GREEN == "green"
        assert HealthLevel.YELLOW == "yellow"
        assert HealthLevel.RED == "red"
        assert len(HealthLevel) == 3


class TestMetricsSnapshot:
    def test_defaults(self) -> None:
        m = MetricsSnapshot()
        assert m.test_count == 0
        assert m.test_pass_rate == 1.0
        assert m.lint_errors == 0
        assert m.type_errors == 0
        assert m.coverage_pct == 0.0

    def test_custom_values(self) -> None:
        m = MetricsSnapshot(test_count=3072, coverage_pct=0.78)
        assert m.test_count == 3072
        assert m.coverage_pct == 0.78

    def test_frozen(self) -> None:
        m = MetricsSnapshot()
        with pytest.raises(Exception):  # noqa: B017
            m.test_count = 100  # type: ignore[misc]


class TestModuleHealth:
    def test_creation(self) -> None:
        mh = ModuleHealth(module="src/eda", health=HealthLevel.GREEN, coverage_pct=0.89)
        assert mh.module == "src/eda"
        assert mh.health == HealthLevel.GREEN
        assert mh.coverage_pct == 0.89

    def test_optional_coverage(self) -> None:
        mh = ModuleHealth(module="src/core", health=HealthLevel.YELLOW)
        assert mh.coverage_pct is None
        assert mh.notes == ""


class TestStrategySummary:
    def test_defaults(self) -> None:
        ss = StrategySummary()
        assert ss.total == 0
        assert ss.active == 0

    def test_custom(self) -> None:
        ss = StrategySummary(total=50, active=5, testing=8, candidate=12, retired=25)
        assert ss.total == 50
        assert ss.retired == 25


class TestFinding:
    def test_minimal_creation(self) -> None:
        f = Finding(
            id=1,
            title="Test finding",
            severity=AuditSeverity.CRITICAL,
            category=AuditCategory.RISK_SAFETY,
            discovered_at=date(2026, 2, 13),
        )
        assert f.id == 1
        assert f.status == FindingStatus.OPEN
        assert f.related_actions == []
        assert f.tags == []
        assert f.resolved_at is None

    def test_full_creation(self) -> None:
        f = Finding(
            id=1,
            title="OMS 주문 멱등성",
            severity=AuditSeverity.CRITICAL,
            category=AuditCategory.RISK_SAFETY,
            status=FindingStatus.OPEN,
            location="src/eda/oms.py:56",
            description="메모리 전용",
            impact="주문 중복",
            proposed_fix="SQLite 영속화",
            effort="2h",
            related_actions=[1],
            discovered_at=date(2026, 2, 13),
            tags=["live-trading", "persistence"],
        )
        assert f.location == "src/eda/oms.py:56"
        assert f.related_actions == [1]
        assert "live-trading" in f.tags

    def test_frozen(self) -> None:
        f = Finding(
            id=1,
            title="Test",
            severity=AuditSeverity.LOW,
            category=AuditCategory.CODE_QUALITY,
            discovered_at=date(2026, 2, 13),
        )
        with pytest.raises(Exception):  # noqa: B017
            f.title = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        f = Finding(
            id=1,
            title="Test",
            severity=AuditSeverity.HIGH,
            category=AuditCategory.ARCHITECTURE,
            discovered_at=date(2026, 2, 13),
            tags=["test"],
        )
        data = f.model_dump(mode="json")
        restored = Finding(**data)
        assert restored == f

    def test_model_dump_json(self) -> None:
        f = Finding(
            id=1,
            title="Test",
            severity=AuditSeverity.CRITICAL,
            category=AuditCategory.RISK_SAFETY,
            discovered_at=date(2026, 2, 13),
        )
        data = f.model_dump(mode="json")
        assert data["severity"] == "critical"
        assert data["category"] == "risk-safety"
        assert data["discovered_at"] == "2026-02-13"


class TestActionItem:
    def test_minimal_creation(self) -> None:
        a = ActionItem(
            id=1,
            title="Test action",
            priority=ActionPriority.P0,
            created_at=date(2026, 2, 13),
        )
        assert a.id == 1
        assert a.status == ActionStatus.PENDING
        assert a.assigned_to is None
        assert a.related_findings == []
        assert a.tags == []

    def test_full_creation(self) -> None:
        a = ActionItem(
            id=1,
            title="OMS 영속화",
            priority=ActionPriority.P0,
            status=ActionStatus.PENDING,
            phase="A",
            related_findings=[1],
            description="SQLite 영속화",
            estimated_effort="2h",
            created_at=date(2026, 2, 13),
            verification="pytest -k test_oms",
            tags=["live-trading"],
        )
        assert a.phase == "A"
        assert a.related_findings == [1]

    def test_frozen(self) -> None:
        a = ActionItem(
            id=1,
            title="Test",
            priority=ActionPriority.P1,
            created_at=date(2026, 2, 13),
        )
        with pytest.raises(Exception):  # noqa: B017
            a.title = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        a = ActionItem(
            id=1,
            title="Test",
            priority=ActionPriority.P2,
            created_at=date(2026, 2, 13),
            tags=["test"],
        )
        data = a.model_dump(mode="json")
        restored = ActionItem(**data)
        assert restored == a


class TestAuditSnapshot:
    def test_minimal_creation(self) -> None:
        s = AuditSnapshot(date=date(2026, 2, 13))
        assert s.date == date(2026, 2, 13)
        assert s.auditor == "claude"
        assert s.scope == []
        assert s.new_findings == []
        assert s.new_actions == []
        assert s.summary == ""

    def test_full_creation(self) -> None:
        s = AuditSnapshot(
            date=date(2026, 2, 13),
            git_sha="c833ff2",
            auditor="claude",
            scope=[AuditCategory.ARCHITECTURE, AuditCategory.RISK_SAFETY],
            metrics=MetricsSnapshot(test_count=3072, coverage_pct=0.78),
            module_health=[
                ModuleHealth(module="src/eda", health=HealthLevel.GREEN, coverage_pct=0.89),
            ],
            strategy_summary=StrategySummary(total=50, active=5),
            new_findings=[1, 2, 3],
            new_actions=[1, 2],
            grades=AuditGrades(overall="B"),
            summary="Test summary",
        )
        assert s.git_sha == "c833ff2"
        assert len(s.scope) == 2
        assert s.metrics.test_count == 3072
        assert len(s.module_health) == 1
        assert s.strategy_summary.total == 50
        assert s.new_findings == [1, 2, 3]
        assert s.grades.overall == "B"

    def test_frozen(self) -> None:
        s = AuditSnapshot(date=date(2026, 2, 13))
        with pytest.raises(Exception):  # noqa: B017
            s.auditor = "human"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        s = AuditSnapshot(
            date=date(2026, 2, 13),
            git_sha="abc123",
            scope=[AuditCategory.CODE_QUALITY],
            metrics=MetricsSnapshot(test_count=100),
            grades=AuditGrades(overall="A"),
        )
        data = s.model_dump(mode="json")
        restored = AuditSnapshot(**data)
        assert restored == s
