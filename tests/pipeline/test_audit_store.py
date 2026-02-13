"""Tests for src/pipeline/audit_store.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

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
from src.pipeline.audit_store import AuditStore


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    return AuditStore(base_dir=tmp_path)


@pytest.fixture
def sample_snapshot() -> AuditSnapshot:
    return AuditSnapshot(
        date=date(2026, 2, 13),
        git_sha="c833ff2",
        auditor="claude",
        scope=[AuditCategory.ARCHITECTURE, AuditCategory.RISK_SAFETY],
        metrics=MetricsSnapshot(test_count=3072, coverage_pct=0.78),
        module_health=[
            ModuleHealth(module="src/eda", health=HealthLevel.GREEN, coverage_pct=0.89),
        ],
        strategy_summary=StrategySummary(total=50, active=5, retired=25),
        new_findings=[1, 2],
        new_actions=[1],
        grades=AuditGrades(overall="B"),
        summary="Test snapshot",
    )


@pytest.fixture
def sample_finding() -> Finding:
    return Finding(
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


@pytest.fixture
def sample_finding_low() -> Finding:
    return Finding(
        id=2,
        title="Minor issue",
        severity=AuditSeverity.LOW,
        category=AuditCategory.CODE_QUALITY,
        status=FindingStatus.OPEN,
        discovered_at=date(2026, 2, 13),
        tags=["minor"],
    )


@pytest.fixture
def sample_action() -> ActionItem:
    return ActionItem(
        id=1,
        title="OMS 영속화",
        priority=ActionPriority.P0,
        status=ActionStatus.PENDING,
        phase="A",
        related_findings=[1],
        description="SQLite 영속화",
        estimated_effort="2h",
        created_at=date(2026, 2, 13),
        tags=["live-trading"],
    )


@pytest.fixture
def sample_action_p2() -> ActionItem:
    return ActionItem(
        id=2,
        title="Minor fix",
        priority=ActionPriority.P2,
        status=ActionStatus.PENDING,
        phase="C",
        created_at=date(2026, 2, 13),
    )


# ─── Snapshot Tests ───────────────────────────────────────────────────


class TestSnapshotCRUD:
    def test_save_and_load(self, store: AuditStore, sample_snapshot: AuditSnapshot) -> None:
        store.save_snapshot(sample_snapshot)
        loaded = store.load_snapshot("2026-02-13")
        assert loaded == sample_snapshot

    def test_load_nonexistent(self, store: AuditStore) -> None:
        with pytest.raises(FileNotFoundError, match="Audit snapshot not found"):
            store.load_snapshot("2099-01-01")

    def test_load_all_empty(self, store: AuditStore) -> None:
        assert store.load_all_snapshots() == []

    def test_load_all_sorted(self, store: AuditStore) -> None:
        s1 = AuditSnapshot(date=date(2026, 3, 1))
        s2 = AuditSnapshot(date=date(2026, 2, 13))
        store.save_snapshot(s1)
        store.save_snapshot(s2)
        snapshots = store.load_all_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0].date == date(2026, 2, 13)
        assert snapshots[1].date == date(2026, 3, 1)

    def test_latest_snapshot_empty(self, store: AuditStore) -> None:
        assert store.latest_snapshot() is None

    def test_latest_snapshot(self, store: AuditStore) -> None:
        s1 = AuditSnapshot(date=date(2026, 2, 13))
        s2 = AuditSnapshot(date=date(2026, 3, 1))
        store.save_snapshot(s1)
        store.save_snapshot(s2)
        latest = store.latest_snapshot()
        assert latest is not None
        assert latest.date == date(2026, 3, 1)

    def test_roundtrip_yaml(self, store: AuditStore, sample_snapshot: AuditSnapshot) -> None:
        store.save_snapshot(sample_snapshot)
        store._snapshot_cache.clear()
        loaded = store.load_snapshot("2026-02-13")
        assert loaded == sample_snapshot

    def test_cache_hit(self, store: AuditStore, sample_snapshot: AuditSnapshot) -> None:
        store.save_snapshot(sample_snapshot)
        loaded1 = store.load_snapshot("2026-02-13")
        loaded2 = store.load_snapshot("2026-02-13")
        assert loaded1 is loaded2

    def test_file_naming(self, store: AuditStore, sample_snapshot: AuditSnapshot) -> None:
        store.save_snapshot(sample_snapshot)
        assert (store.base_dir / "snapshots" / "2026-02-13.yaml").exists()


# ─── Finding Tests ────────────────────────────────────────────────────


class TestFindingCRUD:
    def test_save_and_load(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        loaded = store.load_finding(1)
        assert loaded == sample_finding

    def test_load_nonexistent(self, store: AuditStore) -> None:
        with pytest.raises(FileNotFoundError, match="Finding not found"):
            store.load_finding(999)

    def test_load_all_empty(self, store: AuditStore) -> None:
        assert store.load_all_findings() == []

    def test_load_all_sorted(
        self, store: AuditStore, sample_finding: Finding, sample_finding_low: Finding
    ) -> None:
        store.save_finding(sample_finding_low)
        store.save_finding(sample_finding)
        findings = store.load_all_findings()
        assert len(findings) == 2
        assert findings[0].id == 1
        assert findings[1].id == 2

    def test_next_finding_id_empty(self, store: AuditStore) -> None:
        assert store.next_finding_id() == 1

    def test_next_finding_id(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        assert store.next_finding_id() == 2

    def test_file_naming(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        assert (store.base_dir / "findings" / "001.yaml").exists()

    def test_roundtrip_yaml(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        store._finding_cache.clear()
        loaded = store.load_finding(1)
        assert loaded == sample_finding

    def test_cache_hit(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        loaded1 = store.load_finding(1)
        loaded2 = store.load_finding(1)
        assert loaded1 is loaded2

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        store = AuditStore(base_dir=tmp_path / "nonexistent")
        assert store.load_all_findings() == []


# ─── Action Tests ─────────────────────────────────────────────────────


class TestActionCRUD:
    def test_save_and_load(self, store: AuditStore, sample_action: ActionItem) -> None:
        store.save_action(sample_action)
        loaded = store.load_action(1)
        assert loaded == sample_action

    def test_load_nonexistent(self, store: AuditStore) -> None:
        with pytest.raises(FileNotFoundError, match="ActionItem not found"):
            store.load_action(999)

    def test_load_all_empty(self, store: AuditStore) -> None:
        assert store.load_all_actions() == []

    def test_load_all_sorted(
        self, store: AuditStore, sample_action: ActionItem, sample_action_p2: ActionItem
    ) -> None:
        store.save_action(sample_action_p2)
        store.save_action(sample_action)
        actions = store.load_all_actions()
        assert len(actions) == 2
        assert actions[0].id == 1
        assert actions[1].id == 2

    def test_next_action_id_empty(self, store: AuditStore) -> None:
        assert store.next_action_id() == 1

    def test_next_action_id(self, store: AuditStore, sample_action: ActionItem) -> None:
        store.save_action(sample_action)
        assert store.next_action_id() == 2

    def test_file_naming(self, store: AuditStore, sample_action: ActionItem) -> None:
        store.save_action(sample_action)
        assert (store.base_dir / "actions" / "001.yaml").exists()

    def test_roundtrip_yaml(self, store: AuditStore, sample_action: ActionItem) -> None:
        store.save_action(sample_action)
        store._action_cache.clear()
        loaded = store.load_action(1)
        assert loaded == sample_action

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        store = AuditStore(base_dir=tmp_path / "nonexistent")
        assert store.load_all_actions() == []


# ─── Query Tests ──────────────────────────────────────────────────────


class TestQuery:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        store: AuditStore,
        sample_finding: Finding,
        sample_finding_low: Finding,
        sample_action: ActionItem,
        sample_action_p2: ActionItem,
    ) -> None:
        store.save_finding(sample_finding)
        store.save_finding(sample_finding_low)
        store.save_action(sample_action)
        store.save_action(sample_action_p2)

    def test_open_findings(self, store: AuditStore) -> None:
        results = store.open_findings()
        assert len(results) == 2

    def test_findings_by_severity(self, store: AuditStore) -> None:
        results = store.findings_by_severity(AuditSeverity.CRITICAL)
        assert len(results) == 1
        assert results[0].id == 1

    def test_findings_by_severity_no_match(self, store: AuditStore) -> None:
        results = store.findings_by_severity(AuditSeverity.MEDIUM)
        assert results == []

    def test_findings_by_category(self, store: AuditStore) -> None:
        results = store.findings_by_category(AuditCategory.RISK_SAFETY)
        assert len(results) == 1
        assert results[0].id == 1

    def test_findings_by_status(self, store: AuditStore) -> None:
        results = store.findings_by_status(FindingStatus.OPEN)
        assert len(results) == 2

    def test_findings_by_status_no_match(self, store: AuditStore) -> None:
        results = store.findings_by_status(FindingStatus.RESOLVED)
        assert results == []

    def test_pending_actions(self, store: AuditStore) -> None:
        results = store.pending_actions()
        assert len(results) == 2

    def test_actions_by_priority(self, store: AuditStore) -> None:
        results = store.actions_by_priority(ActionPriority.P0)
        assert len(results) == 1
        assert results[0].id == 1

    def test_actions_by_priority_no_match(self, store: AuditStore) -> None:
        results = store.actions_by_priority(ActionPriority.P3)
        assert results == []


# ─── Status Change Tests ─────────────────────────────────────────────


class TestStatusChange:
    def test_resolve_finding(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        store.resolve_finding(1, resolved_date=date(2026, 2, 14))
        resolved = store.load_finding(1)
        assert resolved.status == FindingStatus.RESOLVED
        assert resolved.resolved_at == date(2026, 2, 14)

    def test_resolve_finding_default_date(self, store: AuditStore, sample_finding: Finding) -> None:
        store.save_finding(sample_finding)
        store.resolve_finding(1)
        resolved = store.load_finding(1)
        assert resolved.status == FindingStatus.RESOLVED
        assert resolved.resolved_at is not None

    def test_resolve_finding_nonexistent(self, store: AuditStore) -> None:
        with pytest.raises(FileNotFoundError):
            store.resolve_finding(999)

    def test_update_action_status_completed(
        self, store: AuditStore, sample_action: ActionItem
    ) -> None:
        store.save_action(sample_action)
        store.update_action_status(1, ActionStatus.COMPLETED, completed_date=date(2026, 2, 15))
        updated = store.load_action(1)
        assert updated.status == ActionStatus.COMPLETED
        assert updated.completed_at == date(2026, 2, 15)

    def test_update_action_status_in_progress(
        self, store: AuditStore, sample_action: ActionItem
    ) -> None:
        store.save_action(sample_action)
        store.update_action_status(1, ActionStatus.IN_PROGRESS, started_date=date(2026, 2, 14))
        updated = store.load_action(1)
        assert updated.status == ActionStatus.IN_PROGRESS
        assert updated.started_at == date(2026, 2, 14)

    def test_update_action_status_nonexistent(self, store: AuditStore) -> None:
        with pytest.raises(FileNotFoundError):
            store.update_action_status(999, ActionStatus.COMPLETED)

    def test_update_action_preserves_started_at(
        self, store: AuditStore, sample_action: ActionItem
    ) -> None:
        store.save_action(sample_action)
        store.update_action_status(1, ActionStatus.IN_PROGRESS, started_date=date(2026, 2, 14))
        # Complete — started_at should be preserved
        store.update_action_status(1, ActionStatus.COMPLETED, completed_date=date(2026, 2, 15))
        updated = store.load_action(1)
        assert updated.started_at == date(2026, 2, 14)
        assert updated.completed_at == date(2026, 2, 15)

    def test_update_action_cancelled(self, store: AuditStore, sample_action: ActionItem) -> None:
        store.save_action(sample_action)
        store.update_action_status(1, ActionStatus.CANCELLED)
        updated = store.load_action(1)
        assert updated.status == ActionStatus.CANCELLED
