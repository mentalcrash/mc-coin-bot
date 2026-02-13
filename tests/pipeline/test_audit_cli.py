"""Tests for src/cli/audit.py — audit CLI commands."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from src.cli.audit import app
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

runner = CliRunner()


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    """Temporary audits directory."""
    d = tmp_path / "audits"
    d.mkdir()
    return d


@pytest.fixture
def _patch_store(audit_dir: Path) -> None:
    """Patch AuditStore to use tmp_path."""
    original_init = AuditStore.__init__

    def patched_init(self: AuditStore, base_dir: Path = audit_dir) -> None:
        original_init(self, base_dir=base_dir)

    with patch.object(AuditStore, "__init__", patched_init):
        yield


@pytest.fixture
def populated_store(audit_dir: Path) -> AuditStore:
    """Store with sample data."""
    store = AuditStore(base_dir=audit_dir)

    snapshot = AuditSnapshot(
        date=date(2026, 2, 13),
        git_sha="c833ff2",
        scope=[AuditCategory.ARCHITECTURE],
        metrics=MetricsSnapshot(test_count=3072, coverage_pct=0.78),
        module_health=[
            ModuleHealth(module="src/eda", health=HealthLevel.GREEN, coverage_pct=0.89),
        ],
        strategy_summary=StrategySummary(total=50, active=5),
        new_findings=[1],
        new_actions=[1],
        grades=AuditGrades(overall="B"),
        summary="Test snapshot summary",
    )
    store.save_snapshot(snapshot)

    finding = Finding(
        id=1,
        title="Critical finding",
        severity=AuditSeverity.CRITICAL,
        category=AuditCategory.RISK_SAFETY,
        status=FindingStatus.OPEN,
        location="src/eda/oms.py:56",
        description="Test description",
        discovered_at=date(2026, 2, 13),
        tags=["live-trading"],
    )
    store.save_finding(finding)

    action = ActionItem(
        id=1,
        title="Fix critical issue",
        priority=ActionPriority.P0,
        status=ActionStatus.PENDING,
        phase="A",
        related_findings=[1],
        estimated_effort="2h",
        created_at=date(2026, 2, 13),
    )
    store.save_action(action)

    return store


# ─── List / Show / Latest ────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestSnapshotCommands:
    def test_list_empty(self) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No audit snapshots" in result.output

    def test_list_with_data(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "2026-02-13" in result.output
        assert "3072" in result.output

    def test_show(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["show", "2026-02-13"])
        assert result.exit_code == 0
        assert "c833ff2" in result.output
        assert "src/eda" in result.output

    def test_show_not_found(self) -> None:
        result = runner.invoke(app, ["show", "2099-01-01"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_latest_empty(self) -> None:
        result = runner.invoke(app, ["latest"])
        assert result.exit_code == 0
        assert "No audit snapshots" in result.output

    def test_latest_with_data(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["latest"])
        assert result.exit_code == 0
        assert "2026-02-13" in result.output


# ─── Findings Commands ────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestFindingCommands:
    def test_findings_empty(self) -> None:
        result = runner.invoke(app, ["findings"])
        assert result.exit_code == 0
        assert "No findings" in result.output

    def test_findings_all(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["findings"])
        assert result.exit_code == 0
        assert "Critical finding" in result.output

    def test_findings_by_status(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["findings", "--status", "open"])
        assert result.exit_code == 0
        assert "Critical finding" in result.output

    def test_findings_by_severity(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["findings", "--severity", "critical"])
        assert result.exit_code == 0
        assert "Critical finding" in result.output

    def test_findings_by_severity_no_match(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["findings", "--severity", "low"])
        assert result.exit_code == 0
        assert "No findings" in result.output

    def test_findings_by_category(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["findings", "--category", "risk-safety"])
        assert result.exit_code == 0
        assert "Critical finding" in result.output

    def test_findings_invalid_status(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["findings", "--status", "invalid"])
        assert result.exit_code == 1

    def test_finding_show(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["finding-show", "1"])
        assert result.exit_code == 0
        assert "Critical finding" in result.output
        assert "src/eda/oms.py:56" in result.output

    def test_finding_show_not_found(self) -> None:
        result = runner.invoke(app, ["finding-show", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output


# ─── Action Commands ──────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestActionCommands:
    def test_actions_empty(self) -> None:
        result = runner.invoke(app, ["actions"])
        assert result.exit_code == 0
        assert "No actions" in result.output

    def test_actions_all(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["actions"])
        assert result.exit_code == 0
        assert "Fix critical issue" in result.output

    def test_actions_by_status(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["actions", "--status", "pending"])
        assert result.exit_code == 0
        assert "Fix critical issue" in result.output

    def test_actions_by_priority(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["actions", "--priority", "P0"])
        assert result.exit_code == 0
        assert "Fix critical issue" in result.output

    def test_actions_by_priority_no_match(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["actions", "--priority", "P3"])
        assert result.exit_code == 0
        assert "No actions" in result.output

    def test_action_show(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["action-show", "1"])
        assert result.exit_code == 0
        assert "Fix critical issue" in result.output
        assert "P0" in result.output

    def test_action_show_not_found(self) -> None:
        result = runner.invoke(app, ["action-show", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output


# ─── Trend Command ───────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestTrendCommand:
    def test_trend_empty(self) -> None:
        result = runner.invoke(app, ["trend"])
        assert result.exit_code == 0
        assert "No audit snapshots" in result.output

    def test_trend_with_data(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["trend"])
        assert result.exit_code == 0
        assert "Audit Trend" in result.output
        # Summary line at bottom always visible
        assert "Open findings" in result.output


# ─── Status Change Commands ──────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestStatusChangeCommands:
    def test_resolve_finding(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["resolve-finding", "1"])
        assert result.exit_code == 0
        assert "resolved" in result.output

        # Clear cache to re-read updated YAML
        populated_store._finding_cache.clear()
        f = populated_store.load_finding(1)
        assert f.status == FindingStatus.RESOLVED

    def test_resolve_finding_not_found(self) -> None:
        result = runner.invoke(app, ["resolve-finding", "999"])
        assert result.exit_code == 1

    def test_update_action(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["update-action", "1", "--status", "completed"])
        assert result.exit_code == 0
        assert "completed" in result.output

        # Clear cache to re-read updated YAML
        populated_store._action_cache.clear()
        a = populated_store.load_action(1)
        assert a.status == ActionStatus.COMPLETED

    def test_update_action_not_found(self) -> None:
        result = runner.invoke(app, ["update-action", "999", "--status", "completed"])
        assert result.exit_code == 1

    def test_update_action_invalid_status(self, populated_store: AuditStore) -> None:
        result = runner.invoke(app, ["update-action", "1", "--status", "invalid"])
        assert result.exit_code == 1
