"""YAML-based Audit data store.

audits/ 디렉토리에 감사 데이터를 저장/로드:
- Snapshots: audits/snapshots/{date}.yaml
- Findings: audits/findings/{id:03d}.yaml
- Actions: audits/actions/{id:03d}.yaml
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.audit_models import (
    ActionItem,
    ActionPriority,
    ActionStatus,
    AuditCategory,
    AuditSeverity,
    AuditSnapshot,
    Finding,
    FindingStatus,
)

_DEFAULT_BASE_DIR = Path("audits")


class AuditStore:
    """YAML 기반 감사 데이터 저장소."""

    def __init__(self, base_dir: Path = _DEFAULT_BASE_DIR) -> None:
        self._base_dir = base_dir
        self._snapshot_dir = base_dir / "snapshots"
        self._finding_dir = base_dir / "findings"
        self._action_dir = base_dir / "actions"

        self._snapshot_cache: dict[str, AuditSnapshot] = {}
        self._finding_cache: dict[int, Finding] = {}
        self._action_cache: dict[int, ActionItem] = {}

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ─── Snapshot CRUD ────────────────────────────────────────────────

    def load_snapshot(self, date_str: str) -> AuditSnapshot:
        """날짜 기반 스냅샷 로드."""
        if date_str in self._snapshot_cache:
            return self._snapshot_cache[date_str]

        path = self._snapshot_dir / f"{date_str}.yaml"
        if not path.exists():
            msg = f"Audit snapshot not found: {path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        snapshot = _deserialize_snapshot(raw)
        self._snapshot_cache[date_str] = snapshot
        return snapshot

    def save_snapshot(self, snapshot: AuditSnapshot) -> None:
        """스냅샷 YAML 저장."""
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        date_str = snapshot.date.isoformat()
        path = self._snapshot_dir / f"{date_str}.yaml"
        data = _serialize(snapshot)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._snapshot_cache[date_str] = snapshot

    def load_all_snapshots(self) -> list[AuditSnapshot]:
        """모든 스냅샷 로드 (날짜순 정렬)."""
        if not self._snapshot_dir.exists():
            return []
        snapshots: list[AuditSnapshot] = []
        for path in sorted(self._snapshot_dir.glob("*.yaml")):
            date_str = path.stem
            snapshots.append(self.load_snapshot(date_str))
        return snapshots

    def latest_snapshot(self) -> AuditSnapshot | None:
        """최신 스냅샷 반환."""
        snapshots = self.load_all_snapshots()
        return snapshots[-1] if snapshots else None

    # ─── Finding CRUD ─────────────────────────────────────────────────

    def load_finding(self, finding_id: int) -> Finding:
        """Finding ID로 로드."""
        if finding_id in self._finding_cache:
            return self._finding_cache[finding_id]

        path = self._finding_dir / f"{finding_id:03d}.yaml"
        if not path.exists():
            msg = f"Finding not found: {path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        finding = Finding(**raw)
        self._finding_cache[finding_id] = finding
        return finding

    def save_finding(self, finding: Finding) -> None:
        """Finding YAML 저장."""
        self._finding_dir.mkdir(parents=True, exist_ok=True)
        path = self._finding_dir / f"{finding.id:03d}.yaml"
        data = _serialize(finding)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._finding_cache[finding.id] = finding

    def load_all_findings(self) -> list[Finding]:
        """모든 발견사항 로드 (ID순 정렬)."""
        if not self._finding_dir.exists():
            return []
        findings: list[Finding] = []
        for path in sorted(self._finding_dir.glob("*.yaml")):
            try:
                finding_id = int(path.stem)
            except ValueError:
                continue
            findings.append(self.load_finding(finding_id))
        return findings

    def next_finding_id(self) -> int:
        """다음 Finding ID."""
        findings = self.load_all_findings()
        if not findings:
            return 1
        return max(f.id for f in findings) + 1

    # ─── ActionItem CRUD ──────────────────────────────────────────────

    def load_action(self, action_id: int) -> ActionItem:
        """ActionItem ID로 로드."""
        if action_id in self._action_cache:
            return self._action_cache[action_id]

        path = self._action_dir / f"{action_id:03d}.yaml"
        if not path.exists():
            msg = f"ActionItem not found: {path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        action = ActionItem(**raw)
        self._action_cache[action_id] = action
        return action

    def save_action(self, action: ActionItem) -> None:
        """ActionItem YAML 저장."""
        self._action_dir.mkdir(parents=True, exist_ok=True)
        path = self._action_dir / f"{action.id:03d}.yaml"
        data = _serialize(action)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._action_cache[action.id] = action

    def load_all_actions(self) -> list[ActionItem]:
        """모든 액션 아이템 로드 (ID순 정렬)."""
        if not self._action_dir.exists():
            return []
        actions: list[ActionItem] = []
        for path in sorted(self._action_dir.glob("*.yaml")):
            try:
                action_id = int(path.stem)
            except ValueError:
                continue
            actions.append(self.load_action(action_id))
        return actions

    def next_action_id(self) -> int:
        """다음 ActionItem ID."""
        actions = self.load_all_actions()
        if not actions:
            return 1
        return max(a.id for a in actions) + 1

    # ─── Query ────────────────────────────────────────────────────────

    def open_findings(self) -> list[Finding]:
        """미해결 발견사항."""
        return [f for f in self.load_all_findings() if f.status == FindingStatus.OPEN]

    def findings_by_severity(self, severity: AuditSeverity) -> list[Finding]:
        """심각도별 필터링."""
        return [f for f in self.load_all_findings() if f.severity == severity]

    def findings_by_category(self, category: AuditCategory) -> list[Finding]:
        """카테고리별 필터링."""
        return [f for f in self.load_all_findings() if f.category == category]

    def findings_by_status(self, status: FindingStatus) -> list[Finding]:
        """상태별 필터링."""
        return [f for f in self.load_all_findings() if f.status == status]

    def pending_actions(self) -> list[ActionItem]:
        """대기 중 액션."""
        return [a for a in self.load_all_actions() if a.status == ActionStatus.PENDING]

    def actions_by_priority(self, priority: ActionPriority) -> list[ActionItem]:
        """우선순위별 필터링."""
        return [a for a in self.load_all_actions() if a.priority == priority]

    # ─── 상태 변경 ────────────────────────────────────────────────────

    def resolve_finding(self, finding_id: int, resolved_date: date | None = None) -> None:
        """발견사항 해결 처리."""
        finding = self.load_finding(finding_id)
        resolved = resolved_date or date.today()
        updated = finding.model_copy(
            update={"status": FindingStatus.RESOLVED, "resolved_at": resolved}
        )
        self._finding_cache.pop(finding_id, None)
        self.save_finding(updated)

    def update_action_status(
        self,
        action_id: int,
        status: ActionStatus,
        *,
        completed_date: date | None = None,
        started_date: date | None = None,
    ) -> None:
        """액션 상태 변경."""
        action = self.load_action(action_id)
        updates: dict[str, Any] = {"status": status}
        if status == ActionStatus.IN_PROGRESS and action.started_at is None:
            updates["started_at"] = started_date or date.today()
        if status == ActionStatus.COMPLETED:
            updates["completed_at"] = completed_date or date.today()
        updated = action.model_copy(update=updates)
        self._action_cache.pop(action_id, None)
        self.save_action(updated)


# ─── Serialization helpers ───────────────────────────────────────────


def _serialize(model: AuditSnapshot | Finding | ActionItem) -> dict[str, Any]:
    """Pydantic model → YAML-safe dict."""
    return model.model_dump(mode="json")


def _deserialize_snapshot(raw: dict[str, Any]) -> AuditSnapshot:
    """YAML dict → AuditSnapshot."""
    return AuditSnapshot(**raw)
