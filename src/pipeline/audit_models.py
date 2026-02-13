"""Audit report data models.

Architecture audit 시스템의 핵심 모델:
- AuditSnapshot: 특정 시점의 프로젝트 건강 상태
- Finding: 개별 발견사항 (상태 추적)
- ActionItem: 액션 아이템 (생명주기 추적)
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ─── Enums ───────────────────────────────────────────────────────────


class AuditSeverity(StrEnum):
    """발견사항 심각도."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditCategory(StrEnum):
    """감사 카테고리."""

    ARCHITECTURE = "architecture"
    RISK_SAFETY = "risk-safety"
    CODE_QUALITY = "code-quality"
    DATA_PIPELINE = "data-pipeline"
    TESTING_OPS = "testing-ops"
    PERFORMANCE = "performance"


class FindingStatus(StrEnum):
    """발견사항 상태."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    DEFERRED = "deferred"


class ActionPriority(StrEnum):
    """액션 우선순위."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ActionStatus(StrEnum):
    """액션 상태."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class HealthLevel(StrEnum):
    """모듈 건강도."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# ─── Sub-Models ──────────────────────────────────────────────────────


class MetricsSnapshot(BaseModel):
    """정량 지표 스냅샷."""

    model_config = ConfigDict(frozen=True)

    test_count: int = 0
    test_pass_rate: float = 1.0
    lint_errors: int = 0
    type_errors: int = 0
    coverage_pct: float = 0.0


class ModuleHealth(BaseModel):
    """모듈별 건강도."""

    model_config = ConfigDict(frozen=True)

    module: str
    health: HealthLevel
    coverage_pct: float | None = None
    notes: str = ""


class StrategySummary(BaseModel):
    """전략 파이프라인 요약."""

    model_config = ConfigDict(frozen=True)

    total: int = 0
    active: int = 0
    testing: int = 0
    candidate: int = 0
    retired: int = 0


class AuditGrades(BaseModel):
    """등급 요약."""

    model_config = ConfigDict(frozen=True)

    architecture: str = ""
    risk_safety: str = ""
    code_quality: str = ""
    data_pipeline: str = ""
    testing_ops: str = ""
    overall: str = ""


# ─── Main Models ─────────────────────────────────────────────────────


class Finding(BaseModel):
    """개별 발견사항 (스냅샷과 독립적으로 상태 추적)."""

    model_config = ConfigDict(frozen=True)

    id: int
    title: str
    severity: AuditSeverity
    category: AuditCategory
    status: FindingStatus = FindingStatus.OPEN
    location: str = ""
    description: str = ""
    impact: str = ""
    proposed_fix: str = ""
    effort: str = ""
    related_actions: list[int] = Field(default_factory=list)
    discovered_at: date
    resolved_at: date | None = None
    tags: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    """액션 아이템 (발견사항 기반 개선 계획)."""

    model_config = ConfigDict(frozen=True)

    id: int
    title: str
    priority: ActionPriority
    status: ActionStatus = ActionStatus.PENDING
    phase: str = ""
    assigned_to: str | None = None
    related_findings: list[int] = Field(default_factory=list)
    description: str = ""
    estimated_effort: str = ""
    created_at: date
    started_at: date | None = None
    completed_at: date | None = None
    verification: str = ""
    tags: list[str] = Field(default_factory=list)


class AuditSnapshot(BaseModel):
    """감사 스냅샷 (특정 시점의 프로젝트 건강 상태)."""

    model_config = ConfigDict(frozen=True)

    date: date
    git_sha: str = ""
    auditor: str = "claude"
    scope: list[AuditCategory] = Field(default_factory=list)
    metrics: MetricsSnapshot = Field(default_factory=MetricsSnapshot)
    module_health: list[ModuleHealth] = Field(default_factory=list)
    strategy_summary: StrategySummary = Field(default_factory=StrategySummary)
    new_findings: list[int] = Field(default_factory=list)
    new_actions: list[int] = Field(default_factory=list)
    grades: AuditGrades = Field(default_factory=AuditGrades)
    summary: str = ""
