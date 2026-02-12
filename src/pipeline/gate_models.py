"""Gate criteria data models.

Gate 평가 기준을 구조화된 데이터로 관리:
- GateType: Gate 유형 (SCORING, CHECKLIST, THRESHOLD)
- Severity: 체크리스트 항목 심각도
- GateCriteria: Gate별 평가 기준 (composition 패턴)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ─── Enums ───────────────────────────────────────────────────────────


class GateType(StrEnum):
    """Gate 평가 유형."""

    SCORING = "scoring"
    CHECKLIST = "checklist"
    THRESHOLD = "threshold"


class Severity(StrEnum):
    """체크리스트 항목 심각도."""

    CRITICAL = "critical"
    WARNING = "warning"


# ─── Scoring (G0A) ──────────────────────────────────────────────────


class ScoringItem(BaseModel):
    """채점 항목 (1~5점)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="항목명 (예: 경제적 논거)")
    description: str = Field(description="채점 기준 설명")
    min_score: int = Field(default=1, description="최소 점수")
    max_score: int = Field(default=5, description="최대 점수")


class ScoringCriteria(BaseModel):
    """채점 기반 통과 기준 (G0A)."""

    model_config = ConfigDict(frozen=True)

    items: list[ScoringItem] = Field(description="채점 항목 목록")
    pass_threshold: int = Field(default=18, description="PASS 기준 합계 점수")
    max_total: int = Field(default=30, description="만점")


# ─── Checklist (G0B) ────────────────────────────────────────────────


class ChecklistItem(BaseModel):
    """체크리스트 항목."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="항목 코드 (C1~C7, W1~W5)")
    name: str = Field(description="항목명")
    description: str = Field(description="검증 내용")
    severity: Severity = Field(description="심각도")


class ChecklistCriteria(BaseModel):
    """체크리스트 기반 통과 기준 (G0B)."""

    model_config = ConfigDict(frozen=True)

    items: list[ChecklistItem] = Field(description="체크리스트 항목 목록")
    pass_rule: str = Field(description="PASS 조건 설명")


# ─── Threshold (G1~G7) ──────────────────────────────────────────────


class ThresholdMetric(BaseModel):
    """임계값 기반 통과 지표."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="지표명 (Sharpe, CAGR, ...)")
    operator: str = Field(description="비교 연산자 (>, >=, <, <=)")
    value: float = Field(description="임계값")
    unit: str = Field(default="", description="단위 (%, x, ...)")
    description: str = Field(default="", description="설명")


class ImmediateFailRule(BaseModel):
    """즉시 폐기 규칙."""

    model_config = ConfigDict(frozen=True)

    condition: str = Field(description="조건 (예: MDD > 50%)")
    reason: str = Field(description="사유")


class ThresholdCriteria(BaseModel):
    """임계값 기반 통과 기준 (G1~G7)."""

    model_config = ConfigDict(frozen=True)

    pass_metrics: list[ThresholdMetric] = Field(description="PASS 필수 조건")
    auxiliary_metrics: list[ThresholdMetric] = Field(
        default_factory=list, description="보조 지표 (판정 비영향)"
    )
    immediate_fail: list[ImmediateFailRule] = Field(
        default_factory=list, description="즉시 폐기 규칙"
    )
    watch_rule: str = Field(default="", description="WATCH 조건 설명")


# ─── Gate Criteria (Composition) ────────────────────────────────────


class GateCriteria(BaseModel):
    """Gate별 평가 기준 — composition 패턴 (상속 대신)."""

    model_config = ConfigDict(frozen=True)

    gate_id: str = Field(description="Gate 식별자 (G0A, G0B, G1, ...)")
    name: str = Field(description="Gate 이름")
    description: str = Field(default="", description="Gate 설명")
    gate_type: GateType = Field(description="평가 유형")
    cli_command: str = Field(default="", description="CLI 실행 명령")

    scoring: ScoringCriteria | None = Field(default=None, description="채점 기준 (gate_type=scoring)")
    checklist: ChecklistCriteria | None = Field(
        default=None, description="체크리스트 기준 (gate_type=checklist)"
    )
    threshold: ThresholdCriteria | None = Field(
        default=None, description="임계값 기준 (gate_type=threshold)"
    )
