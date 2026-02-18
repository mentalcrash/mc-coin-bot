"""Failure Pattern Catalog models.

RETIRED 전략의 반복 실패 패턴을 구조화합니다.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Frequency(StrEnum):
    """실패 패턴 빈도."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DetectionRule(BaseModel):
    """실패 탐지 규칙."""

    model_config = ConfigDict(frozen=True)

    metric: str = Field(description="검사 대상 메트릭")
    operator: str = Field(description="비교 연산자 (>, <, >=, <=, ==)")
    threshold: float | str = Field(description="임계값")


class FailurePattern(BaseModel):
    """개별 실패 패턴."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="패턴 식별자")
    name: str = Field(description="패턴 이름")
    description: str = Field(description="패턴 설명")
    frequency: Frequency = Field(description="발생 빈도 (high/medium/low)")
    affected_phases: list[str] = Field(default_factory=list, description="영향받는 Phase")
    detection_rules: list[DetectionRule] = Field(default_factory=list, description="탐지 규칙")
    prevention: list[str] = Field(default_factory=list, description="예방 조치")
    related_lessons: list[int] = Field(default_factory=list, description="관련 Lesson ID")
    examples: list[str] = Field(default_factory=list, description="해당 전략 예시")


class FailurePatternCatalog(BaseModel):
    """실패 패턴 카탈로그."""

    model_config = ConfigDict(frozen=True)

    patterns: list[FailurePattern] = Field(default_factory=list)
