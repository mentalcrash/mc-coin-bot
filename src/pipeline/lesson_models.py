"""Lesson data models.

전략 파이프라인에서 축적한 교훈을 구조화된 데이터로 관리:
- LessonCategory: 교훈 분류 (6종)
- LessonRecord: 개별 교훈 레코드 (YAML 1:1 매핑)
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LessonCategory(StrEnum):
    """교훈 분류."""

    STRATEGY_DESIGN = "strategy-design"
    RISK_MANAGEMENT = "risk-management"
    MARKET_STRUCTURE = "market-structure"
    DATA_RESOLUTION = "data-resolution"
    PIPELINE_PROCESS = "pipeline-process"
    META_ANALYSIS = "meta-analysis"


class LessonRecord(BaseModel):
    """개별 교훈 레코드 (YAML 1:1 매핑)."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(description="1-based ID, 파일명 매핑 (001.yaml)")
    title: str = Field(description="교훈 제목 (Bold 텍스트)")
    body: str = Field(description="상세 설명")
    category: LessonCategory = Field(description="6분류")
    tags: list[str] = Field(default_factory=list, description="검색용 자유 태그")
    strategies: list[str] = Field(default_factory=list, description="관련 전략 registry key")
    timeframes: list[str] = Field(default_factory=list, description="관련 TF")
    added_at: date = Field(description="추가일")
