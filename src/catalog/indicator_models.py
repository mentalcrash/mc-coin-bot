"""Indicator Catalog models.

53개 지표의 메타데이터 + 전략 매핑 + 미사용/고가치 식별을 구조화합니다.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class IndicatorCategory(StrEnum):
    """지표 카테고리."""

    TREND = "trend"
    OSCILLATOR = "oscillator"
    VOLATILITY = "volatility"
    CHANNEL = "channel"
    RETURNS = "returns"
    VOLUME = "volume"
    DERIVATIVES = "derivatives"
    COMPOSITE = "composite"


class AlphaPotential(StrEnum):
    """Alpha 잠재력."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IndicatorEntry(BaseModel):
    """개별 지표 메타데이터."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="지표 식별자 (함수명과 1:1)")
    name: str = Field(description="표시 이름")
    module: str = Field(description="소속 서브모듈")
    category: IndicatorCategory = Field(description="지표 카테고리")
    description: str = Field(default="", description="지표 설명")
    default_params: dict[str, object] = Field(default_factory=dict, description="기본 파라미터")
    used_by: list[str] = Field(default_factory=list, description="사용 중인 전략 registry key")
    alpha_potential: AlphaPotential = Field(
        default=AlphaPotential.MEDIUM, description="Alpha 잠재력"
    )
    notes: str = Field(default="", description="비고")


class IndicatorCatalog(BaseModel):
    """전체 지표 카탈로그."""

    model_config = ConfigDict(frozen=True)

    indicators: list[IndicatorEntry] = Field(default_factory=list)
