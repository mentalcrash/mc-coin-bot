"""Experiment Tracking models.

Phase 실행 시 에셋별 상세 결과를 기록하는 모델.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True
    - Pydantic V2: model_dump(mode="json") 지원
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AssetResult(BaseModel):
    """개별 에셋의 Phase 실행 결과."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    sharpe: float
    cagr: float
    mdd: float
    trades: int
    win_rate: float | None = None
    profit_factor: float | None = None


class ExperimentRecord(BaseModel):
    """Phase 실행의 전체 기록 (에셋별 결과 포함)."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    phase_id: str
    timestamp: datetime
    params: dict[str, Any]
    asset_results: list[AssetResult]
    passed: bool
    rationale: str = ""
    duration_seconds: float = 0.0


class ExperimentAnalysis(BaseModel):
    """전략별 실험 요약 분석."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    total_experiments: int
    pass_rate: float
    best_phase: str
    best_sharpe: float
    avg_mdd: float = Field(description="전체 에셋 평균 MDD")
