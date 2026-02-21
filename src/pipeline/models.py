"""Strategy Pipeline Pydantic models.

전략 수명주기 메타데이터를 관리하는 모델:
- StrategyMeta: 전략 기본 정보
- AssetMetrics: 에셋별 백테스트 성과
- PhaseResult: Phase 검증 결과
- Decision: 의사결정 기록
- StrategyRecord: 전략 전체 레코드 (YAML 1:1 매핑)
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

# ─── Enums ───────────────────────────────────────────────────────────


class StrategyStatus(StrEnum):
    """전략 수명주기 상태."""

    CANDIDATE = "CANDIDATE"
    IMPLEMENTED = "IMPLEMENTED"
    TESTING = "TESTING"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class PhaseId(StrEnum):
    """Phase 식별자 (v2)."""

    P1 = "P1"  # Alpha Research
    P2 = "P2"  # Implementation
    P3 = "P3"  # Code Audit
    P4 = "P4"  # Backtest
    P5 = "P5"  # Robustness
    P6 = "P6"  # Deep Validation
    P7 = "P7"  # Live Readiness


# Phase 순서 (진행도 계산용)
PHASE_ORDER: list[PhaseId] = [
    PhaseId.P1,
    PhaseId.P2,
    PhaseId.P3,
    PhaseId.P4,
    PhaseId.P5,
    PhaseId.P6,
    PhaseId.P7,
]


class PhaseVerdict(StrEnum):
    """Phase 판정 결과."""

    PASS = "PASS"
    FAIL = "FAIL"


# ─── Rationale Reference ─────────────────────────────────────────────


class RationaleRefType(StrEnum):
    """학술 근거 참조 유형."""

    PAPER = "paper"
    LESSON = "lesson"
    PRIOR_STRATEGY = "prior_strategy"


class RationaleReference(BaseModel):
    """학술 근거 참조."""

    model_config = ConfigDict(frozen=True)

    type: RationaleRefType
    title: str = ""
    source: str = ""
    url: str = ""
    relevance: str = ""


# ─── Models ──────────────────────────────────────────────────────────


class StrategyMeta(BaseModel):
    """전략 기본 정보."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Registry key (kebab-case)")
    display_name: str = Field(description="표시 이름")
    category: str = Field(description="전략 분류 (ML Ensemble, Trend Following, etc.)")
    timeframe: str = Field(description="타임프레임 (1D, 4H, 12H, etc.)")
    short_mode: str = Field(description="DISABLED | HEDGE_ONLY | FULL")
    status: StrategyStatus = Field(description="수명주기 상태")
    created_at: date = Field(description="생성일")
    retired_at: date | None = Field(default=None, description="폐기일")
    economic_rationale: str = Field(default="", description="경제적 논거")
    rationale_references: list[RationaleReference] = Field(
        default_factory=list, description="학술 근거 참조 목록"
    )
    rationale_category: str | None = Field(default=None, description="논거 카테고리")


class AssetMetrics(BaseModel):
    """에셋별 백테스트 성과."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    sharpe: float
    cagr: float = Field(description="Percentage (97.8 = +97.8%)")
    mdd: float = Field(description="Percentage (27.7 = -27.7%)")
    trades: int
    profit_factor: float | None = None
    win_rate: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    alpha: float | None = None
    beta: float | None = None


class PhaseResult(BaseModel):
    """Phase 검증 결과 (공통 스키마 + 유연한 details)."""

    model_config = ConfigDict(frozen=True)

    status: PhaseVerdict
    date: date
    details: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """의사결정 기록."""

    model_config = ConfigDict(frozen=True)

    date: date
    phase: PhaseId
    verdict: PhaseVerdict
    rationale: str


class StrategyRecord(BaseModel):
    """전략 전체 메타데이터 (YAML 1:1 매핑)."""

    model_config = ConfigDict(frozen=True)

    meta: StrategyMeta
    parameters: dict[str, Any] = Field(default_factory=dict)
    phases: dict[PhaseId, PhaseResult] = Field(default_factory=dict)
    asset_performance: list[AssetMetrics] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    version: int = Field(default=2, description="Schema version")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_asset(self) -> str | None:
        """Sharpe 기준 최고 에셋.

        Fallback: asset_performance가 비어있으면 P4 phase details에서 best_asset 조회.
        """
        if self.asset_performance:
            best = max(self.asset_performance, key=lambda a: a.sharpe)
            return best.symbol
        # Fallback: P4 details에서 best_asset 조회
        p4_result = self.phases.get(PhaseId.P4)
        if p4_result and p4_result.details.get("best_asset"):
            return str(p4_result.details["best_asset"])
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_phase(self) -> str | None:
        """현재 도달한 최고 PASS Phase (gap 허용)."""
        last_pass: PhaseId | None = None
        for pid in PHASE_ORDER:
            result = self.phases.get(pid)
            if result is None:
                continue
            if result.status == PhaseVerdict.PASS:
                last_pass = pid
            else:
                break  # FAIL 만나면 중단
        return last_pass

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fail_phase(self) -> str | None:
        """FAIL 발생한 Phase (있으면)."""
        for pid in PHASE_ORDER:
            result = self.phases.get(pid)
            if result is not None and result.status == PhaseVerdict.FAIL:
                return pid
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def next_phase(self) -> str | None:
        """다음 수행할 Phase. FAIL 시 None, 전체 PASS 시 None."""
        for pid in PHASE_ORDER:
            result = self.phases.get(pid)
            if result is not None and result.status == PhaseVerdict.FAIL:
                return None  # 파이프라인 blocked
        for pid in PHASE_ORDER:
            result = self.phases.get(pid)
            if result is None or result.status != PhaseVerdict.PASS:
                return pid  # 첫 미통과 Phase
        return None  # 전체 완료

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_sharpe(self) -> float | None:
        """Best Asset의 Sharpe."""
        if not self.asset_performance:
            return None
        return max(a.sharpe for a in self.asset_performance)
