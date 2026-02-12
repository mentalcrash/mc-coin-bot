"""Strategy Pipeline Pydantic models.

전략 수명주기 메타데이터를 관리하는 모델:
- StrategyMeta: 전략 기본 정보
- AssetMetrics: 에셋별 백테스트 성과
- GateResult: Gate 검증 결과
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


class GateId(StrEnum):
    """Gate 식별자."""

    G0A = "G0A"
    G0B = "G0B"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    G6 = "G6"
    G7 = "G7"


# Gate 순서 (진행도 계산용)
GATE_ORDER: list[GateId] = [
    GateId.G0A,
    GateId.G0B,
    GateId.G1,
    GateId.G2,
    GateId.G3,
    GateId.G4,
    GateId.G5,
    GateId.G6,
    GateId.G7,
]


class GateVerdict(StrEnum):
    """Gate 판정 결과."""

    PASS = "PASS"
    FAIL = "FAIL"


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


class GateResult(BaseModel):
    """Gate 검증 결과 (공통 스키마 + 유연한 details)."""

    model_config = ConfigDict(frozen=True)

    status: GateVerdict
    date: date
    details: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """의사결정 기록."""

    model_config = ConfigDict(frozen=True)

    date: date
    gate: GateId
    verdict: GateVerdict
    rationale: str


class StrategyRecord(BaseModel):
    """전략 전체 메타데이터 (YAML 1:1 매핑)."""

    model_config = ConfigDict(frozen=True)

    meta: StrategyMeta
    parameters: dict[str, Any] = Field(default_factory=dict)
    gates: dict[GateId, GateResult] = Field(default_factory=dict)
    asset_performance: list[AssetMetrics] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_asset(self) -> str | None:
        """Sharpe 기준 최고 에셋."""
        if not self.asset_performance:
            return None
        best = max(self.asset_performance, key=lambda a: a.sharpe)
        return best.symbol

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_gate(self) -> str | None:
        """현재 도달한 최고 PASS Gate (gap 허용)."""
        last_pass: GateId | None = None
        for gid in GATE_ORDER:
            result = self.gates.get(gid)
            if result is None:
                continue
            if result.status == GateVerdict.PASS:
                last_pass = gid
            else:
                break  # FAIL 만나면 중단
        return last_pass

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fail_gate(self) -> str | None:
        """FAIL 발생한 Gate (있으면)."""
        for gid in GATE_ORDER:
            result = self.gates.get(gid)
            if result is not None and result.status == GateVerdict.FAIL:
                return gid
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_sharpe(self) -> float | None:
        """Best Asset의 Sharpe."""
        if not self.asset_performance:
            return None
        return max(a.sharpe for a in self.asset_performance)
