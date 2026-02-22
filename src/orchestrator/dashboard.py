"""AllocationDashboard — Pod 자본 배분 추적 및 시각화 데이터 제공.

배분 변화, 드리프트, PRC timeline, lifecycle overlay 등
orchestrator의 allocation follow-up 데이터를 수집합니다.

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Stateless query: Orchestrator 참조만, 상태 미보유
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestrator.orchestrator import StrategyOrchestrator

# ── Constants ─────────────────────────────────────────────────────

_EPSILON = 1e-12


# ── Data Classes ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PodDrift:
    """Pod별 현재 드리프트 정보."""

    pod_id: str
    current_fraction: float
    target_fraction: float
    drift: float
    drift_pct: float


@dataclass(frozen=True)
class AllocationSnapshot:
    """포트폴리오 전체 배분 스냅샷."""

    pod_drifts: tuple[PodDrift, ...]
    total_drift: float
    max_drift: float
    effective_n: float


@dataclass
class AllocationTimeline:
    """시간에 따른 배분 변화 추적 데이터."""

    timestamps: list[object] = field(default_factory=list)
    pod_fractions: dict[str, list[float]] = field(default_factory=dict)
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)


# ── AllocationDashboard ───────────────────────────────────────────


class AllocationDashboard:
    """Orchestrator의 allocation follow-up 데이터를 제공합니다.

    Args:
        orchestrator: StrategyOrchestrator 인스턴스
    """

    def __init__(self, orchestrator: StrategyOrchestrator) -> None:
        self._orchestrator = orchestrator

    def compute_drift(self) -> AllocationSnapshot:
        """현재 배분 vs 목표 배분의 드리프트를 계산합니다.

        Returns:
            AllocationSnapshot (pod별 drift, total, max, effective_n)
        """
        orch = self._orchestrator
        pods = orch.pods
        last_weights = orch.last_allocated_weights

        drifts: list[PodDrift] = []
        for pod in pods:
            if not pod.is_active:
                continue
            target = last_weights.get(pod.pod_id, pod.config.initial_fraction)
            current = pod.capital_fraction
            drift_abs = abs(current - target)
            drift_pct = drift_abs / target if target > _EPSILON else 0.0
            drifts.append(
                PodDrift(
                    pod_id=pod.pod_id,
                    current_fraction=current,
                    target_fraction=target,
                    drift=drift_abs,
                    drift_pct=drift_pct,
                )
            )

        total_drift = sum(d.drift for d in drifts)
        max_drift = max((d.drift for d in drifts), default=0.0)

        # Effective N: 1 / HHI (Herfindahl-Hirschman Index)
        fractions = [pod.capital_fraction for pod in pods if pod.is_active]
        total_frac = sum(fractions)
        if total_frac > _EPSILON:
            normalized = [f / total_frac for f in fractions]
            hhi = sum(n**2 for n in normalized)
            effective_n = 1.0 / hhi if hhi > _EPSILON else 0.0
        else:
            effective_n = 0.0

        return AllocationSnapshot(
            pod_drifts=tuple(drifts),
            total_drift=total_drift,
            max_drift=max_drift,
            effective_n=effective_n,
        )

    def get_timeline(self) -> AllocationTimeline:
        """Allocation history를 timeline 형식으로 변환합니다.

        Returns:
            AllocationTimeline (timestamps, pod별 fraction 히스토리, lifecycle 이벤트)
        """
        orch = self._orchestrator
        history = orch.allocation_history
        pod_ids = [pod.pod_id for pod in orch.pods]

        timeline = AllocationTimeline()
        timeline.lifecycle_events = list(orch.lifecycle_events)

        for record in history:
            timeline.timestamps.append(record.get("timestamp"))
            for pid in pod_ids:
                if pid not in timeline.pod_fractions:
                    timeline.pod_fractions[pid] = []
                val = record.get(pid)
                timeline.pod_fractions[pid].append(
                    float(val) if isinstance(val, int | float) else 0.0
                )

        return timeline

    def get_pod_leverage_usage(self) -> dict[str, dict[str, float]]:
        """Pod별 현재 레버리지 사용 현황을 반환합니다.

        Returns:
            {pod_id: {"gross_leverage": float, "max_leverage": float, "usage_pct": float}}
        """
        orch = self._orchestrator
        result: dict[str, dict[str, float]] = {}

        for pod in orch.pods:
            if not pod.is_active:
                continue
            pod_targets = orch.last_pod_targets.get(pod.pod_id, {})
            pod_gross = sum(abs(w) for w in pod_targets.values())
            cap_frac = pod.capital_fraction
            max_lev = pod.config.max_leverage
            pod_limit = max_lev * cap_frac

            result[pod.pod_id] = {
                "gross_leverage": pod_gross,
                "max_leverage": pod_limit,
                "usage_pct": pod_gross / pod_limit if pod_limit > _EPSILON else 0.0,
            }

        return result
