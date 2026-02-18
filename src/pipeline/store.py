"""YAML-based Strategy metadata store.

strategies/ 디렉토리에 전략별 YAML 파일을 저장/로드:
- CRUD: load, save, load_all, exists
- Query: filter_by_status, get_active, get_at_phase, get_failed_at_phase
- Mutation: record_phase, update_status, set_asset_performance
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.models import (
    AssetMetrics,
    Decision,
    PhaseId,
    PhaseResult,
    PhaseVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)

_DEFAULT_BASE_DIR = Path("strategies")

# v1 Gate → v2 Phase 매핑 (역직렬화용)
GATE_TO_PHASE: dict[str, PhaseId] = {
    "G0A": PhaseId.P1,
    "G0B": PhaseId.P3,
    "G1": PhaseId.P4,
    "G2": PhaseId.P4,  # G1+G2 → P4 (merge)
    "G2H": PhaseId.P5,
    "G3": PhaseId.P5,  # G2H+G3 → P5 (merge)
    "G4": PhaseId.P6,
    "G5": PhaseId.P7,
}


class StrategyStore:
    """YAML 기반 전략 메타데이터 저장소."""

    def __init__(self, base_dir: Path = _DEFAULT_BASE_DIR) -> None:
        self._base_dir = base_dir
        self._cache: dict[str, StrategyRecord] = {}

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ─── CRUD ────────────────────────────────────────────────────────

    def load(self, name: str) -> StrategyRecord:
        """YAML 파일에서 StrategyRecord 로드."""
        if name in self._cache:
            return self._cache[name]

        path = self._base_dir / f"{name}.yaml"
        if not path.exists():
            msg = f"Strategy YAML not found: {path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        record = _deserialize(raw)
        self._cache[name] = record
        return record

    def save(self, record: StrategyRecord) -> None:
        """StrategyRecord를 YAML 파일로 저장."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / f"{record.meta.name}.yaml"
        data = _serialize(record)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._cache[record.meta.name] = record

    def load_all(self) -> list[StrategyRecord]:
        """모든 전략 YAML 로드."""
        if not self._base_dir.exists():
            return []
        records: list[StrategyRecord] = []
        for path in sorted(self._base_dir.glob("*.yaml")):
            name = path.stem
            records.append(self.load(name))
        return records

    def exists(self, name: str) -> bool:
        """전략 YAML 존재 여부."""
        return (self._base_dir / f"{name}.yaml").exists()

    # ─── Query ───────────────────────────────────────────────────────

    def filter_by_status(self, status: StrategyStatus) -> list[StrategyRecord]:
        """상태별 필터링."""
        return [r for r in self.load_all() if r.meta.status == status]

    def get_active(self) -> list[StrategyRecord]:
        """ACTIVE 전략 목록."""
        return self.filter_by_status(StrategyStatus.ACTIVE)

    def get_retired(self) -> list[StrategyRecord]:
        """RETIRED 전략 목록."""
        return self.filter_by_status(StrategyStatus.RETIRED)

    def get_at_phase(self, phase: PhaseId) -> list[StrategyRecord]:
        """특정 Phase에 도달한 전략 (current_phase == phase)."""
        return [r for r in self.load_all() if r.current_phase == phase]

    def get_failed_at_phase(self, phase: PhaseId) -> list[StrategyRecord]:
        """특정 Phase에서 FAIL된 전략."""
        return [r for r in self.load_all() if r.fail_phase == phase]

    # ─── Mutation ────────────────────────────────────────────────────

    def record_phase(
        self,
        name: str,
        phase: PhaseId,
        verdict: PhaseVerdict,
        details: dict[str, Any],
        rationale: str,
    ) -> StrategyRecord:
        """Phase 결과 기록 (frozen model → copy + modify)."""
        record = self.load(name)
        new_phases = dict(record.phases)
        new_phases[phase] = PhaseResult(status=verdict, date=date.today(), details=details)

        new_decisions = [
            *record.decisions,
            Decision(
                date=date.today(),
                phase=phase,
                verdict=verdict,
                rationale=rationale,
            ),
        ]

        updated = record.model_copy(update={"phases": new_phases, "decisions": new_decisions})

        # IMPLEMENTED + PASS → TESTING 자동 전환
        if updated.meta.status == StrategyStatus.IMPLEMENTED and verdict == PhaseVerdict.PASS:
            new_meta = updated.meta.model_copy(update={"status": StrategyStatus.TESTING})
            updated = updated.model_copy(update={"meta": new_meta})

        self.save(updated)
        return updated

    def update_status(self, name: str, status: StrategyStatus) -> StrategyRecord:
        """전략 상태 변경."""
        record = self.load(name)
        new_meta = record.meta.model_copy(update={"status": status})
        if status == StrategyStatus.RETIRED:
            new_meta = new_meta.model_copy(update={"retired_at": date.today()})

        updated = record.model_copy(update={"meta": new_meta})
        self.save(updated)
        return updated

    def set_asset_performance(
        self,
        name: str,
        metrics: list[AssetMetrics],
    ) -> StrategyRecord:
        """에셋별 성과 데이터 설정."""
        record = self.load(name)
        updated = record.model_copy(update={"asset_performance": metrics})
        self.save(updated)
        return updated


# ─── Serialization helpers ───────────────────────────────────────────


def _serialize(record: StrategyRecord) -> dict[str, Any]:
    """StrategyRecord → YAML-safe dict (StrEnum → plain str)."""
    # mode="json" converts StrEnum/date to plain str automatically
    meta = record.meta.model_dump(mode="json")

    phases: dict[str, Any] = {}
    for pid, result in record.phases.items():
        phases[str(pid)] = result.model_dump(mode="json")

    assets = [a.model_dump(mode="json", exclude_none=True) for a in record.asset_performance]

    decisions = [d.model_dump(mode="json") for d in record.decisions]

    return {
        "version": record.version,
        "meta": meta,
        "parameters": record.parameters,
        "phases": phases,
        "asset_performance": assets,
        "decisions": decisions,
    }


_V2_FORMAT_VERSION = 2


def _merge_gate_results(
    results: dict[str, dict[str, Any]],
    gate_a: str,
    gate_b: str,
) -> dict[str, Any] | None:
    """두 Gate 결과를 하나의 Phase 결과로 머지.

    규칙: 나중 날짜 사용, details는 gate prefix로 합침.
    둘 다 없으면 None, 하나만 있으면 그 결과 사용.
    """
    a = results.get(gate_a)
    b = results.get(gate_b)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    # 둘 다 존재 → 머지
    merged_status = "FAIL" if a["status"] == "FAIL" or b["status"] == "FAIL" else "PASS"

    # 나중 날짜 사용
    date_a = a["date"]
    date_b = b["date"]
    merged_date = max(date_a, date_b) if date_a and date_b else date_a or date_b

    # details 합침 with prefix
    prefix_a = gate_a.lower() + "_"
    prefix_b = gate_b.lower() + "_"
    merged_details: dict[str, Any] = {}
    for k, v in a.get("details", {}).items():
        merged_details[prefix_a + k] = v
    for k, v in b.get("details", {}).items():
        merged_details[prefix_b + k] = v

    return {"status": merged_status, "date": merged_date, "details": merged_details}


def _deserialize(raw: dict[str, Any]) -> StrategyRecord:
    """YAML dict → StrategyRecord (v1/v2 듀얼 지원)."""
    raw_meta = raw.get("meta", {})
    meta = StrategyMeta(**raw_meta)

    version = raw.get("version", 1)

    phases: dict[PhaseId, PhaseResult] = {}

    if "phases" in raw and version >= _V2_FORMAT_VERSION:
        # v2 직접 로드
        for pid_str, pdata in raw["phases"].items():
            phases[PhaseId(pid_str)] = PhaseResult(
                status=PhaseVerdict(pdata["status"]),
                date=pdata["date"],
                details=pdata.get("details", {}),
            )
    elif "gates" in raw:
        # v1 → v2 변환
        gate_results: dict[str, dict[str, Any]] = dict(raw["gates"].items())

        # 단순 1:1 매핑 (G0A→P1, G0B→P3, G4→P6, G5→P7)
        simple_map = {"G0A": PhaseId.P1, "G0B": PhaseId.P3, "G4": PhaseId.P6, "G5": PhaseId.P7}
        for gid, pid in simple_map.items():
            if gid in gate_results:
                gdata = gate_results[gid]
                phases[pid] = PhaseResult(
                    status=PhaseVerdict(gdata["status"]),
                    date=gdata["date"],
                    details=gdata.get("details", {}),
                )

        # G1+G2 → P4 머지
        merged_p4 = _merge_gate_results(gate_results, "G1", "G2")
        if merged_p4:
            phases[PhaseId.P4] = PhaseResult(
                status=PhaseVerdict(merged_p4["status"]),
                date=merged_p4["date"],
                details=merged_p4.get("details", {}),
            )

        # G2H+G3 → P5 머지
        merged_p5 = _merge_gate_results(gate_results, "G2H", "G3")
        if merged_p5:
            phases[PhaseId.P5] = PhaseResult(
                status=PhaseVerdict(merged_p5["status"]),
                date=merged_p5["date"],
                details=merged_p5.get("details", {}),
            )

    assets = [AssetMetrics(**a) for a in raw.get("asset_performance", [])]

    decisions: list[Decision] = []
    for d in raw.get("decisions", []):
        # v1: "gate" key → v2: "phase" key
        if "phase" in d:
            phase_id = PhaseId(d["phase"])
        elif "gate" in d:
            phase_id = GATE_TO_PHASE.get(d["gate"], PhaseId.P1)
        else:
            phase_id = PhaseId.P1
        decisions.append(
            Decision(
                date=d["date"],
                phase=phase_id,
                verdict=PhaseVerdict(d["verdict"]),
                rationale=d["rationale"],
            )
        )

    return StrategyRecord(
        meta=meta,
        parameters=raw.get("parameters", {}),
        phases=phases,
        asset_performance=assets,
        decisions=decisions,
        version=2,
    )
