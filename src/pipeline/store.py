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


def _deserialize(raw: dict[str, Any]) -> StrategyRecord:
    """YAML dict → StrategyRecord."""
    meta = StrategyMeta(**raw.get("meta", {}))

    phases: dict[PhaseId, PhaseResult] = {}
    for pid_str, pdata in raw.get("phases", {}).items():
        phases[PhaseId(pid_str)] = PhaseResult(
            status=PhaseVerdict(pdata["status"]),
            date=pdata["date"],
            details=pdata.get("details", {}),
        )

    assets = [AssetMetrics(**a) for a in raw.get("asset_performance", [])]

    decisions = [
        Decision(
            date=d["date"],
            phase=PhaseId(d["phase"]),
            verdict=PhaseVerdict(d["verdict"]),
            rationale=d["rationale"],
        )
        for d in raw.get("decisions", [])
    ]

    return StrategyRecord(
        meta=meta,
        parameters=raw.get("parameters", {}),
        phases=phases,
        asset_performance=assets,
        decisions=decisions,
        version=2,
    )
