"""YAML-based Experiment data store.

AuditStore와 동일한 YAML CRUD 패턴을 따릅니다.

Directory structure:
    experiments/{strategy_name}/{phase_id}_{YYYYMMDD_HHMMSS}.yaml

Rules Applied:
    - AuditStore YAML CRUD 패턴
    - model_dump(mode="json") for serialization
    - Cache invalidation on save
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.pipeline.experiment_models import ExperimentAnalysis, ExperimentRecord

_DEFAULT_BASE_DIR = Path("experiments")


class ExperimentStore:
    """YAML 기반 Experiment 데이터 저장소."""

    def __init__(self, base_dir: Path = _DEFAULT_BASE_DIR) -> None:
        self._base_dir = base_dir
        self._cache: dict[str, list[ExperimentRecord]] = {}

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def save(self, record: ExperimentRecord) -> Path:
        """ExperimentRecord를 YAML로 저장.

        Directory: base_dir/{strategy_name}/
        Filename: {phase_id}_{YYYYMMDD_HHMMSS}.yaml

        Returns:
            저장된 파일 경로.
        """
        strategy_dir = self._base_dir / record.strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        ts_str = record.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{record.phase_id}_{ts_str}.yaml"
        path = strategy_dir / filename

        data = self._serialize(record)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        # Cache invalidation
        self._cache.pop(record.strategy_name, None)

        return path

    def load_all_for_strategy(self, strategy_name: str) -> list[ExperimentRecord]:
        """전략의 모든 실험 기록 로드 (timestamp 정렬).

        Args:
            strategy_name: 전략 이름.

        Returns:
            timestamp 오름차순 정렬된 ExperimentRecord 리스트.
        """
        if strategy_name in self._cache:
            return self._cache[strategy_name]

        strategy_dir = self._base_dir / strategy_name
        if not strategy_dir.exists():
            return []

        records: list[ExperimentRecord] = []
        for path in sorted(strategy_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw is not None:
                records.append(self._deserialize(raw))

        records.sort(key=lambda r: r.timestamp)
        self._cache[strategy_name] = records
        return records

    def get_latest(self, strategy_name: str, phase_id: str | None = None) -> ExperimentRecord | None:
        """가장 최근 실험 기록 반환.

        Args:
            strategy_name: 전략 이름.
            phase_id: 특정 phase로 필터링 (선택).

        Returns:
            가장 최근 ExperimentRecord 또는 None.
        """
        records = self.load_all_for_strategy(strategy_name)
        if phase_id is not None:
            records = [r for r in records if r.phase_id == phase_id]
        return records[-1] if records else None

    def analyze(self, strategy_name: str) -> ExperimentAnalysis | None:
        """전략의 실험 요약 분석 생성.

        Args:
            strategy_name: 전략 이름.

        Returns:
            ExperimentAnalysis 또는 기록 없으면 None.
        """
        records = self.load_all_for_strategy(strategy_name)
        if not records:
            return None

        total = len(records)
        passed_count = sum(1 for r in records if r.passed)
        pass_rate = passed_count / total

        # Best phase: 에셋별 sharpe 평균이 가장 높은 record의 phase_id
        best_record = max(records, key=_avg_sharpe)
        best_sharpe = _avg_sharpe(best_record)

        # 전체 에셋 MDD 평균
        all_mdds: list[float] = [ar.mdd for r in records for ar in r.asset_results]
        avg_mdd = sum(all_mdds) / len(all_mdds) if all_mdds else 0.0

        return ExperimentAnalysis(
            strategy_name=strategy_name,
            total_experiments=total,
            pass_rate=pass_rate,
            best_phase=best_record.phase_id,
            best_sharpe=best_sharpe,
            avg_mdd=avg_mdd,
        )

    def _serialize(self, record: ExperimentRecord) -> dict[str, Any]:
        """ExperimentRecord → YAML-safe dict."""
        return record.model_dump(mode="json")

    def _deserialize(self, raw: dict[str, Any]) -> ExperimentRecord:
        """YAML dict → ExperimentRecord."""
        return ExperimentRecord(**raw)


def _avg_sharpe(record: ExperimentRecord) -> float:
    """Record 내 에셋별 sharpe 평균."""
    if not record.asset_results:
        return 0.0
    return sum(ar.sharpe for ar in record.asset_results) / len(record.asset_results)
