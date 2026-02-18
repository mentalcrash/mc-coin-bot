"""Read-only YAML-based Phase Criteria store.

gates/phase-criteria.yaml에서 Phase 평가 기준을 로드:
- load(phase_id): 단일 Phase 기준 로드
- load_all(): 전체 Phase 기준 로드 (순서 유지)
- get_pass_thresholds(phase_id): PASS 임계값 dict 반환
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.pipeline.phase_criteria_models import PhaseCriteria, ThresholdMetric

_DEFAULT_PATH = Path("gates/phase-criteria.yaml")


class PhaseCriteriaStore:
    """Read-only YAML 기반 Phase 평가 기준 저장소."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._cache: dict[str, PhaseCriteria] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_loaded(self) -> dict[str, PhaseCriteria]:
        """Lazy load + cache."""
        if self._cache is not None:
            return self._cache

        if not self._path.exists():
            msg = f"Phase criteria YAML not found: {self._path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        phases_list: list[dict[str, object]] = raw["phases"]

        self._cache = {}
        for phase_raw in phases_list:
            criteria = PhaseCriteria(**phase_raw)  # type: ignore[arg-type]
            self._cache[criteria.phase_id] = criteria

        return self._cache

    def load(self, phase_id: str) -> PhaseCriteria:
        """단일 Phase 기준 로드."""
        cache = self._ensure_loaded()
        if phase_id not in cache:
            msg = f"Phase not found: {phase_id}"
            raise KeyError(msg)
        return cache[phase_id]

    def load_all(self) -> list[PhaseCriteria]:
        """전체 Phase 기준 로드 (YAML 순서 유지)."""
        cache = self._ensure_loaded()
        return list(cache.values())

    def get_pass_thresholds(self, phase_id: str) -> dict[str, tuple[str, float]]:
        """Phase의 PASS 임계값을 dict로 반환.

        Returns:
            {metric_name: (operator, value)} 형식.
            threshold 타입이 아닌 Phase는 빈 dict 반환.
        """
        criteria = self.load(phase_id)
        if criteria.threshold is None:
            return {}
        return _metrics_to_dict(criteria.threshold.pass_metrics)


def _metrics_to_dict(metrics: list[ThresholdMetric]) -> dict[str, tuple[str, float]]:
    """ThresholdMetric 리스트 → {name: (operator, value)} dict."""
    return {m.name: (m.operator, m.value) for m in metrics}
