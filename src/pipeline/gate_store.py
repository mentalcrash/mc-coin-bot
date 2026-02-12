"""Read-only YAML-based Gate Criteria store.

gates/criteria.yaml에서 Gate 평가 기준을 로드:
- load(gate_id): 단일 Gate 기준 로드
- load_all(): 전체 Gate 기준 로드 (GATE_ORDER 순)
- get_pass_thresholds(gate_id): PASS 임계값 dict 반환
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.pipeline.gate_models import GateCriteria, ThresholdMetric

_DEFAULT_PATH = Path("gates/criteria.yaml")


class GateCriteriaStore:
    """Read-only YAML 기반 Gate 평가 기준 저장소."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._cache: dict[str, GateCriteria] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_loaded(self) -> dict[str, GateCriteria]:
        """Lazy load + cache."""
        if self._cache is not None:
            return self._cache

        if not self._path.exists():
            msg = f"Gate criteria YAML not found: {self._path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        gates_list: list[dict[str, object]] = raw["gates"]

        self._cache = {}
        for gate_raw in gates_list:
            criteria = GateCriteria(**gate_raw)  # type: ignore[arg-type]
            self._cache[criteria.gate_id] = criteria

        return self._cache

    def load(self, gate_id: str) -> GateCriteria:
        """단일 Gate 기준 로드."""
        cache = self._ensure_loaded()
        if gate_id not in cache:
            msg = f"Gate not found: {gate_id}"
            raise KeyError(msg)
        return cache[gate_id]

    def load_all(self) -> list[GateCriteria]:
        """전체 Gate 기준 로드 (YAML 순서 유지)."""
        cache = self._ensure_loaded()
        return list(cache.values())

    def get_pass_thresholds(self, gate_id: str) -> dict[str, tuple[str, float]]:
        """Gate의 PASS 임계값을 dict로 반환.

        Returns:
            {metric_name: (operator, value)} 형식.
            threshold 타입이 아닌 Gate는 빈 dict 반환.
        """
        criteria = self.load(gate_id)
        if criteria.threshold is None:
            return {}
        return _metrics_to_dict(criteria.threshold.pass_metrics)


def _metrics_to_dict(metrics: list[ThresholdMetric]) -> dict[str, tuple[str, float]]:
    """ThresholdMetric 리스트 → {name: (operator, value)} dict."""
    return {m.name: (m.operator, m.value) for m in metrics}
