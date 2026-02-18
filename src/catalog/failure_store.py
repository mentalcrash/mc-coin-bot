"""Read-only YAML-based Failure Pattern store.

catalogs/failure_patterns.yaml에서 실패 패턴 메타데이터를 로드합니다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.catalog.failure_models import FailurePattern, FailurePatternCatalog, Frequency

_DEFAULT_PATH = Path("catalogs/failure_patterns.yaml")


class FailurePatternStore:
    """Read-only YAML 기반 실패 패턴 저장소."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._catalog: FailurePatternCatalog | None = None
        self._patterns: dict[str, FailurePattern] | None = None

    def _ensure_loaded(self) -> FailurePatternCatalog:
        """Lazy load + cache."""
        if self._catalog is not None:
            return self._catalog

        if not self._path.exists():
            msg = f"Failure patterns YAML not found: {self._path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        self._catalog = FailurePatternCatalog(**raw)
        self._patterns = {p.id: p for p in self._catalog.patterns}
        return self._catalog

    def load(self, pattern_id: str) -> FailurePattern:
        """단일 패턴 로드."""
        self._ensure_loaded()
        assert self._patterns is not None
        if pattern_id not in self._patterns:
            msg = f"Failure pattern not found: {pattern_id}"
            raise KeyError(msg)
        return self._patterns[pattern_id]

    def load_all(self) -> list[FailurePattern]:
        """전체 패턴 로드 (YAML 순서 유지)."""
        catalog = self._ensure_loaded()
        return list(catalog.patterns)

    def filter_by_phase(self, phase: str) -> list[FailurePattern]:
        """Phase별 필터."""
        catalog = self._ensure_loaded()
        return [p for p in catalog.patterns if phase in p.affected_phases]

    def filter_by_frequency(self, freq: str) -> list[FailurePattern]:
        """빈도별 필터."""
        frequency = Frequency(freq)
        catalog = self._ensure_loaded()
        return [p for p in catalog.patterns if p.frequency == frequency]
