"""Read-only YAML-based Indicator Catalog store.

catalogs/indicators.yaml에서 지표 메타데이터를 로드합니다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.catalog.indicator_models import (
    AlphaPotential,
    IndicatorCatalog,
    IndicatorCategory,
    IndicatorEntry,
)

_DEFAULT_PATH = Path("catalogs/indicators.yaml")


class IndicatorCatalogStore:
    """Read-only YAML 기반 지표 카탈로그 저장소."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._catalog: IndicatorCatalog | None = None
        self._indicators: dict[str, IndicatorEntry] | None = None

    def _ensure_loaded(self) -> IndicatorCatalog:
        """Lazy load + cache."""
        if self._catalog is not None:
            return self._catalog

        if not self._path.exists():
            msg = f"Indicator catalog YAML not found: {self._path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        self._catalog = IndicatorCatalog(**raw)
        self._indicators = {i.id: i for i in self._catalog.indicators}
        return self._catalog

    def load(self, indicator_id: str) -> IndicatorEntry:
        """단일 지표 로드."""
        self._ensure_loaded()
        assert self._indicators is not None
        if indicator_id not in self._indicators:
            msg = f"Indicator not found: {indicator_id}"
            raise KeyError(msg)
        return self._indicators[indicator_id]

    def load_all(self) -> list[IndicatorEntry]:
        """전체 지표 로드 (YAML 순서 유지)."""
        catalog = self._ensure_loaded()
        return list(catalog.indicators)

    def get_by_category(self, category: str) -> list[IndicatorEntry]:
        """카테고리별 필터."""
        cat = IndicatorCategory(category)
        catalog = self._ensure_loaded()
        return [i for i in catalog.indicators if i.category == cat]

    def get_unused(self) -> list[IndicatorEntry]:
        """미사용 지표 (used_by가 비어있는 지표)."""
        catalog = self._ensure_loaded()
        return [i for i in catalog.indicators if not i.used_by]

    def get_by_potential(self, potential: str) -> list[IndicatorEntry]:
        """Alpha potential별 필터."""
        pot = AlphaPotential(potential)
        catalog = self._ensure_loaded()
        return [i for i in catalog.indicators if i.alpha_potential == pot]

    def get_by_strategy(self, strategy: str) -> list[IndicatorEntry]:
        """전략에서 사용하는 지표."""
        catalog = self._ensure_loaded()
        return [i for i in catalog.indicators if strategy in i.used_by]
