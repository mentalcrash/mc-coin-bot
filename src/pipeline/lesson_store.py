"""YAML-based Lesson data store.

lessons/ 디렉토리에 교훈별 YAML 파일을 저장/로드:
- CRUD: load, save, load_all, exists, next_id
- Query: filter_by_category, filter_by_tag, filter_by_strategy, filter_by_timeframe
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.pipeline.lesson_models import LessonCategory, LessonRecord

_DEFAULT_BASE_DIR = Path("lessons")


class LessonStore:
    """YAML 기반 교훈 데이터 저장소."""

    def __init__(self, base_dir: Path = _DEFAULT_BASE_DIR) -> None:
        self._base_dir = base_dir
        self._cache: dict[int, LessonRecord] = {}

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ─── CRUD ────────────────────────────────────────────────────────

    def load(self, lesson_id: int) -> LessonRecord:
        """YAML 파일에서 LessonRecord 로드."""
        if lesson_id in self._cache:
            return self._cache[lesson_id]

        path = self._base_dir / f"{lesson_id:03d}.yaml"
        if not path.exists():
            msg = f"Lesson YAML not found: {path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        record = _deserialize(raw)
        self._cache[lesson_id] = record
        return record

    def save(self, record: LessonRecord) -> None:
        """LessonRecord를 YAML 파일로 저장."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / f"{record.id:03d}.yaml"
        data = _serialize(record)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._cache[record.id] = record

    def load_all(self) -> list[LessonRecord]:
        """모든 교훈 YAML 로드 (id 순 정렬)."""
        if not self._base_dir.exists():
            return []
        records: list[LessonRecord] = []
        for path in sorted(self._base_dir.glob("*.yaml")):
            stem = path.stem
            try:
                lesson_id = int(stem)
            except ValueError:
                continue
            records.append(self.load(lesson_id))
        return records

    def exists(self, lesson_id: int) -> bool:
        """교훈 YAML 존재 여부."""
        return (self._base_dir / f"{lesson_id:03d}.yaml").exists()

    def next_id(self) -> int:
        """다음 사용 가능한 ID (max(ids) + 1)."""
        records = self.load_all()
        if not records:
            return 1
        return max(r.id for r in records) + 1

    # ─── Query ───────────────────────────────────────────────────────

    def filter_by_category(self, category: LessonCategory) -> list[LessonRecord]:
        """카테고리별 필터링."""
        return [r for r in self.load_all() if r.category == category]

    def filter_by_tag(self, tag: str) -> list[LessonRecord]:
        """태그별 필터링 (case-insensitive)."""
        tag_lower = tag.lower()
        return [r for r in self.load_all() if any(t.lower() == tag_lower for t in r.tags)]

    def filter_by_strategy(self, strategy: str) -> list[LessonRecord]:
        """관련 전략별 필터링."""
        return [r for r in self.load_all() if strategy in r.strategies]

    def filter_by_timeframe(self, timeframe: str) -> list[LessonRecord]:
        """타임프레임별 필터링."""
        return [r for r in self.load_all() if timeframe in r.timeframes]


# ─── Serialization helpers ───────────────────────────────────────────


def _serialize(record: LessonRecord) -> dict[str, Any]:
    """LessonRecord → YAML-safe dict."""
    return record.model_dump(mode="json")


def _deserialize(raw: dict[str, Any]) -> LessonRecord:
    """YAML dict → LessonRecord."""
    return LessonRecord(**raw)
