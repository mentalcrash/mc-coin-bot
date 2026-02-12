"""Tests for src/pipeline/lesson_store.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.pipeline.lesson_models import LessonCategory, LessonRecord
from src.pipeline.lesson_store import LessonStore


@pytest.fixture
def store(tmp_path: Path) -> LessonStore:
    return LessonStore(base_dir=tmp_path)


class TestCRUD:
    def test_save_and_load(self, store: LessonStore, sample_lesson: LessonRecord) -> None:
        store.save(sample_lesson)
        loaded = store.load(sample_lesson.id)
        assert loaded == sample_lesson

    def test_load_nonexistent(self, store: LessonStore) -> None:
        with pytest.raises(FileNotFoundError, match="Lesson YAML not found"):
            store.load(999)

    def test_exists(self, store: LessonStore, sample_lesson: LessonRecord) -> None:
        assert not store.exists(sample_lesson.id)
        store.save(sample_lesson)
        assert store.exists(sample_lesson.id)

    def test_load_all_empty(self, store: LessonStore) -> None:
        assert store.load_all() == []

    def test_load_all_sorted(
        self,
        store: LessonStore,
        sample_lesson: LessonRecord,
        sample_lesson_market: LessonRecord,
    ) -> None:
        store.save(sample_lesson_market)  # id=13
        store.save(sample_lesson)  # id=1
        records = store.load_all()
        assert len(records) == 2
        assert records[0].id == 1
        assert records[1].id == 13

    def test_roundtrip_yaml(self, store: LessonStore, sample_lesson: LessonRecord) -> None:
        store.save(sample_lesson)
        # Clear cache to force re-read from YAML
        store._cache.clear()
        loaded = store.load(sample_lesson.id)
        assert loaded == sample_lesson

    def test_next_id_empty(self, store: LessonStore) -> None:
        assert store.next_id() == 1

    def test_next_id_with_data(
        self,
        store: LessonStore,
        sample_lesson: LessonRecord,
        sample_lesson_market: LessonRecord,
    ) -> None:
        store.save(sample_lesson)  # id=1
        store.save(sample_lesson_market)  # id=13
        assert store.next_id() == 14

    def test_file_naming(self, store: LessonStore, sample_lesson: LessonRecord) -> None:
        store.save(sample_lesson)
        assert (store.base_dir / "001.yaml").exists()

    def test_file_naming_large_id(self, store: LessonStore) -> None:
        record = LessonRecord(
            id=100,
            title="Test",
            body="body",
            category=LessonCategory.META_ANALYSIS,
            added_at=date(2026, 2, 10),
        )
        store.save(record)
        assert (store.base_dir / "100.yaml").exists()

    def test_load_all_nonexistent_dir(self, tmp_path: Path) -> None:
        store = LessonStore(base_dir=tmp_path / "nonexistent")
        assert store.load_all() == []

    def test_cache_hit(self, store: LessonStore, sample_lesson: LessonRecord) -> None:
        store.save(sample_lesson)
        loaded1 = store.load(sample_lesson.id)
        loaded2 = store.load(sample_lesson.id)
        assert loaded1 is loaded2  # Same object from cache


class TestQuery:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        store: LessonStore,
        sample_lesson: LessonRecord,
        sample_lesson_market: LessonRecord,
    ) -> None:
        store.save(sample_lesson)
        store.save(sample_lesson_market)

    def test_filter_by_category(self, store: LessonStore) -> None:
        results = store.filter_by_category(LessonCategory.STRATEGY_DESIGN)
        assert len(results) == 1
        assert results[0].id == 1

    def test_filter_by_category_no_match(self, store: LessonStore) -> None:
        results = store.filter_by_category(LessonCategory.DATA_RESOLUTION)
        assert results == []

    def test_filter_by_tag(self, store: LessonStore) -> None:
        results = store.filter_by_tag("ML")
        assert len(results) == 1
        assert results[0].id == 1

    def test_filter_by_tag_case_insensitive(self, store: LessonStore) -> None:
        results = store.filter_by_tag("ml")
        assert len(results) == 1
        assert results[0].id == 1

    def test_filter_by_tag_no_match(self, store: LessonStore) -> None:
        results = store.filter_by_tag("nonexistent")
        assert results == []

    def test_filter_by_strategy(self, store: LessonStore) -> None:
        results = store.filter_by_strategy("ctrend")
        assert len(results) == 1
        assert results[0].id == 1

    def test_filter_by_strategy_no_match(self, store: LessonStore) -> None:
        results = store.filter_by_strategy("nonexistent")
        assert results == []

    def test_filter_by_timeframe(self, store: LessonStore) -> None:
        results = store.filter_by_timeframe("1H")
        assert len(results) == 1
        assert results[0].id == 13

    def test_filter_by_timeframe_no_match(self, store: LessonStore) -> None:
        results = store.filter_by_timeframe("4H")
        assert results == []
