"""Tests for src/pipeline/lesson_models.py."""

from __future__ import annotations

from datetime import date

import pytest

from src.pipeline.lesson_models import LessonCategory, LessonRecord


class TestLessonCategory:
    def test_values(self) -> None:
        assert LessonCategory.STRATEGY_DESIGN == "strategy-design"
        assert LessonCategory.RISK_MANAGEMENT == "risk-management"
        assert LessonCategory.MARKET_STRUCTURE == "market-structure"
        assert LessonCategory.DATA_RESOLUTION == "data-resolution"
        assert LessonCategory.PIPELINE_PROCESS == "pipeline-process"
        assert LessonCategory.META_ANALYSIS == "meta-analysis"

    def test_count(self) -> None:
        assert len(LessonCategory) == 6


class TestLessonRecord:
    def test_minimal_creation(self) -> None:
        record = LessonRecord(
            id=1,
            title="Test",
            body="Test body",
            category=LessonCategory.STRATEGY_DESIGN,
            added_at=date(2026, 2, 10),
        )
        assert record.id == 1
        assert record.tags == []
        assert record.strategies == []
        assert record.timeframes == []

    def test_full_creation(self, sample_lesson: LessonRecord) -> None:
        assert sample_lesson.id == 1
        assert sample_lesson.title == "앙상블 > 단일지표"
        assert sample_lesson.category == LessonCategory.STRATEGY_DESIGN
        assert "ML" in sample_lesson.tags
        assert "ctrend" in sample_lesson.strategies
        assert "1D" in sample_lesson.timeframes

    def test_frozen(self, sample_lesson: LessonRecord) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_lesson.title = "changed"  # type: ignore[misc]

    def test_model_dump_json(self, sample_lesson: LessonRecord) -> None:
        data = sample_lesson.model_dump(mode="json")
        assert data["category"] == "strategy-design"
        assert data["added_at"] == "2026-02-10"

    def test_roundtrip(self, sample_lesson: LessonRecord) -> None:
        data = sample_lesson.model_dump(mode="json")
        restored = LessonRecord(**data)
        assert restored == sample_lesson
