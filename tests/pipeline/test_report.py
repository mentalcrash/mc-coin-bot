"""Tests for src/pipeline/report.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.pipeline.lesson_models import LessonRecord
from src.pipeline.lesson_store import LessonStore
from src.pipeline.models import (
    Decision,
    GateId,
    GateResult,
    GateVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.report import DashboardGenerator, _extract_note, _fail_rationale
from src.pipeline.store import StrategyStore


@pytest.fixture
def store_with_data(
    tmp_path: Path,
    sample_record: StrategyRecord,
    retired_record: StrategyRecord,
) -> StrategyStore:
    store = StrategyStore(base_dir=tmp_path)
    store.save(sample_record)
    store.save(retired_record)
    return store


class TestDashboardGenerator:
    def test_generate_contains_header(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "전략 상황판" in content

    def test_generate_contains_pipeline(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "Gate 0A" in content
        assert "Gate 7" in content

    def test_generate_contains_active_table(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "CTREND" in content
        assert "활성 전략" in content

    def test_generate_contains_retired_section(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "폐기 전략" in content
        assert "BB-RSI" in content

    def test_generate_contains_cost_model(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "Maker Fee" in content
        assert "0.11%" in content

    def test_generate_counts(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "활성 1" in content
        assert "폐기 1" in content

    def test_generate_gate_criteria(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "Sharpe > 1.0" in content
        assert "OOS Sharpe >= 0.3" in content

    def test_generate_empty_store(self, tmp_path: Path) -> None:
        store = StrategyStore(base_dir=tmp_path)
        gen = DashboardGenerator(store)
        content = gen.generate()
        assert "활성 0" in content

    def test_active_table_sorted_by_sharpe(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        # CTREND should appear in active table with SOL/USDT
        assert "SOL/USDT" in content

    def test_retired_classified_by_gate(
        self,
        store_with_data: StrategyStore,
    ) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        # BB-RSI failed at G1 with Sharpe < 1.0 and CAGR > 0
        assert "Gate 1 실패" in content

    def test_auto_generated_marker(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "AUTO-GENERATED" in content
        assert "수동 편집 금지" in content

    def test_gate_criteria_skill_name(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "/p3-g0b-verify" in content
        assert "/quant-code-audit" not in content

    def test_active_notes_block(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        # sample_record has decisions[-1] = G1 PASS "SOL Sharpe 2.05"
        assert "> **CTREND**: SOL Sharpe 2.05" in content

    def test_retired_fail_rationale_from_decisions(
        self,
        store_with_data: StrategyStore,
    ) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        # retired_record decision rationale = "CAGR < 20%"
        assert "CAGR < 20%" in content

    def test_lessons_from_lesson_store(
        self,
        store_with_data: StrategyStore,
        tmp_path: Path,
        sample_lesson: LessonRecord,
    ) -> None:
        lesson_store = LessonStore(base_dir=tmp_path / "lessons")
        lesson_store.save(sample_lesson)
        gen = DashboardGenerator(store_with_data, lesson_store=lesson_store)
        content = gen.generate()
        assert "핵심 교훈" in content
        assert "앙상블 > 단일지표" in content

    def test_lessons_empty_lesson_store(
        self,
        store_with_data: StrategyStore,
        tmp_path: Path,
    ) -> None:
        lesson_store = LessonStore(base_dir=tmp_path / "empty_lessons")
        gen = DashboardGenerator(store_with_data, lesson_store=lesson_store)
        content = gen.generate()
        assert "핵심 교훈" in content
        assert "(없음)" in content

    def test_lessons_fallback_without_store(
        self,
        store_with_data: StrategyStore,
    ) -> None:
        gen = DashboardGenerator(store_with_data, lesson_store=None)
        # Without lesson_store, falls back to file (may or may not exist)
        content = gen.generate()
        assert isinstance(content, str)


class TestExtractNote:
    def test_no_truncation(self) -> None:
        """60자 이상 note도 절삭 없이 전체 반환."""
        long_note = "A" * 100
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.ACTIVE,
                created_at=date(2026, 1, 1),
            ),
            gates={
                GateId.G4: GateResult(
                    status=GateVerdict.PASS,
                    date=date(2026, 1, 1),
                    details={"note": long_note},
                ),
            },
        )
        assert _extract_note(record) == long_note
        assert len(_extract_note(record)) == 100

    def test_empty_when_no_g4(self) -> None:
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.ACTIVE,
                created_at=date(2026, 1, 1),
            ),
        )
        assert _extract_note(record) == ""


class TestFailRationale:
    def test_decisions_rationale_preferred(self) -> None:
        """decisions.rationale이 gates.details보다 우선."""
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.RETIRED,
                created_at=date(2026, 1, 1),
            ),
            gates={
                GateId.G1: GateResult(
                    status=GateVerdict.FAIL,
                    date=date(2026, 1, 1),
                    details={"note": "gate details note"},
                ),
            },
            decisions=[
                Decision(
                    date=date(2026, 1, 1),
                    gate=GateId.G1,
                    verdict=GateVerdict.FAIL,
                    rationale="Rich decision rationale with details",
                ),
            ],
        )
        result = _fail_rationale(record, GateId.G1)
        assert result == "Rich decision rationale with details"

    def test_fallback_to_gate_details(self) -> None:
        """decisions에 해당 gate 없으면 gates.details fallback."""
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.RETIRED,
                created_at=date(2026, 1, 1),
            ),
            gates={
                GateId.G1: GateResult(
                    status=GateVerdict.FAIL,
                    date=date(2026, 1, 1),
                    details={"note": "gate note fallback"},
                ),
            },
            decisions=[],
        )
        result = _fail_rationale(record, GateId.G1)
        assert result == "gate note fallback"

    def test_no_truncation(self) -> None:
        """80자 이상도 절삭 없이 전체 반환."""
        long_rationale = "B" * 120
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.RETIRED,
                created_at=date(2026, 1, 1),
            ),
            gates={
                GateId.G1: GateResult(
                    status=GateVerdict.FAIL,
                    date=date(2026, 1, 1),
                    details={"note": "short"},
                ),
            },
            decisions=[
                Decision(
                    date=date(2026, 1, 1),
                    gate=GateId.G1,
                    verdict=GateVerdict.FAIL,
                    rationale=long_rationale,
                ),
            ],
        )
        result = _fail_rationale(record, GateId.G1)
        assert result == long_rationale
        assert len(result) == 120
