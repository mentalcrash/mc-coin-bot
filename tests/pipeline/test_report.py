"""Tests for src/pipeline/report.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from rich.console import Console

from src.pipeline.lesson_models import LessonRecord
from src.pipeline.lesson_store import LessonStore
from src.pipeline.models import (
    Decision,
    PhaseId,
    PhaseResult,
    PhaseVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.phase_criteria_store import PhaseCriteriaStore
from src.pipeline.report import ConsoleRenderer, DashboardGenerator, _extract_note, _fail_rationale
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
        assert "Phase 1" in content
        assert "Phase 7" in content

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
        assert "편도 합계" in content

    def test_generate_counts(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "활성 1" in content
        assert "폐기 1" in content

    def test_generate_phase_criteria_fallback(self, store_with_data: StrategyStore) -> None:
        """phase_store=None이면 하드코딩 fallback 사용."""
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "Sharpe > 1.0" in content
        assert "OOS Sharpe >= 0.3" in content

    def test_generate_phase_criteria_dynamic(
        self,
        store_with_data: StrategyStore,
        phase_yaml_path: Path,
    ) -> None:
        """phase_store 있으면 동적 테이블 생성."""
        p_store = PhaseCriteriaStore(path=phase_yaml_path)
        gen = DashboardGenerator(store_with_data, phase_store=p_store)
        content = gen.generate()
        assert "Phase별 통과 기준" in content
        assert "아이디어 검증" in content
        assert "Sharpe > 1" in content

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

    def test_retired_classified_by_phase(
        self,
        store_with_data: StrategyStore,
    ) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        # BB-RSI failed at P4 with Sharpe < 1.0 and CAGR > 0
        assert "Phase 4 실패" in content

    def test_auto_generated_marker(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "AUTO-GENERATED" in content
        assert "--output" in content

    def test_phase_criteria_skill_name(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        assert "/p3-g0b-verify" in content
        assert "/quant-code-audit" not in content

    def test_active_notes_block(self, store_with_data: StrategyStore) -> None:
        gen = DashboardGenerator(store_with_data)
        content = gen.generate()
        # sample_record has decisions[-1] = P4 PASS "SOL Sharpe 2.05"
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

    def test_lessons_without_store_returns_empty(
        self,
        store_with_data: StrategyStore,
    ) -> None:
        gen = DashboardGenerator(store_with_data, lesson_store=None)
        content = gen.generate()
        # lesson_store=None이면 교훈 섹션 없음
        assert "핵심 교훈" not in content


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
            phases={
                PhaseId.P6: PhaseResult(
                    status=PhaseVerdict.PASS,
                    date=date(2026, 1, 1),
                    details={"note": long_note},
                ),
            },
        )
        assert _extract_note(record) == long_note
        assert len(_extract_note(record)) == 100

    def test_empty_when_no_p6(self) -> None:
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
        """decisions.rationale이 phases.details보다 우선."""
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
            phases={
                PhaseId.P4: PhaseResult(
                    status=PhaseVerdict.FAIL,
                    date=date(2026, 1, 1),
                    details={"note": "phase details note"},
                ),
            },
            decisions=[
                Decision(
                    date=date(2026, 1, 1),
                    phase=PhaseId.P4,
                    verdict=PhaseVerdict.FAIL,
                    rationale="Rich decision rationale with details",
                ),
            ],
        )
        result = _fail_rationale(record, PhaseId.P4)
        assert result == "Rich decision rationale with details"

    def test_fallback_to_phase_details(self) -> None:
        """decisions에 해당 phase 없으면 phases.details fallback."""
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
            phases={
                PhaseId.P4: PhaseResult(
                    status=PhaseVerdict.FAIL,
                    date=date(2026, 1, 1),
                    details={"note": "phase note fallback"},
                ),
            },
            decisions=[],
        )
        result = _fail_rationale(record, PhaseId.P4)
        assert result == "phase note fallback"

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
            phases={
                PhaseId.P4: PhaseResult(
                    status=PhaseVerdict.FAIL,
                    date=date(2026, 1, 1),
                    details={"note": "short"},
                ),
            },
            decisions=[
                Decision(
                    date=date(2026, 1, 1),
                    phase=PhaseId.P4,
                    verdict=PhaseVerdict.FAIL,
                    rationale=long_rationale,
                ),
            ],
        )
        result = _fail_rationale(record, PhaseId.P4)
        assert result == long_rationale
        assert len(result) == 120


def _capture(store_with_data: StrategyStore, **kwargs: object) -> str:
    """ConsoleRenderer 출력 캡처 헬퍼."""
    console = Console(file=None, no_color=True, width=120)
    renderer = ConsoleRenderer(store_with_data, console=console, **kwargs)  # type: ignore[arg-type]
    with console.capture() as capture:
        renderer.render()
    return capture.get()


class TestConsoleRenderer:
    def test_render_header(self, store_with_data: StrategyStore) -> None:
        output = _capture(store_with_data)
        assert "Strategy Dashboard" in output
        assert "Active: 1" in output
        assert "Retired: 1" in output

    def test_render_active_strategy(self, store_with_data: StrategyStore) -> None:
        output = _capture(store_with_data)
        assert "CTREND" in output
        assert "SOL/USDT" in output

    def test_render_retired_summary(self, store_with_data: StrategyStore) -> None:
        output = _capture(store_with_data)
        assert "Retired Strategies" in output

    def test_render_active_notes(self, store_with_data: StrategyStore) -> None:
        output = _capture(store_with_data)
        assert "SOL Sharpe 2.05" in output

    def test_render_empty_store(self, tmp_path: Path) -> None:
        store = StrategyStore(base_dir=tmp_path)
        output = _capture(store)
        assert "Active: 0" in output
        assert "No active strategies" in output

    def test_render_with_lessons(
        self,
        store_with_data: StrategyStore,
        tmp_path: Path,
        sample_lesson: LessonRecord,
    ) -> None:
        lesson_store = LessonStore(base_dir=tmp_path / "lessons")
        lesson_store.save(sample_lesson)
        output = _capture(store_with_data, lesson_store=lesson_store)
        assert "Lessons" in output
        assert "앙상블 > 단일지표" in output

    def test_render_without_lessons(self, store_with_data: StrategyStore) -> None:
        output = _capture(store_with_data)
        # lesson_store=None → 교훈 테이블 미출력
        assert "Lessons" not in output
