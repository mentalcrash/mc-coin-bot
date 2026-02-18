"""Tests for src/pipeline/phase_criteria_models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pipeline.phase_criteria_models import (
    ChecklistCriteria,
    ChecklistItem,
    ImmediateFailRule,
    PhaseCriteria,
    PhaseType,
    ScoringCriteria,
    ScoringItem,
    Severity,
    ThresholdCriteria,
    ThresholdMetric,
)


class TestEnums:
    def test_phase_type_values(self) -> None:
        assert PhaseType.SCORING == "scoring"
        assert PhaseType.CHECKLIST == "checklist"
        assert PhaseType.THRESHOLD == "threshold"

    def test_severity_values(self) -> None:
        assert Severity.CRITICAL == "critical"
        assert Severity.WARNING == "warning"


class TestScoringModels:
    def test_scoring_item_create(self) -> None:
        item = ScoringItem(name="경제적 논거", description="5=행동편향으로 설명 가능")
        assert item.name == "경제적 논거"
        assert item.min_score == 1
        assert item.max_score == 5

    def test_scoring_criteria_defaults(self) -> None:
        items = [ScoringItem(name="test", description="desc")]
        criteria = ScoringCriteria(items=items)
        assert criteria.pass_threshold == 18
        assert criteria.max_total == 30

    def test_scoring_item_frozen(self) -> None:
        item = ScoringItem(name="test", description="desc")
        with pytest.raises((ValidationError, TypeError)):
            item.name = "changed"  # type: ignore[misc]


class TestChecklistModels:
    def test_checklist_item_create(self) -> None:
        item = ChecklistItem(
            code="C1",
            name="Look-Ahead Bias",
            description="미래 데이터 사용 여부",
            severity=Severity.CRITICAL,
        )
        assert item.code == "C1"
        assert item.severity == Severity.CRITICAL

    def test_checklist_criteria(self) -> None:
        items = [
            ChecklistItem(code="C1", name="Test", description="desc", severity=Severity.CRITICAL),
            ChecklistItem(code="W1", name="Warn", description="desc", severity=Severity.WARNING),
        ]
        criteria = ChecklistCriteria(items=items, pass_rule="Critical 0개")
        assert len(criteria.items) == 2
        assert criteria.pass_rule == "Critical 0개"


class TestThresholdModels:
    def test_threshold_metric_create(self) -> None:
        m = ThresholdMetric(name="Sharpe", operator=">", value=1.0)
        assert m.name == "Sharpe"
        assert m.operator == ">"
        assert m.value == 1.0
        assert m.unit == ""

    def test_threshold_metric_with_unit(self) -> None:
        m = ThresholdMetric(name="CAGR", operator=">", value=20.0, unit="%")
        assert m.unit == "%"

    def test_immediate_fail_rule(self) -> None:
        rule = ImmediateFailRule(condition="MDD > 50%", reason="파산 위험")
        assert rule.condition == "MDD > 50%"

    def test_threshold_criteria(self) -> None:
        metrics = [ThresholdMetric(name="Sharpe", operator=">", value=1.0)]
        criteria = ThresholdCriteria(pass_metrics=metrics)
        assert len(criteria.pass_metrics) == 1
        assert criteria.auxiliary_metrics == []
        assert criteria.immediate_fail == []


class TestPhaseCriteria:
    def test_scoring_phase(self) -> None:
        items = [ScoringItem(name="test", description="desc")]
        phase = PhaseCriteria(
            phase_id="P1",
            name="Alpha Research",
            phase_type=PhaseType.SCORING,
            scoring=ScoringCriteria(items=items),
        )
        assert phase.phase_id == "P1"
        assert phase.phase_type == PhaseType.SCORING
        assert phase.scoring is not None
        assert phase.checklist is None
        assert phase.threshold is None

    def test_threshold_phase(self) -> None:
        metrics = [ThresholdMetric(name="Sharpe", operator=">", value=1.0)]
        phase = PhaseCriteria(
            phase_id="P4",
            name="Backtest",
            phase_type=PhaseType.THRESHOLD,
            threshold=ThresholdCriteria(pass_metrics=metrics),
        )
        assert phase.phase_type == PhaseType.THRESHOLD
        assert phase.threshold is not None

    def test_frozen(self) -> None:
        phase = PhaseCriteria(
            phase_id="P1",
            name="test",
            phase_type=PhaseType.SCORING,
        )
        with pytest.raises((ValidationError, TypeError)):
            phase.phase_id = "P2"  # type: ignore[misc]

    def test_model_dump_roundtrip(self) -> None:
        metrics = [
            ThresholdMetric(name="Sharpe", operator=">", value=1.0),
            ThresholdMetric(name="CAGR", operator=">", value=20.0, unit="%"),
        ]
        phase = PhaseCriteria(
            phase_id="P4",
            name="Backtest",
            phase_type=PhaseType.THRESHOLD,
            cli_command="run {config}",
            threshold=ThresholdCriteria(pass_metrics=metrics),
        )
        data = phase.model_dump(mode="json")
        restored = PhaseCriteria(**data)
        assert restored == phase
        assert restored.threshold is not None
        assert len(restored.threshold.pass_metrics) == 2
