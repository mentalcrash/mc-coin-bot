"""Tests for src/pipeline/gate_models.py."""

from __future__ import annotations

from src.pipeline.gate_models import (
    ChecklistCriteria,
    ChecklistItem,
    GateCriteria,
    GateType,
    ImmediateFailRule,
    ScoringCriteria,
    ScoringItem,
    Severity,
    ThresholdCriteria,
    ThresholdMetric,
)


class TestEnums:
    def test_gate_type_values(self) -> None:
        assert GateType.SCORING == "scoring"
        assert GateType.CHECKLIST == "checklist"
        assert GateType.THRESHOLD == "threshold"

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
        try:
            item.name = "changed"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except Exception:
            pass


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


class TestGateCriteria:
    def test_scoring_gate(self) -> None:
        items = [ScoringItem(name="test", description="desc")]
        gate = GateCriteria(
            gate_id="G0A",
            name="아이디어 검증",
            gate_type=GateType.SCORING,
            scoring=ScoringCriteria(items=items),
        )
        assert gate.gate_id == "G0A"
        assert gate.gate_type == GateType.SCORING
        assert gate.scoring is not None
        assert gate.checklist is None
        assert gate.threshold is None

    def test_threshold_gate(self) -> None:
        metrics = [ThresholdMetric(name="Sharpe", operator=">", value=1.0)]
        gate = GateCriteria(
            gate_id="G1",
            name="단일에셋 백테스트",
            gate_type=GateType.THRESHOLD,
            threshold=ThresholdCriteria(pass_metrics=metrics),
        )
        assert gate.gate_type == GateType.THRESHOLD
        assert gate.threshold is not None

    def test_frozen(self) -> None:
        gate = GateCriteria(
            gate_id="G0A",
            name="test",
            gate_type=GateType.SCORING,
        )
        try:
            gate.gate_id = "G1"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except Exception:
            pass

    def test_model_dump_roundtrip(self) -> None:
        metrics = [
            ThresholdMetric(name="Sharpe", operator=">", value=1.0),
            ThresholdMetric(name="CAGR", operator=">", value=20.0, unit="%"),
        ]
        gate = GateCriteria(
            gate_id="G1",
            name="단일에셋 백테스트",
            gate_type=GateType.THRESHOLD,
            cli_command="run {config}",
            threshold=ThresholdCriteria(pass_metrics=metrics),
        )
        data = gate.model_dump(mode="json")
        restored = GateCriteria(**data)
        assert restored == gate
        assert restored.threshold is not None
        assert len(restored.threshold.pass_metrics) == 2
