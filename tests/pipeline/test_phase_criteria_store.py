"""Tests for src/pipeline/phase_criteria_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipeline.phase_criteria_store import PhaseCriteriaStore

_SAMPLE_YAML = {
    "phases": [
        {
            "phase_id": "P1",
            "name": "Alpha Research",
            "phase_type": "scoring",
            "scoring": {
                "pass_threshold": 18,
                "max_total": 30,
                "items": [
                    {"name": "경제적 논거", "description": "5=행동편향"},
                    {"name": "참신성", "description": "5=미공개"},
                ],
            },
        },
        {
            "phase_id": "P4",
            "name": "Backtest",
            "phase_type": "threshold",
            "cli_command": "run {config}",
            "threshold": {
                "pass_metrics": [
                    {"name": "Sharpe", "operator": ">", "value": 1.0},
                    {"name": "CAGR", "operator": ">", "value": 20.0, "unit": "%"},
                    {"name": "MDD", "operator": "<", "value": 40.0, "unit": "%"},
                    {"name": "Trades", "operator": ">", "value": 50},
                ],
                "immediate_fail": [
                    {"condition": "MDD > 50%", "reason": "파산 위험"},
                ],
            },
        },
    ],
}


@pytest.fixture
def phase_yaml_path(tmp_path: Path) -> Path:
    """임시 phase-criteria.yaml 작성."""
    path = tmp_path / "phase-criteria.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def phase_store(phase_yaml_path: Path) -> PhaseCriteriaStore:
    return PhaseCriteriaStore(path=phase_yaml_path)


class TestPhaseCriteriaStore:
    def test_load_all_count(self, phase_store: PhaseCriteriaStore) -> None:
        phases = phase_store.load_all()
        assert len(phases) == 2

    def test_load_all_order(self, phase_store: PhaseCriteriaStore) -> None:
        phases = phase_store.load_all()
        assert phases[0].phase_id == "P1"
        assert phases[1].phase_id == "P4"

    def test_load_single(self, phase_store: PhaseCriteriaStore) -> None:
        p4 = phase_store.load("P4")
        assert p4.name == "Backtest"
        assert p4.threshold is not None
        assert len(p4.threshold.pass_metrics) == 4

    def test_load_scoring_phase(self, phase_store: PhaseCriteriaStore) -> None:
        p1 = phase_store.load("P1")
        assert p1.scoring is not None
        assert p1.scoring.pass_threshold == 18
        assert len(p1.scoring.items) == 2

    def test_load_not_found(self, phase_store: PhaseCriteriaStore) -> None:
        with pytest.raises(KeyError, match="P99"):
            phase_store.load("P99")

    def test_file_not_found(self, tmp_path: Path) -> None:
        store = PhaseCriteriaStore(path=tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            store.load_all()

    def test_get_pass_thresholds(self, phase_store: PhaseCriteriaStore) -> None:
        thresholds = phase_store.get_pass_thresholds("P4")
        assert "Sharpe" in thresholds
        assert thresholds["Sharpe"] == (">", 1.0)
        assert thresholds["CAGR"] == (">", 20.0)
        assert thresholds["MDD"] == ("<", 40.0)
        assert thresholds["Trades"] == (">", 50)

    def test_get_pass_thresholds_non_threshold_phase(self, phase_store: PhaseCriteriaStore) -> None:
        thresholds = phase_store.get_pass_thresholds("P1")
        assert thresholds == {}

    def test_cache_hit(self, phase_store: PhaseCriteriaStore) -> None:
        phase_store.load_all()
        # Second call should use cache
        assert phase_store._cache is not None
        phases = phase_store.load_all()
        assert len(phases) == 2

    def test_immediate_fail_loaded(self, phase_store: PhaseCriteriaStore) -> None:
        p4 = phase_store.load("P4")
        assert p4.threshold is not None
        assert len(p4.threshold.immediate_fail) == 1
        assert p4.threshold.immediate_fail[0].reason == "파산 위험"


class TestRealCriteriaYaml:
    """실제 gates/phase-criteria.yaml 로드 테스트."""

    def test_load_real_yaml(self) -> None:
        real_path = Path("gates/phase-criteria.yaml")
        if not real_path.exists():
            pytest.skip("gates/phase-criteria.yaml not found")
        store = PhaseCriteriaStore(path=real_path)
        phases = store.load_all()
        assert len(phases) == 7

    def test_real_yaml_phase_ids(self) -> None:
        real_path = Path("gates/phase-criteria.yaml")
        if not real_path.exists():
            pytest.skip("gates/phase-criteria.yaml not found")
        store = PhaseCriteriaStore(path=real_path)
        phases = store.load_all()
        ids = [p.phase_id for p in phases]
        assert ids == ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
