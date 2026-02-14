"""Tests for src/pipeline/gate_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipeline.gate_store import GateCriteriaStore

_SAMPLE_YAML = {
    "gates": [
        {
            "gate_id": "G0A",
            "name": "아이디어 검증",
            "gate_type": "scoring",
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
            "gate_id": "G1",
            "name": "단일에셋 백테스트",
            "gate_type": "threshold",
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
def gate_yaml_path(tmp_path: Path) -> Path:
    """임시 criteria.yaml 작성."""
    path = tmp_path / "criteria.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def gate_store(gate_yaml_path: Path) -> GateCriteriaStore:
    return GateCriteriaStore(path=gate_yaml_path)


class TestGateCriteriaStore:
    def test_load_all_count(self, gate_store: GateCriteriaStore) -> None:
        gates = gate_store.load_all()
        assert len(gates) == 2

    def test_load_all_order(self, gate_store: GateCriteriaStore) -> None:
        gates = gate_store.load_all()
        assert gates[0].gate_id == "G0A"
        assert gates[1].gate_id == "G1"

    def test_load_single(self, gate_store: GateCriteriaStore) -> None:
        g1 = gate_store.load("G1")
        assert g1.name == "단일에셋 백테스트"
        assert g1.threshold is not None
        assert len(g1.threshold.pass_metrics) == 4

    def test_load_scoring_gate(self, gate_store: GateCriteriaStore) -> None:
        g0a = gate_store.load("G0A")
        assert g0a.scoring is not None
        assert g0a.scoring.pass_threshold == 18
        assert len(g0a.scoring.items) == 2

    def test_load_not_found(self, gate_store: GateCriteriaStore) -> None:
        with pytest.raises(KeyError, match="G99"):
            gate_store.load("G99")

    def test_file_not_found(self, tmp_path: Path) -> None:
        store = GateCriteriaStore(path=tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            store.load_all()

    def test_get_pass_thresholds(self, gate_store: GateCriteriaStore) -> None:
        thresholds = gate_store.get_pass_thresholds("G1")
        assert "Sharpe" in thresholds
        assert thresholds["Sharpe"] == (">", 1.0)
        assert thresholds["CAGR"] == (">", 20.0)
        assert thresholds["MDD"] == ("<", 40.0)
        assert thresholds["Trades"] == (">", 50)

    def test_get_pass_thresholds_non_threshold_gate(self, gate_store: GateCriteriaStore) -> None:
        thresholds = gate_store.get_pass_thresholds("G0A")
        assert thresholds == {}

    def test_cache_hit(self, gate_store: GateCriteriaStore) -> None:
        gate_store.load_all()
        # Second call should use cache
        assert gate_store._cache is not None
        gates = gate_store.load_all()
        assert len(gates) == 2

    def test_immediate_fail_loaded(self, gate_store: GateCriteriaStore) -> None:
        g1 = gate_store.load("G1")
        assert g1.threshold is not None
        assert len(g1.threshold.immediate_fail) == 1
        assert g1.threshold.immediate_fail[0].reason == "파산 위험"


class TestRealCriteriaYaml:
    """실제 gates/criteria.yaml 로드 테스트."""

    def test_load_real_yaml(self) -> None:
        real_path = Path("gates/criteria.yaml")
        if not real_path.exists():
            pytest.skip("gates/criteria.yaml not found")
        store = GateCriteriaStore(path=real_path)
        gates = store.load_all()
        assert len(gates) == 8

    def test_real_yaml_gate_ids(self) -> None:
        real_path = Path("gates/criteria.yaml")
        if not real_path.exists():
            pytest.skip("gates/criteria.yaml not found")
        store = GateCriteriaStore(path=real_path)
        gates = store.load_all()
        ids = [g.gate_id for g in gates]
        assert ids == ["G0A", "G0B", "G1", "G2", "G2H", "G3", "G4", "G5"]
