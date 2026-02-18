"""Tests for src/catalog/failure_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.catalog.failure_models import Frequency
from src.catalog.failure_store import FailurePatternStore

_SAMPLE_YAML = {
    "patterns": [
        {
            "id": "cost_erosion",
            "name": "거래비용 잠식",
            "description": "4H 이하 TF에서 비용 잠식",
            "frequency": "high",
            "affected_gates": ["G1", "G2"],
            "detection_rules": [
                {"metric": "total_trades", "operator": ">", "threshold": 500},
            ],
            "prevention": ["1D TF 우선"],
            "related_lessons": [3, 15],
            "examples": ["rsi-regime-4h"],
        },
        {
            "id": "oos_collapse",
            "name": "IS/OOS 붕괴",
            "description": "Out-of-Sample 성과 급락",
            "frequency": "high",
            "affected_gates": ["G2"],
            "prevention": ["Walk-Forward 검증"],
            "examples": ["kama-1d", "max-min-1d"],
        },
        {
            "id": "low_trade_count",
            "name": "극저빈도 거래",
            "description": "거래 횟수 부족",
            "frequency": "medium",
            "affected_gates": ["G1"],
            "prevention": ["최소 연간 10건"],
            "examples": ["squeeze-breakout-1d"],
        },
    ]
}


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    path = tmp_path / "failure_patterns.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def store(store_path: Path) -> FailurePatternStore:
    return FailurePatternStore(path=store_path)


class TestFailurePatternStore:
    def test_load_all(self, store: FailurePatternStore) -> None:
        patterns = store.load_all()
        assert len(patterns) == 3

    def test_load_single(self, store: FailurePatternStore) -> None:
        p = store.load("cost_erosion")
        assert p.name == "거래비용 잠식"
        assert p.frequency == Frequency.HIGH
        assert "G1" in p.affected_gates

    def test_load_not_found(self, store: FailurePatternStore) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            store.load("nonexistent")

    def test_file_not_found(self, tmp_path: Path) -> None:
        s = FailurePatternStore(path=tmp_path / "missing.yaml")
        with pytest.raises(FileNotFoundError):
            s.load_all()

    def test_filter_by_gate(self, store: FailurePatternStore) -> None:
        g1_patterns = store.filter_by_gate("G1")
        assert len(g1_patterns) == 2
        ids = [p.id for p in g1_patterns]
        assert "cost_erosion" in ids
        assert "low_trade_count" in ids

    def test_filter_by_gate_g2(self, store: FailurePatternStore) -> None:
        g2_patterns = store.filter_by_gate("G2")
        assert len(g2_patterns) == 2

    def test_filter_by_frequency_high(self, store: FailurePatternStore) -> None:
        high = store.filter_by_frequency("high")
        assert len(high) == 2
        assert all(p.frequency == Frequency.HIGH for p in high)

    def test_filter_by_frequency_medium(self, store: FailurePatternStore) -> None:
        med = store.filter_by_frequency("medium")
        assert len(med) == 1
        assert med[0].id == "low_trade_count"

    def test_detection_rules(self, store: FailurePatternStore) -> None:
        p = store.load("cost_erosion")
        assert len(p.detection_rules) == 1
        rule = p.detection_rules[0]
        assert rule.metric == "total_trades"
        assert rule.operator == ">"
        assert rule.threshold == 500

    def test_cache_hit(self, store: FailurePatternStore) -> None:
        store.load_all()
        assert store._catalog is not None
        patterns = store.load_all()
        assert len(patterns) == 3


class TestRealFailurePatterns:
    """실제 catalogs/failure_patterns.yaml 로드 테스트."""

    def test_load_real_yaml(self) -> None:
        real_path = Path("catalogs/failure_patterns.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/failure_patterns.yaml not found")
        store = FailurePatternStore(path=real_path)
        patterns = store.load_all()
        assert len(patterns) >= 7

    def test_real_yaml_known_patterns(self) -> None:
        real_path = Path("catalogs/failure_patterns.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/failure_patterns.yaml not found")
        store = FailurePatternStore(path=real_path)
        ids = [p.id for p in store.load_all()]
        assert "cost_erosion" in ids
        assert "oos_collapse" in ids
        assert "btc_eth_negative" in ids
