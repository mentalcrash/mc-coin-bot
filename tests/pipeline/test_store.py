"""Tests for src/pipeline/store.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.models import (
    AssetMetrics,
    GateId,
    GateVerdict,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore


@pytest.fixture
def store(tmp_path: Path) -> StrategyStore:
    return StrategyStore(base_dir=tmp_path)


@pytest.fixture
def populated_store(
    store: StrategyStore,
    sample_record: StrategyRecord,
    retired_record: StrategyRecord,
) -> StrategyStore:
    """2개 전략이 저장된 store."""
    store.save(sample_record)
    store.save(retired_record)
    return store


class TestCRUD:
    def test_save_and_load(self, store: StrategyStore, sample_record: StrategyRecord) -> None:
        store.save(sample_record)
        loaded = store.load("ctrend")
        assert loaded.meta.name == "ctrend"
        assert loaded.meta.display_name == "CTREND"
        assert loaded.best_asset == "SOL/USDT"

    def test_load_nonexistent_raises(self, store: StrategyStore) -> None:
        with pytest.raises(FileNotFoundError, match="Strategy YAML not found"):
            store.load("nonexistent")

    def test_exists(self, store: StrategyStore, sample_record: StrategyRecord) -> None:
        assert not store.exists("ctrend")
        store.save(sample_record)
        assert store.exists("ctrend")

    def test_load_all(self, populated_store: StrategyStore) -> None:
        records = populated_store.load_all()
        assert len(records) == 2
        names = {r.meta.name for r in records}
        assert names == {"ctrend", "bb-rsi"}

    def test_load_all_empty(self, tmp_path: Path) -> None:
        store = StrategyStore(base_dir=tmp_path / "empty")
        assert store.load_all() == []

    def test_roundtrip_preserves_data(
        self,
        store: StrategyStore,
        sample_record: StrategyRecord,
    ) -> None:
        store.save(sample_record)
        # Clear cache
        store._cache.clear()
        loaded = store.load("ctrend")
        assert loaded.meta == sample_record.meta
        assert len(loaded.gates) == len(sample_record.gates)
        assert len(loaded.asset_performance) == len(sample_record.asset_performance)
        assert len(loaded.decisions) == len(sample_record.decisions)
        assert loaded.parameters == sample_record.parameters

    def test_gates_roundtrip(self, store: StrategyStore, sample_record: StrategyRecord) -> None:
        store.save(sample_record)
        store._cache.clear()
        loaded = store.load("ctrend")
        assert GateId.G0A in loaded.gates
        assert loaded.gates[GateId.G0A].status == GateVerdict.PASS
        assert loaded.gates[GateId.G0A].details["score"] == 22


class TestQuery:
    def test_filter_by_status(self, populated_store: StrategyStore) -> None:
        active = populated_store.filter_by_status(StrategyStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].meta.name == "ctrend"

    def test_get_active(self, populated_store: StrategyStore) -> None:
        active = populated_store.get_active()
        assert len(active) == 1

    def test_get_retired(self, populated_store: StrategyStore) -> None:
        retired = populated_store.get_retired()
        assert len(retired) == 1
        assert retired[0].meta.name == "bb-rsi"

    def test_get_at_gate(self, populated_store: StrategyStore) -> None:
        at_g2 = populated_store.get_at_gate(GateId.G2)
        assert len(at_g2) == 1
        assert at_g2[0].meta.name == "ctrend"

    def test_get_failed_at(self, populated_store: StrategyStore) -> None:
        failed_g1 = populated_store.get_failed_at(GateId.G1)
        assert len(failed_g1) == 1
        assert failed_g1[0].meta.name == "bb-rsi"


class TestMutation:
    def test_record_gate(self, store: StrategyStore, sample_record: StrategyRecord) -> None:
        store.save(sample_record)
        updated = store.record_gate(
            "ctrend",
            GateId.G3,
            GateVerdict.PASS,
            details={"plateau": True, "stability": "±20% OK"},
            rationale="4/4 파라미터 고원",
        )
        assert GateId.G3 in updated.gates
        assert updated.gates[GateId.G3].status == GateVerdict.PASS
        assert len(updated.decisions) == len(sample_record.decisions) + 1

    def test_update_status(self, store: StrategyStore, sample_record: StrategyRecord) -> None:
        store.save(sample_record)
        updated = store.update_status("ctrend", StrategyStatus.RETIRED)
        assert updated.meta.status == StrategyStatus.RETIRED
        assert updated.meta.retired_at is not None

    def test_set_asset_performance(
        self,
        store: StrategyStore,
        sample_record: StrategyRecord,
    ) -> None:
        store.save(sample_record)
        new_metrics = [
            AssetMetrics(symbol="DOGE/USDT", sharpe=0.83, cagr=29.9, mdd=64.6, trades=365),
        ]
        updated = store.set_asset_performance("ctrend", new_metrics)
        assert len(updated.asset_performance) == 1
        assert updated.best_asset == "DOGE/USDT"
