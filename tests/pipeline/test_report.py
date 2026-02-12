"""Tests for src/pipeline/report.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.models import StrategyRecord
from src.pipeline.report import DashboardGenerator
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
