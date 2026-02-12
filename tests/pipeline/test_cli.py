"""Tests for src/cli/pipeline.py — create, update-status, record --no-retire, gates-list/show."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.pipeline import app
from src.pipeline.gate_store import GateCriteriaStore
from src.pipeline.models import GateId, GateVerdict, StrategyStatus
from src.pipeline.store import StrategyStore

runner = CliRunner()


@pytest.fixture
def strategies_dir(tmp_path: Path) -> Path:
    """Temporary strategies directory."""
    d = tmp_path / "strategies"
    d.mkdir()
    return d


@pytest.fixture
def _patch_store(strategies_dir: Path) -> None:
    """Patch StrategyStore to use tmp_path."""
    original_init = StrategyStore.__init__

    def patched_init(self: StrategyStore, base_dir: Path = strategies_dir) -> None:
        original_init(self, base_dir=base_dir)

    with patch.object(StrategyStore, "__init__", patched_init):
        yield


# ─── create command ──────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestCreateCommand:
    def test_create_basic(self, strategies_dir: Path) -> None:
        result = runner.invoke(
            app,
            [
                "create",
                "test-strat",
                "--display-name",
                "Test Strategy",
                "--category",
                "Trend Following",
                "--timeframe",
                "1D",
                "--short-mode",
                "HEDGE_ONLY",
                "--g0a-score",
                "22",
            ],
        )
        assert result.exit_code == 0
        assert "Created" in result.output

        # Verify YAML was written
        yaml_path = strategies_dir / "test-strat.yaml"
        assert yaml_path.exists()

        # Verify contents via store
        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("test-strat")
        assert record.meta.status == StrategyStatus.CANDIDATE
        assert record.meta.display_name == "Test Strategy"
        assert record.meta.category == "Trend Following"
        assert record.meta.timeframe == "1D"
        assert record.meta.short_mode == "HEDGE_ONLY"
        assert GateId.G0A in record.gates
        assert record.gates[GateId.G0A].status == GateVerdict.PASS
        assert record.gates[GateId.G0A].details["score"] == 22

    def test_create_with_rationale(self, strategies_dir: Path) -> None:
        result = runner.invoke(
            app,
            [
                "create",
                "my-strat",
                "--display-name",
                "My Strat",
                "--category",
                "Vol",
                "--timeframe",
                "4H",
                "--short-mode",
                "FULL",
                "--rationale",
                "Volatility premium harvesting",
                "--g0a-score",
                "25",
            ],
        )
        assert result.exit_code == 0

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("my-strat")
        assert record.meta.economic_rationale == "Volatility premium harvesting"
        assert len(record.decisions) == 1
        assert record.decisions[0].rationale == "25/30점"

    def test_create_duplicate_fails(self, strategies_dir: Path) -> None:
        # First create
        runner.invoke(
            app,
            [
                "create",
                "dup-strat",
                "--display-name",
                "Dup",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
            ],
        )
        # Second create with same name
        result = runner.invoke(
            app,
            [
                "create",
                "dup-strat",
                "--display-name",
                "Dup2",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
            ],
        )
        assert result.exit_code == 1
        assert "Already exists" in result.output

    def test_create_default_g0a_score(self, strategies_dir: Path) -> None:
        result = runner.invoke(
            app,
            [
                "create",
                "default-score",
                "--display-name",
                "Default",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
            ],
        )
        assert result.exit_code == 0

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("default-score")
        assert record.gates[GateId.G0A].details["score"] == 0


# ─── update-status command ───────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestUpdateStatusCommand:
    def _create_strategy(self) -> None:
        runner.invoke(
            app,
            [
                "create",
                "status-test",
                "--display-name",
                "Status Test",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
            ],
        )

    def test_update_status_candidate_to_implemented(self, strategies_dir: Path) -> None:
        self._create_strategy()
        result = runner.invoke(
            app,
            ["update-status", "status-test", "--status", "IMPLEMENTED"],
        )
        assert result.exit_code == 0
        assert "IMPLEMENTED" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("status-test")
        assert record.meta.status == StrategyStatus.IMPLEMENTED

    def test_update_status_not_found(self) -> None:
        result = runner.invoke(
            app,
            ["update-status", "nonexistent", "--status", "ACTIVE"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_update_status_to_retired_sets_date(self, strategies_dir: Path) -> None:
        self._create_strategy()
        result = runner.invoke(
            app,
            ["update-status", "status-test", "--status", "RETIRED"],
        )
        assert result.exit_code == 0

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("status-test")
        assert record.meta.status == StrategyStatus.RETIRED
        assert record.meta.retired_at is not None


# ─── record --no-retire flag ─────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestRecordNoRetire:
    def _create_implemented(self) -> None:
        runner.invoke(
            app,
            [
                "create",
                "retire-test",
                "--display-name",
                "Retire Test",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
            ],
        )
        runner.invoke(
            app,
            ["update-status", "retire-test", "--status", "IMPLEMENTED"],
        )

    def test_record_fail_default_retires(self, strategies_dir: Path) -> None:
        self._create_implemented()
        result = runner.invoke(
            app,
            [
                "record",
                "retire-test",
                "--gate",
                "G0B",
                "--verdict",
                "FAIL",
                "--rationale",
                "C1 FAIL: look-ahead",
            ],
        )
        assert result.exit_code == 0
        assert "RETIRED" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("retire-test")
        assert record.meta.status == StrategyStatus.RETIRED

    def test_record_fail_no_retire_preserves_status(self, strategies_dir: Path) -> None:
        self._create_implemented()
        result = runner.invoke(
            app,
            [
                "record",
                "retire-test",
                "--gate",
                "G0B",
                "--verdict",
                "FAIL",
                "--no-retire",
                "--rationale",
                "C1 FAIL: look-ahead",
            ],
        )
        assert result.exit_code == 0
        assert "status unchanged" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("retire-test")
        assert record.meta.status == StrategyStatus.IMPLEMENTED

    def test_record_pass_auto_transitions_to_testing(self, strategies_dir: Path) -> None:
        self._create_implemented()
        result = runner.invoke(
            app,
            [
                "record",
                "retire-test",
                "--gate",
                "G0B",
                "--verdict",
                "PASS",
                "--no-retire",
                "--rationale",
                "C1-C7 PASS",
            ],
        )
        assert result.exit_code == 0
        assert "RETIRED" not in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("retire-test")
        # IMPLEMENTED + PASS → TESTING 자동 전환
        assert record.meta.status == StrategyStatus.TESTING


# ─── gates-list / gates-show commands ────────────────────────────────

_SAMPLE_GATES_YAML = {
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
                ],
            },
        },
    ],
}


class TestGatesCommands:
    @pytest.fixture(autouse=True)
    def _patch_gate_store(self, tmp_path: Path) -> None:  # type: ignore[misc]
        """Patch GateCriteriaStore to use tmp gate yaml."""
        gate_path = tmp_path / "criteria.yaml"
        gate_path.write_text(
            yaml.dump(_SAMPLE_GATES_YAML, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        original_init = GateCriteriaStore.__init__

        def patched_init(self: GateCriteriaStore, path: Path = gate_path) -> None:
            original_init(self, path=path)

        with patch.object(GateCriteriaStore, "__init__", patched_init):
            yield

    def test_gates_list(self) -> None:
        result = runner.invoke(app, ["gates-list"])
        assert result.exit_code == 0
        assert "G0A" in result.output
        assert "G1" in result.output
        assert "아이디어 검증" in result.output

    def test_gates_show_scoring(self) -> None:
        result = runner.invoke(app, ["gates-show", "G0A"])
        assert result.exit_code == 0
        assert "아이디어 검증" in result.output
        assert "18" in result.output

    def test_gates_show_threshold(self) -> None:
        result = runner.invoke(app, ["gates-show", "G1"])
        assert result.exit_code == 0
        assert "Sharpe" in result.output
        assert "1" in result.output

    def test_gates_show_not_found(self) -> None:
        result = runner.invoke(app, ["gates-show", "G99"])
        assert result.exit_code == 1
        assert "not found" in result.output
