"""Tests for src/cli/pipeline.py — create, update-status, record --no-retire, phases-list/show."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.pipeline import app
from src.pipeline.models import PhaseId, PhaseVerdict, StrategyStatus
from src.pipeline.phase_criteria_store import PhaseCriteriaStore
from src.pipeline.store import StrategyStore

runner = CliRunner()

# P1 v2 기준 (criteria.yaml와 동기화)
_P1_V2_ITEMS = [
    "경제적 논거 고유성",
    "IC 사전 검증",
    "카테고리 성공률",
    "레짐 독립성",
    "앙상블 기여도",
    "수용 용량",
]
_P1_V2_PASS_THRESHOLD = 21
_P1_V2_MAX_TOTAL = 30


@pytest.fixture
def strategies_dir(tmp_path: Path) -> Path:
    """Temporary strategies directory."""
    d = tmp_path / "strategies"
    d.mkdir()
    return d


@pytest.fixture
def _patch_store(strategies_dir: Path, tmp_path: Path) -> None:
    """Patch StrategyStore and PhaseCriteriaStore to use tmp_path."""
    original_init = StrategyStore.__init__

    def patched_init(self: StrategyStore, base_dir: Path = strategies_dir) -> None:
        original_init(self, base_dir=base_dir)

    # P1 v2 criteria YAML for _load_p1_criteria()
    phase_path = tmp_path / "p1_criteria.yaml"
    phase_yaml = {
        "phases": [
            {
                "phase_id": "P1",
                "name": "아이디어 검증",
                "phase_type": "scoring",
                "scoring": {
                    "pass_threshold": _P1_V2_PASS_THRESHOLD,
                    "max_total": _P1_V2_MAX_TOTAL,
                    "items": [{"name": n, "description": "test"} for n in _P1_V2_ITEMS],
                },
            },
        ],
    }
    phase_path.write_text(
        yaml.dump(phase_yaml, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    original_phase_init = PhaseCriteriaStore.__init__

    def patched_phase_init(self: PhaseCriteriaStore, path: Path = phase_path) -> None:
        original_phase_init(self, path=path)

    with (
        patch.object(StrategyStore, "__init__", patched_init),
        patch.object(PhaseCriteriaStore, "__init__", patched_phase_init),
    ):
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
                "--p1-score",
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
        assert PhaseId.P1 in record.phases
        assert record.phases[PhaseId.P1].status == PhaseVerdict.PASS
        assert record.phases[PhaseId.P1].details["score"] == 22

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
                "--p1-score",
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

    def test_create_default_p1_score_fails(self, strategies_dir: Path) -> None:
        """score=0 (default) -> P1 FAIL verdict."""
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
        assert "P1 FAIL" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("default-score")
        assert record.phases[PhaseId.P1].details["score"] == 0
        assert record.phases[PhaseId.P1].status == PhaseVerdict.FAIL

    def test_create_p1_below_threshold(self, strategies_dir: Path) -> None:
        """score=15 < 21 -> FAIL verdict, CANDIDATE 유지."""
        result = runner.invoke(
            app,
            [
                "create",
                "low-score",
                "--display-name",
                "Low Score",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-score",
                "15",
            ],
        )
        assert result.exit_code == 0
        assert "P1 FAIL" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("low-score")
        assert record.phases[PhaseId.P1].status == PhaseVerdict.FAIL
        assert record.meta.status == StrategyStatus.CANDIDATE
        assert record.decisions[0].verdict == PhaseVerdict.FAIL

    def test_create_p1_at_threshold(self, strategies_dir: Path) -> None:
        """score=21 -> PASS (v2 threshold)."""
        result = runner.invoke(
            app,
            [
                "create",
                "threshold-score",
                "--display-name",
                "Threshold",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-score",
                "21",
            ],
        )
        assert result.exit_code == 0
        assert "P1 FAIL" not in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("threshold-score")
        assert record.phases[PhaseId.P1].status == PhaseVerdict.PASS

    def test_create_p1_invalid_range(self) -> None:
        """score=35 > 30 -> exit(1)."""
        result = runner.invoke(
            app,
            [
                "create",
                "invalid-score",
                "--display-name",
                "Invalid",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-score",
                "35",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid P1 score" in result.output

    def test_create_p1_negative_range(self) -> None:
        """score=-5 < 0 -> exit(1)."""
        result = runner.invoke(
            app,
            [
                "create",
                "neg-score",
                "--display-name",
                "Neg",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-score",
                "-5",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid P1 score" in result.output

    def test_create_with_rationale_category(self, strategies_dir: Path) -> None:
        """--rationale-category 옵션 동작 확인."""
        result = runner.invoke(
            app,
            [
                "create",
                "cat-strat",
                "--display-name",
                "Cat Strat",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-score",
                "22",
                "--rationale-category",
                "momentum",
            ],
        )
        assert result.exit_code == 0

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("cat-strat")
        assert record.meta.rationale_category == "momentum"

    # ─── v2 --p1-items 테스트 ─────────────────────────────────────

    def test_create_with_p1_items(self, strategies_dir: Path) -> None:
        """--p1-items JSON -> v2 details 저장."""
        items = dict.fromkeys(_P1_V2_ITEMS, 4)
        items_json = json.dumps(items, ensure_ascii=False)
        result = runner.invoke(
            app,
            [
                "create",
                "v2-strat",
                "--display-name",
                "V2 Strategy",
                "--category",
                "Trend",
                "--timeframe",
                "1D",
                "--short-mode",
                "FULL",
                "--p1-items",
                items_json,
            ],
        )
        assert result.exit_code == 0
        assert "Created" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("v2-strat")
        details = record.phases[PhaseId.P1].details
        assert details["version"] == 2
        assert details["score"] == 24  # 6 items x 4
        assert details["max_score"] == 30
        assert details["items"]["IC 사전 검증"] == 4
        assert record.phases[PhaseId.P1].status == PhaseVerdict.PASS

    def test_create_p1_items_below_threshold(self, strategies_dir: Path) -> None:
        """--p1-items 합계 < 21 -> FAIL."""
        items = dict.fromkeys(_P1_V2_ITEMS, 2)  # 6 x 2 = 12 < 21
        items_json = json.dumps(items, ensure_ascii=False)
        result = runner.invoke(
            app,
            [
                "create",
                "v2-low",
                "--display-name",
                "V2 Low",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-items",
                items_json,
            ],
        )
        assert result.exit_code == 0
        assert "P1 FAIL" in result.output

        store = StrategyStore(base_dir=strategies_dir)
        record = store.load("v2-low")
        assert record.phases[PhaseId.P1].status == PhaseVerdict.FAIL
        assert record.phases[PhaseId.P1].details["version"] == 2

    def test_create_p1_items_invalid_name(self) -> None:
        """잘못된 항목명 -> exit(1)."""
        items = {"WRONG_ITEM": 3}
        result = runner.invoke(
            app,
            [
                "create",
                "v2-bad-name",
                "--display-name",
                "Bad",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-items",
                json.dumps(items, ensure_ascii=False),
            ],
        )
        assert result.exit_code == 1
        assert "불일치" in result.output

    def test_create_p1_items_missing_item(self) -> None:
        """항목 누락 -> exit(1)."""
        items = dict.fromkeys(_P1_V2_ITEMS[:4], 3)  # 4 of 6
        result = runner.invoke(
            app,
            [
                "create",
                "v2-missing",
                "--display-name",
                "Missing",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-items",
                json.dumps(items, ensure_ascii=False),
            ],
        )
        assert result.exit_code == 1
        assert "누락" in result.output

    def test_create_p1_items_invalid_range(self) -> None:
        """점수 범위 초과 -> exit(1)."""
        items = dict.fromkeys(_P1_V2_ITEMS, 3)
        items[_P1_V2_ITEMS[0]] = 6  # > 5
        result = runner.invoke(
            app,
            [
                "create",
                "v2-range",
                "--display-name",
                "Range",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-items",
                json.dumps(items, ensure_ascii=False),
            ],
        )
        assert result.exit_code == 1
        assert "Invalid score" in result.output

    def test_create_p1_items_invalid_json(self) -> None:
        """잘못된 JSON -> exit(1)."""
        result = runner.invoke(
            app,
            [
                "create",
                "v2-badjson",
                "--display-name",
                "Bad JSON",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-items",
                "not-valid-json",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output


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
                "--phase",
                "P3",
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
                "--phase",
                "P3",
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
                "--phase",
                "P3",
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
        # IMPLEMENTED + PASS -> TESTING 자동 전환
        assert record.meta.status == StrategyStatus.TESTING


# ─── phases-list / phases-show commands ────────────────────────────────

_SAMPLE_PHASES_YAML = {
    "phases": [
        {
            "phase_id": "P1",
            "name": "아이디어 검증",
            "phase_type": "scoring",
            "scoring": {
                "pass_threshold": 18,
                "max_total": 30,
                "items": [
                    {"name": "경제적 논거", "description": "5=행동편향"},
                ],
            },
        },
        {
            "phase_id": "P4",
            "name": "단일에셋 백테스트",
            "phase_type": "threshold",
            "cli_command": "run {config}",
            "threshold": {
                "pass_metrics": [
                    {"name": "Sharpe", "operator": ">", "value": 1.0},
                ],
            },
        },
    ],
}


class TestPhasesCommands:
    @pytest.fixture(autouse=True)
    def _patch_phase_store(self, tmp_path: Path) -> None:  # type: ignore[misc]
        """Patch PhaseCriteriaStore to use tmp phase yaml."""
        phase_path = tmp_path / "criteria.yaml"
        phase_path.write_text(
            yaml.dump(_SAMPLE_PHASES_YAML, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        original_init = PhaseCriteriaStore.__init__

        def patched_init(self: PhaseCriteriaStore, path: Path = phase_path) -> None:
            original_init(self, path=path)

        with patch.object(PhaseCriteriaStore, "__init__", patched_init):
            yield

    def test_phases_list(self) -> None:
        result = runner.invoke(app, ["phases-list"])
        assert result.exit_code == 0
        assert "P1" in result.output
        assert "P4" in result.output
        assert "아이디어 검증" in result.output

    def test_phases_show_scoring(self) -> None:
        result = runner.invoke(app, ["phases-show", "P1"])
        assert result.exit_code == 0
        assert "아이디어 검증" in result.output
        assert "18" in result.output

    def test_phases_show_threshold(self) -> None:
        result = runner.invoke(app, ["phases-show", "P4"])
        assert result.exit_code == 0
        assert "Sharpe" in result.output
        assert "1" in result.output

    def test_phases_show_not_found(self) -> None:
        result = runner.invoke(app, ["phases-show", "P99"])
        assert result.exit_code == 1
        assert "not found" in result.output


# ─── retired-analysis command ────────────────────────────────────────


@pytest.mark.usefixtures("_patch_store")
class TestRetiredAnalysis:
    def _create_strategy(
        self,
        name: str,
        *,
        p1_score: int = 22,
        rationale_category: str | None = None,
    ) -> None:
        args = [
            "create",
            name,
            "--display-name",
            name,
            "--category",
            "Test",
            "--timeframe",
            "1D",
            "--short-mode",
            "DISABLED",
            "--p1-score",
            str(p1_score),
        ]
        if rationale_category:
            args.extend(["--rationale-category", rationale_category])
        runner.invoke(app, args)

    def _retire_with_phase(self, name: str, phase: str) -> None:
        runner.invoke(
            app,
            ["update-status", name, "--status", "IMPLEMENTED"],
        )
        runner.invoke(
            app,
            ["record", name, "--phase", phase, "--verdict", "FAIL", "--rationale", "test"],
        )

    def test_retired_analysis_phase_distribution(self, strategies_dir: Path) -> None:
        """Phase별 FAIL 분포 출력."""
        # P4 FAIL 2개, P4 FAIL 1개 (동일 phase)
        for i in range(3):
            self._create_strategy(f"ret-{i}", rationale_category="momentum")
        self._retire_with_phase("ret-0", "P4")
        self._retire_with_phase("ret-1", "P4")
        self._retire_with_phase("ret-2", "P4")

        result = runner.invoke(app, ["retired-analysis"])
        assert result.exit_code == 0
        assert "P4" in result.output

    def test_retired_analysis_category_stats(self, strategies_dir: Path) -> None:
        """카테고리별 집계 출력."""
        self._create_strategy("cat-a", rationale_category="momentum")
        self._retire_with_phase("cat-a", "P4")

        result = runner.invoke(app, ["retired-analysis"])
        assert result.exit_code == 0
        assert "momentum" in result.output

    def test_retired_analysis_empty(self) -> None:
        """RETIRED 0개 시 메시지."""
        result = runner.invoke(app, ["retired-analysis"])
        assert result.exit_code == 0
        assert "No RETIRED" in result.output

    def test_create_warns_low_category_success(self, strategies_dir: Path) -> None:
        """동일 category RETIRED 존재 시 경고 출력."""
        # RETIRED 전략 생성
        self._create_strategy("old-mom", p1_score=20, rationale_category="vol-premium")
        self._retire_with_phase("old-mom", "P4")

        # 새 전략 생성 — 경고 기대
        result = runner.invoke(
            app,
            [
                "create",
                "new-mom",
                "--display-name",
                "New Mom",
                "--category",
                "Test",
                "--timeframe",
                "1D",
                "--short-mode",
                "DISABLED",
                "--p1-score",
                "20",
                "--rationale-category",
                "vol-premium",
            ],
        )
        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert "vol-premium" in result.output
