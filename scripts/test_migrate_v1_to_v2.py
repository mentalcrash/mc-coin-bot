"""Tests for scripts/migrate_v1_to_v2.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path

from scripts.migrate_v1_to_v2 import main, migrate_file

_V1_YAML = {
    "meta": {
        "name": "test-strat",
        "display_name": "Test Strategy",
        "category": "Test",
        "timeframe": "1D",
        "short_mode": "DISABLED",
        "status": "CANDIDATE",
        "created_at": "2026-01-01",
        "retired_at": None,
    },
    "parameters": {"lookback": 20},
    "gates": {
        "G0A": {
            "status": "PASS",
            "date": "2026-01-01",
            "details": {"score": 22},
        },
        "G1": {
            "status": "PASS",
            "date": "2026-01-10",
            "details": {"sharpe": 1.5, "cagr": 50.0},
        },
        "G2": {
            "status": "PASS",
            "date": "2026-01-15",
            "details": {"oos_sharpe": 1.2},
        },
    },
    "decisions": [
        {
            "date": "2026-01-01",
            "gate": "G0A",
            "verdict": "PASS",
            "rationale": "22/30점",
        },
        {
            "date": "2026-01-10",
            "gate": "G1",
            "verdict": "PASS",
            "rationale": "Sharpe 1.5",
        },
    ],
    "asset_performance": [],
}


def _write_v1_yaml(path: Path, data: dict | None = None) -> None:
    """Write v1 YAML to path."""
    content = data or _V1_YAML
    path.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


class TestMigrateFile:
    def test_migrates_v1_to_v2(self, tmp_path: Path) -> None:
        """v1 YAML → v2 변환."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        result = migrate_file(yaml_path)
        assert result is True

        # Read back and verify v2 format
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert "phases" in raw
        assert "gates" not in raw
        assert "version" in raw
        assert raw["version"] == 2

    def test_gates_merged_correctly(self, tmp_path: Path) -> None:
        """G1+G2 → P4 merge."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        migrate_file(yaml_path)

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        phases = raw["phases"]
        assert "P1" in phases  # G0A → P1
        assert "P4" in phases  # G1+G2 → P4
        assert "G0A" not in phases
        assert "G1" not in phases
        assert "G2" not in phases

    def test_decisions_converted(self, tmp_path: Path) -> None:
        """decisions[].gate → decisions[].phase."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        migrate_file(yaml_path)

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        for d in raw.get("decisions", []):
            assert "phase" in d
            assert "gate" not in d

    def test_backup_created(self, tmp_path: Path) -> None:
        """기본: .bak 파일 생성."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        migrate_file(yaml_path, backup=True)

        bak_path = yaml_path.with_suffix(".yaml.bak")
        assert bak_path.exists()

    def test_no_backup_flag(self, tmp_path: Path) -> None:
        """backup=False → .bak 미생성."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        migrate_file(yaml_path, backup=False)

        bak_path = yaml_path.with_suffix(".yaml.bak")
        assert not bak_path.exists()

    def test_dry_run_no_modification(self, tmp_path: Path) -> None:
        """dry_run=True → 파일 미변경."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)
        original_content = yaml_path.read_text(encoding="utf-8")

        result = migrate_file(yaml_path, dry_run=True)
        assert result is True
        assert yaml_path.read_text(encoding="utf-8") == original_content

    def test_skip_already_v2(self, tmp_path: Path) -> None:
        """이미 v2 파일 → skip."""
        yaml_path = tmp_path / "v2-strat.yaml"
        v2_data = {
            "meta": _V1_YAML["meta"],
            "parameters": {},
            "phases": {"P1": {"status": "PASS", "date": "2026-01-01", "details": {}}},
            "decisions": [],
            "asset_performance": [],
            "version": 2,
        }
        yaml_path.write_text(
            yaml.dump(v2_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

        result = migrate_file(yaml_path)
        assert result is False

    def test_g2h_g3_merged_to_p5(self, tmp_path: Path) -> None:
        """G2H+G3 → P5 merge."""
        data = dict(_V1_YAML)
        data = {**data}
        data["gates"] = {
            **_V1_YAML["gates"],
            "G2H": {
                "status": "PASS",
                "date": "2026-01-20",
                "details": {"best_sharpe_is": 2.0},
            },
            "G3": {
                "status": "PASS",
                "date": "2026-01-25",
                "details": {"plateau": True},
            },
        }
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path, data)

        migrate_file(yaml_path)

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        phases = raw["phases"]
        assert "P5" in phases  # G2H+G3 → P5
        assert "G2H" not in phases
        assert "G3" not in phases


class TestMainCLI:
    def test_main_dry_run(self, tmp_path: Path) -> None:
        """main --dry-run 동작."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        exit_code = main(["--dir", str(tmp_path), "--dry-run"])
        assert exit_code == 0

        # File should be unchanged
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert "gates" in raw

    def test_main_migrate(self, tmp_path: Path) -> None:
        """main 실제 마이그레이션."""
        yaml_path = tmp_path / "test-strat.yaml"
        _write_v1_yaml(yaml_path)

        exit_code = main(["--dir", str(tmp_path), "--no-backup"])
        assert exit_code == 0

        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert "phases" in raw
        assert "gates" not in raw

    def test_main_empty_dir(self, tmp_path: Path) -> None:
        """빈 디렉토리 → 0."""
        exit_code = main(["--dir", str(tmp_path)])
        assert exit_code == 0

    def test_main_nonexistent_dir(self, tmp_path: Path) -> None:
        """존재하지 않는 디렉토리 → 1."""
        exit_code = main(["--dir", str(tmp_path / "nonexistent")])
        assert exit_code == 1
