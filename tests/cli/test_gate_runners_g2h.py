"""Tests for src/cli/_gate_runners_g2h.py — Gate 2H runner."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.models import (
    AssetMetrics,
    GateId,
    GateResult,
    GateVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore


def _make_record(name: str = "kama") -> StrategyRecord:
    return StrategyRecord(
        meta=StrategyMeta(
            name=name,
            display_name="KAMA",
            category="Trend Following",
            timeframe="1D",
            short_mode="HEDGE_ONLY",
            status=StrategyStatus.TESTING,
            created_at=date(2026, 1, 1),
        ),
        parameters={"er_lookback": 10, "vol_target": 0.30},
        gates={
            GateId.G0A: GateResult(status=GateVerdict.PASS, date=date(2026, 1, 1)),
            GateId.G1: GateResult(
                status=GateVerdict.PASS,
                date=date(2026, 1, 1),
                details={"best_asset": "DOGE/USDT"},
            ),
            GateId.G2: GateResult(status=GateVerdict.PASS, date=date(2026, 1, 1)),
        },
        asset_performance=[
            AssetMetrics(symbol="DOGE/USDT", sharpe=1.5, cagr=50.0, mdd=20.0, trades=100),
        ],
    )


class TestUpdateYamlG2h:
    """_update_yaml_g2h YAML 갱신 테스트."""

    def test_yaml_parameters_updated(self, tmp_path: Path) -> None:
        """YAML parameters 섹션에 optimized params 기록."""
        store = StrategyStore(base_dir=tmp_path)
        record = _make_record()
        store.save(record)

        mock_result = MagicMock()
        mock_result.best_params = {"er_lookback": 15, "vol_target": 0.40}
        mock_result.best_sharpe = 2.0
        mock_result.default_sharpe = 1.5
        mock_result.improvement_pct = 33.3
        mock_result.n_trials = 100
        mock_result.n_completed = 95

        from src.cli._gate_runners_g2h import _update_yaml_g2h

        _update_yaml_g2h("kama", mock_result, 1.8, store)

        updated = store.load("kama")
        assert updated.parameters["er_lookback"] == 15
        assert updated.parameters["vol_target"] == 0.40

    def test_gate_result_recorded(self, tmp_path: Path) -> None:
        """G2H gate result YAML 기록."""
        store = StrategyStore(base_dir=tmp_path)
        record = _make_record()
        store.save(record)

        mock_result = MagicMock()
        mock_result.best_params = {"er_lookback": 12}
        mock_result.best_sharpe = 1.8
        mock_result.default_sharpe = 1.5
        mock_result.improvement_pct = 20.0
        mock_result.n_trials = 50
        mock_result.n_completed = 48

        from src.cli._gate_runners_g2h import _update_yaml_g2h

        _update_yaml_g2h("kama", mock_result, 1.2, store)

        # Force cache refresh
        store._cache.clear()
        updated = store.load("kama")
        assert GateId.G2H in updated.gates
        assert updated.gates[GateId.G2H].status == GateVerdict.PASS
        assert updated.gates[GateId.G2H].details["best_sharpe_is"] == 1.8

    def test_oos_sharpe_in_details(self, tmp_path: Path) -> None:
        """OOS 검증 결과가 details에 포함."""
        store = StrategyStore(base_dir=tmp_path)
        record = _make_record()
        store.save(record)

        mock_result = MagicMock()
        mock_result.best_params = {"er_lookback": 12}
        mock_result.best_sharpe = 2.0
        mock_result.default_sharpe = 1.5
        mock_result.improvement_pct = 33.3
        mock_result.n_trials = 100
        mock_result.n_completed = 95

        from src.cli._gate_runners_g2h import _update_yaml_g2h

        _update_yaml_g2h("kama", mock_result, 1.6, store)

        store._cache.clear()
        updated = store.load("kama")
        assert updated.gates[GateId.G2H].details["oos_sharpe"] == 1.6

    def test_always_pass(self, tmp_path: Path) -> None:
        """낮은 성과에도 Always PASS."""
        store = StrategyStore(base_dir=tmp_path)
        record = _make_record()
        store.save(record)

        mock_result = MagicMock()
        mock_result.best_params = {"er_lookback": 10}
        mock_result.best_sharpe = 0.5
        mock_result.default_sharpe = 0.6
        mock_result.improvement_pct = -16.7
        mock_result.n_trials = 10
        mock_result.n_completed = 10

        from src.cli._gate_runners_g2h import _update_yaml_g2h

        _update_yaml_g2h("kama", mock_result, 0.3, store)

        store._cache.clear()
        updated = store.load("kama")
        assert updated.gates[GateId.G2H].status == GateVerdict.PASS


class TestSaveJsonResults:
    """JSON 결과 저장 테스트."""

    def test_g3_sweeps_json_created(self, tmp_path: Path) -> None:
        """G3 sweep JSON 파일 생성."""
        from src.backtest.optimizer import OptimizationResult, ParamSpec
        from src.cli._gate_runners_g2h import _save_json_results

        result = OptimizationResult(
            best_params={"er_lookback": 12, "vol_target": 0.35},
            best_sharpe=2.0,
            default_sharpe=1.5,
            improvement_pct=33.3,
            n_trials=100,
            n_completed=95,
            search_space=[
                ParamSpec(name="er_lookback", param_type="int", low=5, high=100, default=10),
            ],
            top_trials=[{"number": 0, "sharpe": 2.0, "params": {"er_lookback": 12}}],
        )
        g3_sweeps = {"er_lookback": [8, 10, 12, 14, 16]}

        # Patch _RESULTS_DIR
        with patch("src.cli._gate_runners_g2h._RESULTS_DIR", tmp_path):
            _save_json_results("kama", result, 1.8, g3_sweeps, seed=42)

        json_path = tmp_path / "gate2h_kama.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["g3_sweeps"]["er_lookback"] == [8, 10, 12, 14, 16]
        assert data["optimization"]["best_sharpe_is"] == 2.0
        assert data["optimization"]["oos_sharpe"] == 1.8
        assert data["meta"]["strategy"] == "kama"
        assert data["meta"]["seed"] == 42
