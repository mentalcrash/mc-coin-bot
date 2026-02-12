"""Tests for src/cli/_gate_runners.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.cli._gate_runners import (
    GATE3_STRATEGIES,
    GATE3_WEIGHT_PAIRS,
    analyze_sweep,
    resolve_timeframe,
)
from src.pipeline.models import GateId, GateVerdict, StrategyMeta, StrategyRecord, StrategyStatus
from src.pipeline.store import StrategyStore

_STORE_PATH = "src.pipeline.store.StrategyStore"


# =============================================================================
# resolve_timeframe tests
# =============================================================================


class TestResolveTimeframe:
    def test_daily_strategy(self, tmp_path: Path) -> None:
        """1D TF 전략 -> '1D'."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test-daily",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="FULL",
                status=StrategyStatus.CANDIDATE,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        with patch(_STORE_PATH, return_value=store):
            assert resolve_timeframe("test-daily") == "1D"

    def test_4h_strategy_with_annotation(self, tmp_path: Path) -> None:
        """'4H (annualization_factor=2190)' -> '4h'."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test-4h",
                display_name="Test",
                category="Test",
                timeframe="4H (annualization_factor=2190)",
                short_mode="HEDGE_ONLY",
                status=StrategyStatus.CANDIDATE,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        with patch(_STORE_PATH, return_value=store):
            assert resolve_timeframe("test-4h") == "4h"

    def test_12h_strategy(self, tmp_path: Path) -> None:
        """'12H' -> '12h'."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test-12h",
                display_name="Test",
                category="Test",
                timeframe="12H",
                short_mode="HEDGE_ONLY",
                status=StrategyStatus.CANDIDATE,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        with patch(_STORE_PATH, return_value=store):
            assert resolve_timeframe("test-12h") == "12h"

    def test_1h_strategy(self, tmp_path: Path) -> None:
        """'1H' -> '1h'."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test-1h",
                display_name="Test",
                category="Test",
                timeframe="1H",
                short_mode="HEDGE_ONLY",
                status=StrategyStatus.CANDIDATE,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        with patch(_STORE_PATH, return_value=store):
            assert resolve_timeframe("test-1h") == "1h"

    def test_nonexistent_strategy_defaults_to_1d(self) -> None:
        """존재하지 않는 전략 -> '1D' fallback."""
        with patch(_STORE_PATH) as mock_cls:
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_cls.return_value = mock_store
            assert resolve_timeframe("nonexistent") == "1D"


# =============================================================================
# analyze_sweep tests
# =============================================================================


def _make_sweep_results(values: list[float], sharpes: list[float]) -> list[dict[str, Any]]:
    """Sweep 결과 mock 생성."""
    return [
        {
            "value": v,
            "sharpe_ratio": s,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "cagr": 0.0,
            "total_trades": 100,
        }
        for v, s in zip(values, sharpes, strict=True)
    ]


class TestAnalyzeSweep:
    def test_pass_with_plateau_and_stability(self) -> None:
        """고원 3개 + +-20% 안정 -> PASS."""
        values = [8, 9, 10, 11, 12]
        sharpes = [1.0, 1.1, 1.2, 1.15, 0.95]
        results = _make_sweep_results(values, sharpes)

        analysis = analyze_sweep("lookback", 10, results)

        assert analysis["verdict"] == "PASS"
        assert analysis["plateau_exists"] is True
        assert analysis["pm20_stable"] is True

    def test_fail_no_plateau(self) -> None:
        """고원 부재 -> FAIL."""
        values = [5, 10, 15, 20, 25]
        sharpes = [0.1, 0.5, 2.5, 0.2, 0.1]
        results = _make_sweep_results(values, sharpes)

        analysis = analyze_sweep("lookback", 10, results)

        # 2.5 * 0.8 = 2.0 threshold, only value 15 (2.5) >= 2.0 -> count=1 < 3
        assert analysis["plateau_exists"] is False
        assert analysis["verdict"] == "FAIL"

    def test_fail_negative_sharpe_in_pm20(self) -> None:
        """+-20% 내에 음수 Sharpe -> FAIL."""
        values = [8, 9, 10, 11, 12]
        sharpes = [0.8, -0.1, 0.9, 0.85, 0.7]
        results = _make_sweep_results(values, sharpes)

        analysis = analyze_sweep("lookback", 10, results)

        assert analysis["pm20_stable"] is False
        assert analysis["verdict"] == "FAIL"

    def test_empty_results(self) -> None:
        """유효 결과 없음 -> FAIL."""
        results = [
            {
                "value": 10,
                "sharpe_ratio": float("nan"),
                "total_return": 0,
                "max_drawdown": 0,
                "cagr": 0,
                "total_trades": 0,
            }
        ]
        analysis = analyze_sweep("lookback", 10, results)

        assert analysis["verdict"] == "FAIL"
        assert analysis["reason"] == "No valid results"

    def test_baseline_sharpe_recorded(self) -> None:
        """baseline_value에 해당하는 Sharpe가 기록됨."""
        values = [8, 10, 12]
        sharpes = [0.9, 1.2, 1.0]
        results = _make_sweep_results(values, sharpes)

        analysis = analyze_sweep("lookback", 10, results)

        assert analysis["baseline_sharpe"] == 1.2

    def test_plateau_count(self) -> None:
        """고원 카운트 정확도."""
        values = [5, 6, 7, 8, 9, 10]
        sharpes = [0.5, 0.9, 1.0, 1.1, 1.05, 0.95]
        results = _make_sweep_results(values, sharpes)

        analysis = analyze_sweep("lookback", 8, results)

        # best=1.1, threshold=0.88; 0.9, 1.0, 1.1, 1.05, 0.95 >= 0.88 -> count=5
        assert analysis["plateau_count"] == 5
        assert analysis["plateau_exists"] is True

    def test_pm20_range_non_numeric_baseline(self) -> None:
        """비숫자 baseline -> pm20_stable=True (스킵)."""
        results = _make_sweep_results([1, 2, 3], [0.5, 0.6, 0.7])
        analysis = analyze_sweep("mode", "fast", results)

        assert analysis["pm20_stable"] is True

    def test_all_sharpes_recorded(self) -> None:
        """all_sharpes에 (value, sharpe) 쌍 기록."""
        values = [5, 10, 15]
        sharpes = [0.5, 1.0, 1.5]
        results = _make_sweep_results(values, sharpes)

        analysis = analyze_sweep("lookback", 10, results)

        assert analysis["all_sharpes"] == [(5, 0.5), (10, 1.0), (15, 1.5)]


# =============================================================================
# _update_yaml_g1 tests
# =============================================================================


class TestUpdateYamlG1:
    def test_pass_verdict(self, tmp_path: Path) -> None:
        """G1 PASS -> YAML 업데이트 + asset_performance 기록."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test-strat",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="FULL",
                status=StrategyStatus.IMPLEMENTED,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        results = [
            {
                "symbol": "SOL/USDT",
                "sharpe_ratio": 2.0,
                "cagr": 0.5,
                "max_drawdown": 20.0,
                "total_trades": 100,
                "total_return": 200.0,
                "profit_factor": 1.5,
                "win_rate": 55.0,
                "sortino_ratio": 2.5,
                "calmar_ratio": 3.0,
                "alpha": 10.0,
                "beta": 0.5,
            },
        ]

        with patch(_STORE_PATH, return_value=store):
            from src.cli._gate_runners import _update_yaml_g1

            _update_yaml_g1("test-strat", results)

        updated = store.load("test-strat")
        assert updated.gates[GateId.G1].status == GateVerdict.PASS
        assert len(updated.asset_performance) == 1
        assert updated.asset_performance[0].symbol == "SOL/USDT"

    def test_fail_verdict_retires(self, tmp_path: Path) -> None:
        """G1 FAIL -> RETIRED 상태 전환."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="fail-strat",
                display_name="Fail",
                category="Test",
                timeframe="1D",
                short_mode="FULL",
                status=StrategyStatus.IMPLEMENTED,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        results = [
            {
                "symbol": "BTC/USDT",
                "sharpe_ratio": 0.5,
                "cagr": 0.05,
                "max_drawdown": 50.0,
                "total_trades": 20,
                "total_return": 10.0,
                "profit_factor": 1.1,
                "win_rate": 45.0,
                "sortino_ratio": 0.6,
                "calmar_ratio": 0.3,
            },
        ]

        with patch(_STORE_PATH, return_value=store):
            from src.cli._gate_runners import _update_yaml_g1

            _update_yaml_g1("fail-strat", results)

        updated = store.load("fail-strat")
        assert updated.gates[GateId.G1].status == GateVerdict.FAIL
        assert updated.meta.status == StrategyStatus.RETIRED

    def test_nonexistent_strategy_noop(self) -> None:
        """존재하지 않는 전략 -> no-op."""
        with patch(_STORE_PATH) as mock_cls:
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_cls.return_value = mock_store

            from src.cli._gate_runners import _update_yaml_g1

            _update_yaml_g1("nonexistent", [])

            mock_store.record_gate.assert_not_called()


# =============================================================================
# _update_yaml_g3 tests
# =============================================================================


class TestUpdateYamlG3:
    def test_pass_verdict(self, tmp_path: Path) -> None:
        """G3 PASS -> YAML 업데이트."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="g3-strat",
                display_name="G3",
                category="Test",
                timeframe="1D",
                short_mode="FULL",
                status=StrategyStatus.TESTING,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        result = {
            "strategy": "g3-strat",
            "verdict": "PASS",
            "param_verdicts": {"vol_target": "PASS", "lookback": "PASS"},
            "fail_params": [],
        }

        with patch(_STORE_PATH, return_value=store):
            from src.cli._gate_runners import _update_yaml_g3

            _update_yaml_g3(result)

        updated = store.load("g3-strat")
        assert updated.gates[GateId.G3].status == GateVerdict.PASS

    def test_fail_verdict_retires(self, tmp_path: Path) -> None:
        """G3 FAIL -> RETIRED."""
        store = StrategyStore(base_dir=tmp_path)
        record = StrategyRecord(
            meta=StrategyMeta(
                name="g3-fail",
                display_name="G3F",
                category="Test",
                timeframe="1D",
                short_mode="FULL",
                status=StrategyStatus.TESTING,
                created_at=date(2026, 1, 1),
            ),
        )
        store.save(record)

        result = {
            "strategy": "g3-fail",
            "verdict": "FAIL",
            "param_verdicts": {"vol_target": "PASS", "lookback": "FAIL"},
            "fail_params": ["lookback"],
        }

        with patch(_STORE_PATH, return_value=store):
            from src.cli._gate_runners import _update_yaml_g3

            _update_yaml_g3(result)

        updated = store.load("g3-fail")
        assert updated.gates[GateId.G3].status == GateVerdict.FAIL
        assert updated.meta.status == StrategyStatus.RETIRED


# =============================================================================
# CLI command registration tests
# =============================================================================


class TestCLICommands:
    def test_gate1_run_help(self) -> None:
        """gate1-run --help 동작 확인."""
        from src.cli.pipeline import app

        runner = CliRunner()
        result = runner.invoke(app, ["gate1-run", "--help"])
        assert result.exit_code == 0
        assert "Gate 1" in result.output
        assert "--symbols" in result.output
        assert "--start" in result.output
        assert "--end" in result.output
        assert "--capital" in result.output

    def test_gate3_run_help(self) -> None:
        """gate3-run --help 동작 확인."""
        from src.cli.pipeline import app

        runner = CliRunner()
        result = runner.invoke(app, ["gate3-run", "--help"])
        assert result.exit_code == 0
        assert "Gate 3" in result.output
        assert "--json" in result.output


# =============================================================================
# Constants validation
# =============================================================================


class TestConstants:
    def test_gate3_strategies_have_required_keys(self) -> None:
        """모든 GATE3_STRATEGIES에 best_asset, baseline, sweeps 있음."""
        for name, config in GATE3_STRATEGIES.items():
            assert "best_asset" in config, f"{name} missing best_asset"
            assert "baseline" in config, f"{name} missing baseline"
            assert "sweeps" in config, f"{name} missing sweeps"

    def test_gate3_weight_pairs_strategies_exist(self) -> None:
        """GATE3_WEIGHT_PAIRS의 전략이 GATE3_STRATEGIES에 존재."""
        for name in GATE3_WEIGHT_PAIRS:
            assert name in GATE3_STRATEGIES, f"{name} not in GATE3_STRATEGIES"

    def test_gate3_sweep_params_in_baseline(self) -> None:
        """스윕 파라미터가 baseline에 존재."""
        for name, config in GATE3_STRATEGIES.items():
            baseline = config["baseline"]
            for param in config["sweeps"]:
                assert param in baseline, f"{name}: sweep param '{param}' not in baseline"
