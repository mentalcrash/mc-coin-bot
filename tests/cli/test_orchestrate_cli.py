"""Tests for Orchestrate CLI (display helpers, command interfaces)."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.orchestrate import (
    _display_pod_config_table,
    _display_pod_summary,
    _display_portfolio_metrics,
    app,
)
from src.models.backtest import PerformanceMetrics
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.result import OrchestratedResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_metrics() -> PerformanceMetrics:
    """최소 필수 필드만 포함한 PerformanceMetrics."""
    return PerformanceMetrics(
        total_return=25.5,
        cagr=12.3,
        sharpe_ratio=1.57,
        max_drawdown=19.4,
        win_rate=58.0,
        total_trades=120,
        winning_trades=70,
        losing_trades=50,
    )


@pytest.fixture()
def sample_result(sample_metrics: PerformanceMetrics) -> OrchestratedResult:
    """테스트용 OrchestratedResult."""
    return OrchestratedResult(
        portfolio_metrics=sample_metrics,
        portfolio_equity_curve=pd.Series([1.0, 1.05, 1.10, 1.08, 1.15]),
        pod_metrics={
            "pod-a": {"total_return": 0.15, "sharpe": 1.2, "mdd": -0.10, "n_days": 365},
            "pod-b": {"total_return": 0.08, "sharpe": 0.9, "mdd": -0.12, "n_days": 365},
        },
    )


@pytest.fixture()
def sample_orch_config() -> OrchestratorConfig:
    """테스트용 OrchestratorConfig."""
    return OrchestratorConfig(
        pods=(
            PodConfig(
                pod_id="pod-a",
                strategy_name="tsmom",
                symbols=("BTC/USDT", "ETH/USDT"),
                initial_fraction=0.40,
                max_fraction=0.60,
            ),
            PodConfig(
                pod_id="pod-b",
                strategy_name="donchian",
                symbols=("SOL/USDT",),
                initial_fraction=0.30,
                max_fraction=0.50,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# TestCommandHelp
# ---------------------------------------------------------------------------


class TestCommandHelp:
    """CLI --help 인터페이스 검증."""

    def test_backtest_help(self) -> None:
        """backtest --help에 CONFIG_PATH, --report, -V 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output or "config" in result.output.lower()
        assert "--report" in result.output
        assert "--verbose" in result.output

    def test_paper_help(self) -> None:
        """paper --help에 CONFIG_PATH, --db-path, -V 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["paper", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output or "config" in result.output.lower()
        assert "--db-path" in result.output
        assert "--verbose" in result.output

    def test_live_help(self) -> None:
        """live --help에 CONFIG_PATH, --db-path, -V 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["live", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output or "config" in result.output.lower()
        assert "--db-path" in result.output
        assert "--verbose" in result.output


# ---------------------------------------------------------------------------
# TestDisplayHelpers
# ---------------------------------------------------------------------------


class TestDisplayHelpers:
    """Display helper 함수 검증."""

    def test_display_portfolio_metrics(self, sample_result: OrchestratedResult) -> None:
        """_display_portfolio_metrics() — 에러 없이 실행."""
        _display_portfolio_metrics("Test Portfolio", sample_result)

    def test_display_pod_summary(self, sample_result: OrchestratedResult) -> None:
        """_display_pod_summary() — 에러 없이 실행."""
        _display_pod_summary(sample_result)

    def test_display_pod_summary_empty(self, sample_metrics: PerformanceMetrics) -> None:
        """pod_metrics 빈 경우 — 에러 없이 종료."""
        result = OrchestratedResult(
            portfolio_metrics=sample_metrics,
            portfolio_equity_curve=pd.Series(dtype=float),
        )
        _display_pod_summary(result)

    def test_display_pod_config_table(self, sample_orch_config: OrchestratorConfig) -> None:
        """_display_pod_config_table() — 에러 없이 실행."""
        _display_pod_config_table(sample_orch_config)


# ---------------------------------------------------------------------------
# TestBacktestCommand
# ---------------------------------------------------------------------------


class TestBacktestCommand:
    """backtest 커맨드 에러 케이스 검증."""

    def test_backtest_missing_file(self) -> None:
        """존재하지 않는 config → 에러."""
        runner = CliRunner()
        result = runner.invoke(app, ["backtest", "nonexistent.yaml"])
        assert result.exit_code != 0
