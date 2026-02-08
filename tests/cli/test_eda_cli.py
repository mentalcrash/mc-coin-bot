"""Tests for EDA CLI (RunMode enum, _display_metrics helper, config-based commands)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src.cli.eda import RunMode, _display_metrics, app
from src.models.backtest import PerformanceMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_metrics() -> PerformanceMetrics:
    """최소 필수 필드만 포함한 PerformanceMetrics."""
    return PerformanceMetrics(
        total_return=10.5,
        cagr=8.2,
        sharpe_ratio=1.25,
        max_drawdown=15.3,
        win_rate=55.0,
        total_trades=20,
        winning_trades=11,
        losing_trades=9,
    )


@pytest.fixture()
def sample_metrics_full() -> PerformanceMetrics:
    """optional 필드 포함한 PerformanceMetrics."""
    return PerformanceMetrics(
        total_return=10.5,
        cagr=8.2,
        sharpe_ratio=1.25,
        max_drawdown=15.3,
        win_rate=55.0,
        total_trades=20,
        winning_trades=11,
        losing_trades=9,
        volatility=22.5,
        profit_factor=1.85,
    )


# ---------------------------------------------------------------------------
# TestRunMode
# ---------------------------------------------------------------------------


class TestRunMode:
    """RunMode enum 검증."""

    def test_values(self) -> None:
        assert RunMode.BACKTEST.value == "backtest"
        assert RunMode.SHADOW.value == "shadow"

    def test_member_count(self) -> None:
        assert len(RunMode) == 2

    def test_str_enum(self) -> None:
        """str 상속 확인 — Typer CLI 호환."""
        assert isinstance(RunMode.BACKTEST, str)


# ---------------------------------------------------------------------------
# TestDisplayMetrics
# ---------------------------------------------------------------------------


class TestDisplayMetrics:
    """_display_metrics() helper 검증."""

    def test_basic_output(self, sample_metrics: PerformanceMetrics) -> None:
        """기본 metrics 출력 — 에러 없이 실행."""
        _display_metrics("Test Title", sample_metrics)

    def test_with_extra_rows(self, sample_metrics: PerformanceMetrics) -> None:
        """extra_rows 포함 출력."""
        _display_metrics(
            "Test Extra",
            sample_metrics,
            extra_rows=[("Mode", "shadow"), ("Symbol", "BTC/USDT")],
        )

    def test_with_optional_fields(self, sample_metrics_full: PerformanceMetrics) -> None:
        """volatility, profit_factor 포함 출력."""
        _display_metrics("Full Metrics", sample_metrics_full)


# ---------------------------------------------------------------------------
# TestRunCommandInterface
# ---------------------------------------------------------------------------


class TestRunCommandInterface:
    """run 커맨드의 config_path 인터페이스 검증."""

    def test_help_shows_config_path(self) -> None:
        """--help에 CONFIG_PATH 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output or "config" in result.output.lower()

    def test_help_shows_mode_option(self) -> None:
        """--help에 --mode 옵션 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output

    def test_help_shows_verbose_option(self) -> None:
        """--help에 --verbose 옵션 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output


class TestRunMultiCommandInterface:
    """run-multi 커맨드의 config_path 인터페이스 검증."""

    def test_help_shows_config_path(self) -> None:
        """--help에 CONFIG_PATH 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["run-multi", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output or "config" in result.output.lower()

    def test_help_shows_verbose_option(self) -> None:
        """--help에 --verbose 옵션 표시."""
        runner = CliRunner()
        result = runner.invoke(app, ["run-multi", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output
