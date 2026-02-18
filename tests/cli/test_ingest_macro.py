"""Tests for macro ingestion CLI commands."""

from typer.testing import CliRunner

from src.cli.ingest_macro import app

runner = CliRunner()


class TestMacroInfo:
    """info 커맨드 테스트."""

    def test_info_all(self) -> None:
        """info --type all 실행."""
        result = runner.invoke(app, ["info", "--type", "all"])
        assert result.exit_code == 0
        assert "fred" in result.stdout.lower() or "Macro" in result.stdout

    def test_info_fred(self) -> None:
        """info --type fred 실행."""
        result = runner.invoke(app, ["info", "--type", "fred"])
        assert result.exit_code == 0

    def test_info_yfinance(self) -> None:
        """info --type yfinance 실행."""
        result = runner.invoke(app, ["info", "--type", "yfinance"])
        assert result.exit_code == 0

    def test_info_invalid_type(self) -> None:
        """info --type invalid 실행 시 에러."""
        result = runner.invoke(app, ["info", "--type", "invalid"])
        assert result.exit_code == 1


class TestMacroBatch:
    """batch 커맨드 테스트."""

    def test_batch_dry_run(self) -> None:
        """batch --dry-run 실행."""
        result = runner.invoke(app, ["batch", "--type", "all", "--dry-run"])
        assert result.exit_code == 0
        assert "Dry-run" in result.stdout or "dry-run" in result.stdout.lower()

    def test_batch_fred_dry_run(self) -> None:
        """batch --type fred --dry-run 실행."""
        result = runner.invoke(app, ["batch", "--type", "fred", "--dry-run"])
        assert result.exit_code == 0
