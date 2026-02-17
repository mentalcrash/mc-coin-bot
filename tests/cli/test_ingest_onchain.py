"""Tests for src/cli/ingest_onchain.py — CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.cli.ingest_onchain import app

runner = CliRunner()


class TestPipelineCommand:
    def test_pipeline_invokes_fetch_and_process(self, tmp_path) -> None:
        """pipeline 커맨드가 asyncio.run을 호출."""
        with (
            patch("src.cli.ingest_onchain.asyncio.run") as mock_run,
            patch("src.cli.ingest_onchain.get_settings") as mock_settings,
            patch("src.cli.ingest_onchain.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings
            mock_run.return_value = None

            result = runner.invoke(app, ["pipeline", "defillama", "stablecoin_total"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_pipeline_error_handling(self, tmp_path) -> None:
        """pipeline 실패 시 exit code 1."""
        with (
            patch(
                "src.cli.ingest_onchain.asyncio.run",
                side_effect=RuntimeError("test error"),
            ),
            patch("src.cli.ingest_onchain.get_settings") as mock_settings,
            patch("src.cli.ingest_onchain.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings

            result = runner.invoke(app, ["pipeline", "defillama", "stablecoin_total"])
            assert result.exit_code == 1


class TestBatchCommand:
    def test_batch_dry_run(self, tmp_path) -> None:
        """--dry-run이면 fetch 미호출 + 목록만 출력."""
        with (
            patch("src.cli.ingest_onchain.asyncio.run") as mock_run,
            patch("src.cli.ingest_onchain.get_settings") as mock_settings,
            patch("src.cli.ingest_onchain.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings

            result = runner.invoke(app, ["batch", "--type", "stablecoin", "--dry-run"])
            assert result.exit_code == 0
            assert "Dry-run" in result.output
            mock_run.assert_not_called()

    def test_batch_executes(self, tmp_path) -> None:
        """batch 커맨드가 asyncio.run 호출."""
        with (
            patch("src.cli.ingest_onchain.asyncio.run") as mock_run,
            patch("src.cli.ingest_onchain.get_settings") as mock_settings,
            patch("src.cli.ingest_onchain.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings
            mock_run.return_value = MagicMock(total=1, success=["ok"], failed=[], skipped=[])

            result = runner.invoke(app, ["batch", "--type", "dex"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_batch_invalid_type(self, tmp_path) -> None:
        """존재하지 않는 batch type → exit code 1."""
        with (
            patch("src.cli.ingest_onchain.get_settings") as mock_settings,
            patch("src.cli.ingest_onchain.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings

            result = runner.invoke(app, ["batch", "--type", "nonexistent"])
            assert result.exit_code == 1


class TestInfoCommand:
    def test_info_no_data(self, tmp_path) -> None:
        """데이터 없을 때 Missing 표시."""
        with patch("src.cli.ingest_onchain.get_settings") as mock_settings:
            settings = MagicMock()
            settings.bronze_dir = tmp_path / "bronze"
            settings.silver_dir = tmp_path / "silver"
            # exists 경로가 없으니 False 반환
            settings.get_onchain_bronze_path.return_value = (
                tmp_path / "bronze" / "onchain" / "defillama" / "missing.parquet"
            )
            settings.get_onchain_silver_path.return_value = (
                tmp_path / "silver" / "onchain" / "defillama" / "missing.parquet"
            )
            mock_settings.return_value = settings

            result = runner.invoke(app, ["info", "--type", "dex"])
            assert result.exit_code == 0
            assert "Missing" in result.output

    def test_info_default_all(self, tmp_path) -> None:
        """기본 'all' 타입은 모든 카테고리 표시."""
        with patch("src.cli.ingest_onchain.get_settings") as mock_settings:
            settings = MagicMock()
            settings.get_onchain_bronze_path.return_value = tmp_path / "nonexistent.parquet"
            settings.get_onchain_silver_path.return_value = tmp_path / "nonexistent.parquet"
            mock_settings.return_value = settings

            result = runner.invoke(app, ["info"])
            assert result.exit_code == 0
