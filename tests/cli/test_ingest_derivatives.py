"""Tests for src/cli/ingest_derivatives.py — CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from src.cli.ingest_derivatives import app

runner = CliRunner()


class TestInfoCommand:
    def test_info_no_data(self, tmp_path) -> None:
        """데이터 없을 때 Missing 표시."""
        with patch("src.cli.ingest_derivatives.get_settings") as mock_settings:
            settings = MagicMock()
            settings.bronze_dir = tmp_path / "bronze"
            settings.silver_dir = tmp_path / "silver"
            settings.get_bronze_deriv_path.return_value = (
                tmp_path / "bronze" / "BTC_USDT" / "2024_deriv.parquet"
            )
            settings.get_silver_deriv_path.return_value = (
                tmp_path / "silver" / "BTC_USDT" / "2024_deriv.parquet"
            )
            mock_settings.return_value = settings

            result = runner.invoke(app, ["info", "BTC/USDT", "--year", "2024"])
            assert result.exit_code == 0
            assert "Missing" in result.output


class TestBronzeCommand:
    def test_bronze_invokes_fetch(self, tmp_path) -> None:
        """Bronze 커맨드가 _fetch_bronze_deriv를 호출."""
        with (
            patch("src.cli.ingest_derivatives._fetch_bronze_deriv", new_callable=AsyncMock),
            patch("src.cli.ingest_derivatives.asyncio.run") as mock_run,
            patch("src.cli.ingest_derivatives.get_settings") as mock_settings,
            patch("src.cli.ingest_derivatives.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings

            # asyncio.run이 coroutine을 실행하는 대신 모킹
            mock_run.return_value = None

            result = runner.invoke(app, ["bronze", "BTC/USDT", "--year", "2024"])
            assert result.exit_code == 0

    def test_bronze_error_handling(self, tmp_path) -> None:
        """Bronze 실패 시 exit code 1."""
        with (
            patch("src.cli.ingest_derivatives.asyncio.run", side_effect=RuntimeError("test error")),
            patch("src.cli.ingest_derivatives.get_settings") as mock_settings,
            patch("src.cli.ingest_derivatives.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings

            result = runner.invoke(app, ["bronze", "BTC/USDT", "--year", "2024"])
            assert result.exit_code == 1


class TestSilverCommand:
    def test_silver_missing_bronze_fails(self, tmp_path) -> None:
        """Bronze 데이터 없이 Silver 실행 시 실패."""
        with (
            patch("src.cli.ingest_derivatives.get_settings") as mock_settings,
            patch("src.cli.ingest_derivatives.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            settings.bronze_dir = tmp_path / "bronze"
            settings.silver_dir = tmp_path / "silver"
            settings.get_bronze_deriv_path.return_value = (
                tmp_path / "bronze" / "BTC_USDT" / "2024_deriv.parquet"
            )
            mock_settings.return_value = settings

            result = runner.invoke(app, ["silver", "BTC/USDT", "--year", "2024"])
            assert result.exit_code == 1


class TestPipelineCommand:
    def test_pipeline_invokes_both_steps(self, tmp_path) -> None:
        """Pipeline이 Bronze + Silver 양쪽 모두 실행."""
        with (
            patch("src.cli.ingest_derivatives.asyncio.run") as mock_run,
            patch("src.cli.ingest_derivatives.DerivativesSilverProcessor") as mock_processor_cls,
            patch("src.cli.ingest_derivatives.get_settings") as mock_settings,
            patch("src.cli.ingest_derivatives.setup_logger"),
        ):
            settings = MagicMock()
            settings.log_dir = tmp_path / "logs"
            mock_settings.return_value = settings
            mock_run.return_value = None

            mock_processor = MagicMock()
            mock_processor.process.return_value = tmp_path / "silver" / "out.parquet"
            mock_processor_cls.return_value = mock_processor

            result = runner.invoke(app, ["pipeline", "BTC/USDT", "--year", "2024"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            mock_processor.process.assert_called_once()
