"""Tests for CLI ingest deriv-ext commands."""

from unittest.mock import patch

from typer.testing import CliRunner

from src.cli.ingest_deriv_ext import app

runner = CliRunner()


class TestDerivExtInfo:
    """info 커맨드 테스트."""

    def test_info_default(self) -> None:
        """기본 info 출력."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Deriv-Ext Data Inventory" in result.output

    def test_info_coinalyze(self) -> None:
        """coinalyze 타입 필터."""
        result = runner.invoke(app, ["info", "--type", "coinalyze"])
        assert result.exit_code == 0
        assert "coinalyze" in result.output


class TestDerivExtBatch:
    """batch 커맨드 테스트."""

    def test_batch_dry_run(self) -> None:
        """dry-run 모드."""
        result = runner.invoke(app, ["batch", "--dry-run"])
        assert result.exit_code == 0
        assert "Dry-run mode" in result.output
        assert "btc_agg_oi" in result.output


class TestDerivExtPipeline:
    """pipeline 커맨드 테스트."""

    def test_pipeline_no_api_key(self) -> None:
        """API key 미설정 시 에러."""
        with patch("src.cli.ingest_deriv_ext.get_settings") as mock_settings:
            from unittest.mock import MagicMock

            from pydantic import SecretStr

            settings = MagicMock()
            settings.coinalyze_api_key = SecretStr("")
            settings.log_dir = "logs"
            mock_settings.return_value = settings

            result = runner.invoke(app, ["pipeline", "coinalyze", "btc_agg_oi"])
            assert result.exit_code == 1
