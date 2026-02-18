"""Tests for p1-check CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.pipeline import app
from src.pipeline.store import StrategyStore

runner = CliRunner()


@pytest.fixture
def _mock_data_pipeline(tmp_path: Path) -> None:
    """Mock data loading, indicator computation, and regime detection for p1-check."""
    # Create mock OHLCV data
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(np.random.default_rng(42).normal(100, 5, n).cumsum() + 1000, index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.default_rng(42).normal(1e6, 1e5, n),
        },
        index=idx,
    )

    # Mock MarketDataService
    mock_data = MagicMock()
    mock_data.ohlcv = ohlcv

    mock_service = MagicMock()
    mock_service.get.return_value = mock_data

    # Mock indicator
    indicator_series = pd.Series(np.random.default_rng(42).normal(0, 1, n), index=idx)

    # Mock regime detection
    regime_labels = pd.Series(
        np.random.default_rng(42).choice(["trending", "ranging", "volatile"], n),
        index=idx,
    )
    regime_df = pd.DataFrame({"regime_label": regime_labels}, index=idx)

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    original_store_init = StrategyStore.__init__

    def patched_store_init(self: StrategyStore, base_dir: Path = strategies_dir) -> None:
        original_store_init(self, base_dir=base_dir)

    with (
        patch("src.config.settings.get_settings"),
        patch("src.data.service.MarketDataService", return_value=mock_service),
        patch("src.market.feature_store.compute_indicator", return_value=indicator_series),
        patch("src.core.logger.setup_logger"),
        patch(
            "src.regime.detector.RegimeDetector.classify_series",
            return_value=regime_df,
        ),
        patch.object(StrategyStore, "__init__", patched_store_init),
    ):
        yield


@pytest.mark.usefixtures("_mock_data_pipeline")
class TestP1Check:
    def test_basic_ic_check(self) -> None:
        """기본 IC 점수 계산."""
        result = runner.invoke(app, ["p1-check", "rsi", "BTC/USDT", "--tf", "1D"])
        assert result.exit_code == 0
        assert "IC 사전 검증" in result.output
        assert "레짐 독립성" in result.output
        assert "--p1-items JSON" in result.output

    def test_with_category(self) -> None:
        """--category 옵션으로 카테고리 성공률 포함."""
        result = runner.invoke(
            app,
            ["p1-check", "rsi", "BTC/USDT", "--tf", "1D", "--category", "momentum"],
        )
        assert result.exit_code == 0
        assert "카테고리 성공률" in result.output

    def test_with_params(self) -> None:
        """파라미터 전달."""
        result = runner.invoke(
            app,
            ["p1-check", "rsi", "BTC/USDT", "--tf", "1D", "-p", "period=14"],
        )
        assert result.exit_code == 0
        assert "IC 사전 검증" in result.output

    def test_outputs_json(self) -> None:
        """--p1-items 복사 가능 JSON 출력."""
        result = runner.invoke(app, ["p1-check", "rsi", "BTC/USDT", "--tf", "1D"])
        assert result.exit_code == 0
        assert "--p1-items JSON" in result.output
