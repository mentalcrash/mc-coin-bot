"""EDA 설정 모델 테스트."""

import pytest
from pydantic import ValidationError

from src.models.eda import EDAConfig, ExecutionMode


class TestExecutionMode:
    """ExecutionMode StrEnum 테스트."""

    def test_all_modes_defined(self) -> None:
        assert len(ExecutionMode) == 5

    def test_mode_values(self) -> None:
        assert ExecutionMode.BACKTEST == "backtest"
        assert ExecutionMode.SHADOW == "shadow"
        assert ExecutionMode.PAPER == "paper"
        assert ExecutionMode.CANARY == "canary"
        assert ExecutionMode.LIVE == "live"


class TestEDAConfig:
    """EDAConfig 테스트."""

    def test_default_config(self) -> None:
        config = EDAConfig()
        assert config.execution_mode == ExecutionMode.BACKTEST
        assert config.event_queue_size == 10000
        assert config.event_log_path is None
        assert config.enable_heartbeat is True
        assert config.heartbeat_interval_bars == 100
        assert config.backtest_fill_delay_bars == 1

    def test_custom_config(self) -> None:
        config = EDAConfig(
            execution_mode=ExecutionMode.SHADOW,
            event_queue_size=5000,
            event_log_path="logs/events.jsonl",
            enable_heartbeat=False,
            backtest_fill_delay_bars=0,
        )
        assert config.execution_mode == ExecutionMode.SHADOW
        assert config.event_log_path == "logs/events.jsonl"

    def test_frozen_immutability(self) -> None:
        config = EDAConfig()
        with pytest.raises(ValidationError):
            config.execution_mode = ExecutionMode.LIVE  # type: ignore[misc]

    def test_queue_size_minimum(self) -> None:
        with pytest.raises(ValidationError):
            EDAConfig(event_queue_size=50)  # < 100

    def test_heartbeat_interval_minimum(self) -> None:
        with pytest.raises(ValidationError):
            EDAConfig(heartbeat_interval_bars=0)  # < 1

    def test_fill_delay_allows_zero(self) -> None:
        config = EDAConfig(backtest_fill_delay_bars=0)
        assert config.backtest_fill_delay_bars == 0

    def test_json_serialization(self) -> None:
        config = EDAConfig(execution_mode=ExecutionMode.PAPER)
        json_str = config.model_dump_json()
        assert "paper" in json_str
