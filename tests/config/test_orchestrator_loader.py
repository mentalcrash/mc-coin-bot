"""Tests for Orchestrator YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config.orchestrator_loader import (
    OrchestratorBacktestScope,
    OrchestratorRunConfig,
    load_orchestrator_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_orchestrator_dict() -> dict:
    """최소 유효 orchestrator config dict."""
    return {
        "orchestrator": {
            "pods": [
                {
                    "pod_id": "pod-test",
                    "strategy_name": "tsmom",
                    "symbols": ["BTC/USDT"],
                    "initial_fraction": 0.30,
                    "max_fraction": 0.50,
                }
            ],
        },
    }


def _full_orchestrator_dict() -> dict:
    """전체 필드 orchestrator config dict."""
    return {
        "backtest": {
            "timeframe": "4h",
            "start": "2023-01-01",
            "end": "2024-12-31",
            "capital": 50000.0,
        },
        "orchestrator": {
            "allocation_method": "risk_parity",
            "rebalance_trigger": "hybrid",
            "rebalance_calendar_days": 7,
            "rebalance_drift_threshold": 0.10,
            "max_gross_leverage": 3.0,
            "max_portfolio_drawdown": 0.15,
            "daily_loss_limit": 0.03,
            "cost_bps": 4.0,
            "pods": [
                {
                    "pod_id": "pod-a",
                    "strategy_name": "tsmom",
                    "strategy_params": {"lookback": 30},
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "initial_fraction": 0.40,
                    "max_fraction": 0.60,
                    "use_trailing_stop": True,
                    "trailing_stop_atr_multiplier": 3.0,
                },
                {
                    "pod_id": "pod-b",
                    "strategy_name": "donchian",
                    "symbols": ["SOL/USDT"],
                    "initial_fraction": 0.30,
                    "max_fraction": 0.50,
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# TestOrchestratorBacktestScope
# ---------------------------------------------------------------------------


class TestOrchestratorBacktestScope:
    """OrchestratorBacktestScope 모델 검증."""

    def test_defaults(self) -> None:
        """기본값 검증: timeframe=1D, capital=100_000."""
        scope = OrchestratorBacktestScope()
        assert scope.timeframe == "1D"
        assert scope.start == "2020-01-01"
        assert scope.end == "2025-12-31"
        assert scope.capital == 100_000.0

    def test_frozen(self) -> None:
        """frozen=True — immutable."""
        scope = OrchestratorBacktestScope()
        with pytest.raises(ValidationError):
            scope.capital = 50_000.0  # type: ignore[misc]

    def test_invalid_capital(self) -> None:
        """capital <= 0 → ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorBacktestScope(capital=-1000.0)


# ---------------------------------------------------------------------------
# TestOrchestratorRunConfig
# ---------------------------------------------------------------------------


class TestOrchestratorRunConfig:
    """OrchestratorRunConfig 모델 검증."""

    def test_minimal(self) -> None:
        """최소 설정 (pods만) 로드."""
        raw = _minimal_orchestrator_dict()
        cfg = OrchestratorRunConfig.model_validate(raw)

        assert cfg.orchestrator.n_pods == 1
        assert cfg.orchestrator.pods[0].pod_id == "pod-test"
        # backtest defaults
        assert cfg.backtest.timeframe == "1D"
        assert cfg.backtest.capital == 100_000.0

    def test_full(self) -> None:
        """전체 필드 (allocation, lifecycle, rebalance 등) 로드."""
        raw = _full_orchestrator_dict()
        cfg = OrchestratorRunConfig.model_validate(raw)

        assert cfg.backtest.timeframe == "4h"
        assert cfg.backtest.capital == 50000.0
        assert cfg.orchestrator.n_pods == 2
        assert cfg.orchestrator.allocation_method.value == "risk_parity"
        assert cfg.orchestrator.rebalance_trigger.value == "hybrid"
        assert cfg.orchestrator.max_gross_leverage == 3.0
        assert cfg.orchestrator.daily_loss_limit == 0.03
        assert cfg.orchestrator.cost_bps == 4.0

        # Pod A
        pod_a = cfg.orchestrator.pods[0]
        assert pod_a.pod_id == "pod-a"
        assert pod_a.strategy_params == {"lookback": 30}
        assert pod_a.use_trailing_stop is True
        assert pod_a.trailing_stop_atr_multiplier == 3.0

    def test_all_symbols_dedup(self) -> None:
        """all_symbols 중복 제거 검증."""
        raw = _full_orchestrator_dict()
        cfg = OrchestratorRunConfig.model_validate(raw)

        symbols = cfg.orchestrator.all_symbols
        # BTC/USDT, ETH/USDT (pod-a) + SOL/USDT (pod-b) = 3 unique
        assert len(symbols) == 3
        assert "BTC/USDT" in symbols
        assert "SOL/USDT" in symbols

    def test_frozen(self) -> None:
        """frozen=True — immutable."""
        raw = _minimal_orchestrator_dict()
        cfg = OrchestratorRunConfig.model_validate(raw)
        with pytest.raises(ValidationError):
            cfg.backtest = OrchestratorBacktestScope()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestLoadOrchestratorConfig
# ---------------------------------------------------------------------------


class TestLoadOrchestratorConfig:
    """load_orchestrator_config 함수 검증."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """유효한 YAML 파일 로드 성공."""
        raw = _minimal_orchestrator_dict()
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(raw), encoding="utf-8")

        cfg = load_orchestrator_config(path)
        assert isinstance(cfg, OrchestratorRunConfig)
        assert cfg.orchestrator.n_pods == 1

    def test_load_nonexistent_file(self) -> None:
        """존재하지 않는 파일 → FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_orchestrator_config("nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """잘못된 값 → ValidationError (중복 pod_id)."""
        raw = {
            "orchestrator": {
                "pods": [
                    {
                        "pod_id": "dup",
                        "strategy_name": "a",
                        "symbols": ["BTC/USDT"],
                        "initial_fraction": 0.30,
                    },
                    {
                        "pod_id": "dup",
                        "strategy_name": "b",
                        "symbols": ["ETH/USDT"],
                        "initial_fraction": 0.30,
                    },
                ],
            },
        }
        path = tmp_path / "invalid.yaml"
        path.write_text(yaml.dump(raw), encoding="utf-8")

        with pytest.raises(ValidationError, match="Duplicate pod_id"):
            load_orchestrator_config(path)

    def test_load_fraction_exceeds(self, tmp_path: Path) -> None:
        """initial_fraction 합계 > 1.0 → ValidationError."""
        raw = {
            "orchestrator": {
                "pods": [
                    {
                        "pod_id": "a",
                        "strategy_name": "x",
                        "symbols": ["BTC/USDT"],
                        "initial_fraction": 0.60,
                    },
                    {
                        "pod_id": "b",
                        "strategy_name": "y",
                        "symbols": ["ETH/USDT"],
                        "initial_fraction": 0.50,
                    },
                ],
            },
        }
        path = tmp_path / "exceed.yaml"
        path.write_text(yaml.dump(raw), encoding="utf-8")

        with pytest.raises(ValidationError, match="initial_fraction"):
            load_orchestrator_config(path)

    def test_example_config_loads(self) -> None:
        """config/orchestrator-example.yaml 실제 로드 검증."""
        cfg = load_orchestrator_config("config/orchestrator-example.yaml")

        assert cfg.orchestrator.n_pods == 3
        assert cfg.backtest.capital == 100_000.0
        assert cfg.backtest.timeframe == "1D"
        assert cfg.orchestrator.allocation_method.value == "risk_parity"

        # All symbols: BTC, ETH, SOL, BNB, DOGE, LINK, AVAX = 7 unique
        all_syms = cfg.orchestrator.all_symbols
        assert len(all_syms) == 7
        assert "BTC/USDT" in all_syms
        assert "AVAX/USDT" in all_syms
