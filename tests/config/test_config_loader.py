"""Tests for YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config.config_loader import (
    BacktestScope,
    RunConfig,
    StrategySection,
    build_strategy,
    load_config,
)
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def default_config_path() -> Path:
    """config/default.yaml 경로."""
    return Path("config/default.yaml")


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    """최소 설정 YAML 파일."""
    data = {
        "backtest": {
            "symbols": ["BTC/USDT"],
        },
    }
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


@pytest.fixture()
def full_yaml(tmp_path: Path) -> Path:
    """전체 설정 YAML 파일 (cost_model 포함)."""
    data = {
        "backtest": {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1D",
            "start": "2024-01-01",
            "end": "2025-12-31",
            "capital": 50000.0,
        },
        "strategy": {
            "name": "tsmom",
            "params": {
                "lookback": 30,
                "vol_window": 30,
                "vol_target": 0.35,
            },
        },
        "portfolio": {
            "max_leverage_cap": 2.0,
            "rebalance_threshold": 0.10,
            "system_stop_loss": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
            "cost_model": {
                "maker_fee": 0.0002,
                "taker_fee": 0.0004,
                "slippage": 0.0005,
                "funding_rate_8h": 0.0001,
                "market_impact": 0.0002,
            },
        },
    }
    path = tmp_path / "full.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


@pytest.fixture()
def invalid_yaml(tmp_path: Path) -> Path:
    """잘못된 값을 가진 YAML."""
    data = {
        "backtest": {
            "symbols": ["BTC/USDT"],
            "capital": -1000.0,  # 음수 자본
        },
    }
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """load_config 함수 검증."""

    def test_load_default_config(self, default_config_path: Path) -> None:
        """config/default.yaml 로드 검증."""
        cfg = load_config(default_config_path)

        assert isinstance(cfg, RunConfig)
        assert isinstance(cfg.backtest, BacktestScope)
        assert isinstance(cfg.strategy, StrategySection)
        assert isinstance(cfg.portfolio, PortfolioManagerConfig)

        assert len(cfg.backtest.symbols) == 8
        assert cfg.backtest.symbols[0] == "BTC/USDT"
        assert cfg.backtest.capital == 100000.0
        assert cfg.strategy.name == "tsmom"
        assert cfg.strategy.params["lookback"] == 30
        assert cfg.portfolio.max_leverage_cap == 2.0

    def test_load_minimal_yaml(self, minimal_yaml: Path) -> None:
        """최소 설정 (symbols만) 로드."""
        cfg = load_config(minimal_yaml)

        assert cfg.backtest.symbols == ["BTC/USDT"]
        # defaults
        assert cfg.backtest.timeframe == "1D"
        assert cfg.backtest.capital == 100000.0
        assert cfg.strategy.name == "tsmom"
        assert cfg.strategy.params == {}
        assert isinstance(cfg.portfolio, PortfolioManagerConfig)

    def test_load_nested_cost_model(self, full_yaml: Path) -> None:
        """중첩 CostModel 파싱 검증."""
        cfg = load_config(full_yaml)

        assert isinstance(cfg.portfolio.cost_model, CostModel)
        assert cfg.portfolio.cost_model.maker_fee == 0.0002
        assert cfg.portfolio.cost_model.taker_fee == 0.0004
        assert cfg.portfolio.cost_model.slippage == 0.0005

    def test_load_nonexistent_file(self) -> None:
        """존재하지 않는 파일은 FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent.yaml")

    def test_load_invalid_values(self, invalid_yaml: Path) -> None:
        """잘못된 값은 ValidationError."""
        with pytest.raises(ValidationError):
            load_config(invalid_yaml)

    def test_full_config_portfolio_settings(self, full_yaml: Path) -> None:
        """전체 config의 portfolio 설정 검증."""
        cfg = load_config(full_yaml)

        assert cfg.portfolio.max_leverage_cap == 2.0
        assert cfg.portfolio.rebalance_threshold == 0.10
        assert cfg.portfolio.system_stop_loss == 0.10
        assert cfg.portfolio.use_trailing_stop is True
        assert cfg.portfolio.trailing_stop_atr_multiplier == 3.0

    def test_backtest_scope_dates(self, full_yaml: Path) -> None:
        """날짜 문자열 검증."""
        cfg = load_config(full_yaml)

        assert cfg.backtest.start == "2024-01-01"
        assert cfg.backtest.end == "2025-12-31"
        assert cfg.backtest.capital == 50000.0


# ---------------------------------------------------------------------------
# TestBuildStrategy
# ---------------------------------------------------------------------------


class TestBuildStrategy:
    """build_strategy 함수 검증."""

    def test_build_strategy_tsmom(self, default_config_path: Path) -> None:
        """tsmom 전략 생성."""
        cfg = load_config(default_config_path)
        strategy = build_strategy(cfg)

        assert strategy.name == "VW-TSMOM"
        assert strategy.config is not None
        assert strategy.config.lookback == 30
        assert strategy.config.vol_target == 0.35

    def test_build_strategy_no_params(self, minimal_yaml: Path) -> None:
        """params 없이 기본값으로 전략 생성."""
        cfg = load_config(minimal_yaml)
        strategy = build_strategy(cfg)

        assert strategy.name == "VW-TSMOM"
        # TSMOMConfig 기본값
        assert strategy.config is not None
        assert strategy.config.lookback == 30
        assert strategy.config.vol_target == 0.30

    def test_build_strategy_unknown(self, tmp_path: Path) -> None:
        """알 수 없는 전략은 KeyError."""
        data = {
            "backtest": {"symbols": ["BTC/USDT"]},
            "strategy": {"name": "unknown-strategy"},
        }
        path = tmp_path / "unknown.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")

        cfg = load_config(path)
        with pytest.raises(KeyError, match="unknown-strategy"):
            build_strategy(cfg)


# ---------------------------------------------------------------------------
# TestRunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    """RunConfig 모델 검증."""

    def test_frozen(self, default_config_path: Path) -> None:
        """frozen=True 검증."""
        cfg = load_config(default_config_path)
        with pytest.raises(ValidationError):
            cfg.backtest = BacktestScope(symbols=["ETH/USDT"])  # type: ignore[misc]

    def test_symbols_min_length(self, tmp_path: Path) -> None:
        """symbols는 최소 1개 필요."""
        data = {"backtest": {"symbols": []}}
        path = tmp_path / "empty_symbols.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_config(path)
