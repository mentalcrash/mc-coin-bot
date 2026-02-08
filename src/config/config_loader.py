"""YAML 설정 파일 로더.

YAML 파일에서 RunConfig를 로드하고,
전략 인스턴스를 생성하는 유틸리티를 제공합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing, Path
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.portfolio.config import PortfolioManagerConfig
from src.strategy import get_strategy

if TYPE_CHECKING:
    from src.strategy.base import BaseStrategy


class BacktestScope(BaseModel):
    """백테스트 범위 설정."""

    model_config = ConfigDict(frozen=True)

    symbols: list[str] = Field(min_length=1)
    timeframe: str = "1D"
    start: str = "2020-01-01"
    end: str = "2025-12-31"
    capital: float = Field(default=100000.0, gt=0)


class StrategySection(BaseModel):
    """전략 섹션."""

    model_config = ConfigDict(frozen=True)

    name: str = "tsmom"
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """YAML 최상위 모델 — 모든 설정을 통합."""

    model_config = ConfigDict(frozen=True)

    backtest: BacktestScope
    strategy: StrategySection = Field(default_factory=StrategySection)
    portfolio: PortfolioManagerConfig = Field(default_factory=PortfolioManagerConfig)


def load_config(path: str | Path) -> RunConfig:
    """YAML → RunConfig (Pydantic 검증 포함).

    Args:
        path: YAML 설정 파일 경로

    Returns:
        검증된 RunConfig 인스턴스

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        yaml.YAMLError: YAML 파싱 실패
        pydantic.ValidationError: 검증 실패
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Config file not found: {file_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


def build_strategy(cfg: RunConfig) -> BaseStrategy:
    """RunConfig → 전략 인스턴스 (from_params 활용).

    Args:
        cfg: 검증된 RunConfig

    Returns:
        전략 인스턴스

    Raises:
        KeyError: 전략이 등록되지 않은 경우
    """
    strategy_cls = get_strategy(cfg.strategy.name)
    if cfg.strategy.params:
        return strategy_cls.from_params(**cfg.strategy.params)
    return strategy_cls()
