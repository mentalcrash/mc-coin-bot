"""Orchestrator YAML 설정 파일 로더.

YAML 파일에서 OrchestratorRunConfig를 로드합니다.
기존 config_loader.py와 동일한 패턴을 따릅니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Field validators
    - #10 Python Standards: Modern typing, Path
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.orchestrator.config import OrchestratorConfig


class OrchestratorBacktestScope(BaseModel):
    """Orchestrator 백테스트 범위.

    symbols는 pods에서 자동 유도되므로 별도 필드 없음.

    Attributes:
        timeframe: 타임프레임
        start: 시작 날짜 (YYYY-MM-DD)
        end: 종료 날짜 (YYYY-MM-DD)
        capital: 초기 자본 (USD)
    """

    model_config = ConfigDict(frozen=True)

    timeframe: str = "1D"
    start: str = "2020-01-01"
    end: str = "2025-12-31"
    capital: float = Field(default=100_000.0, gt=0)


class OrchestratorRunConfig(BaseModel):
    """YAML 최상위 모델 — orchestrator + backtest 설정 통합.

    Attributes:
        orchestrator: OrchestratorConfig (pods, allocation, risk 등)
        backtest: OrchestratorBacktestScope (timeframe, start, end, capital)
    """

    model_config = ConfigDict(frozen=True)

    orchestrator: OrchestratorConfig
    backtest: OrchestratorBacktestScope = Field(
        default_factory=OrchestratorBacktestScope,
    )


def load_orchestrator_config(path: str | Path) -> OrchestratorRunConfig:
    """YAML → OrchestratorRunConfig (Pydantic 검증 포함).

    Args:
        path: YAML 설정 파일 경로

    Returns:
        검증된 OrchestratorRunConfig 인스턴스

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
    return OrchestratorRunConfig.model_validate(raw)
