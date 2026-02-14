"""Docker ENTRYPOINT — 환경 변수 기반 LiveRunner 실행.

pyproject.toml의 `mc-bot = "src.live.main:app"` 진입점.
DeploymentConfig(MC_* 환경 변수)에서 설정을 로드하여
적절한 모드(paper/shadow/live)로 LiveRunner를 실행합니다.

YAML에 ``orchestrator:`` 키가 있으면 Orchestrated 모드로 자동 전환합니다.

Usage:
    # 직접 실행
    mc-bot

    # Docker
    docker run mc-coin-bot:latest

    # 환경 변수로 모드 오버라이드
    MC_EXECUTION_MODE=live MC_CONFIG_PATH=config/orchestrator-live.yaml mc-bot
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from loguru import logger

from src.config.settings import get_deployment_config
from src.core.logger import setup_logger


def _is_orchestrator_config(config_path: str) -> bool:
    """YAML 파일에 ``orchestrator:`` 키가 있는지 확인."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return isinstance(raw, dict) and "orchestrator" in raw


def app() -> None:
    """Docker ENTRYPOINT — 환경 변수 기반 LiveRunner 실행."""
    config = get_deployment_config()

    setup_logger(console_level="INFO")
    logger.info("MC Coin Bot starting (mode={})", config.execution_mode)

    db_path = config.db_path if config.enable_persistence else None

    try:
        if _is_orchestrator_config(config.config_path):
            from src.cli.orchestrate import launch_orchestrated_live

            logger.info("Orchestrator config detected: {}", config.config_path)
            launch_orchestrated_live(
                config.config_path,
                mode=config.execution_mode,
                db_path=db_path,
            )
        else:
            from src.cli.eda import launch_live

            launch_live(
                config.config_path,
                mode=config.execution_mode,
                initial_capital=config.initial_capital,
                db_path=db_path,
            )
    except KeyboardInterrupt:
        logger.info("Shutdown requested (KeyboardInterrupt)")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
