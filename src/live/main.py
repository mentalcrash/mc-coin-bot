"""Docker ENTRYPOINT — 환경 변수 기반 LiveRunner 실행.

pyproject.toml의 `mc-bot = "src.live.main:app"` 진입점.
DeploymentConfig(MC_* 환경 변수)에서 설정을 로드하여
적절한 모드(paper/shadow/live)로 LiveRunner를 실행합니다.

Usage:
    # 직접 실행
    mc-bot

    # Docker
    docker run mc-coin-bot:latest

    # 환경 변수로 모드 오버라이드
    MC_EXECUTION_MODE=live MC_CONFIG_PATH=config/spot_supertrend.yaml mc-bot
"""

from __future__ import annotations

import sys

from loguru import logger

from src.config.settings import get_deployment_config
from src.core.logger import setup_logger


def app() -> None:
    """Docker ENTRYPOINT — 환경 변수 기반 LiveRunner 실행."""
    config = get_deployment_config()

    setup_logger(console_level="INFO")
    logger.info("MC Coin Bot starting (mode={})", config.execution_mode)

    db_path = config.db_path if config.enable_persistence else None

    try:
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
