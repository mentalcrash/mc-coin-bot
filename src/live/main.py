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
    MC_EXECUTION_MODE=shadow mc-bot
"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from src.config.config_loader import build_strategy, load_config
from src.config.settings import DeploymentConfig, get_deployment_config
from src.core.logger import setup_logger
from src.eda.live_runner import LiveMode, LiveRunner
from src.exchange.binance_client import BinanceClient


def _resolve_live_mode(execution_mode: str) -> LiveMode:
    """execution_mode 문자열을 LiveMode로 변환.

    Args:
        execution_mode: "paper" | "shadow" | "live"

    Returns:
        LiveMode enum 값
    """
    if execution_mode == "shadow":
        return LiveMode.SHADOW
    # paper와 live 모두 PAPER 모드 사용 (live는 향후 BinanceExecutor 추가 시 분기)
    return LiveMode.PAPER


async def _run(config: DeploymentConfig) -> None:
    """비동기 메인 루프."""
    cfg = load_config(config.config_path)
    strategy = build_strategy(cfg)
    symbol_list = cfg.backtest.symbols

    tf = cfg.backtest.timeframe
    target_tf = tf.upper() if tf.lower() == "1d" else tf

    is_multi = len(symbol_list) > 1
    asset_weights: dict[str, float] | None = None
    if is_multi:
        asset_weights = {s: 1.0 / len(symbol_list) for s in symbol_list}

    live_mode = _resolve_live_mode(config.execution_mode)
    db_path = config.db_path if config.enable_persistence else None

    logger.info(
        "Starting LiveRunner: mode={}, strategy={}, symbols={}, tf=1m→{}",
        config.execution_mode,
        cfg.strategy.name,
        len(symbol_list),
        target_tf,
    )

    async with BinanceClient() as client:
        factory = LiveRunner.shadow if live_mode == LiveMode.SHADOW else LiveRunner.paper
        runner = factory(
            strategy=strategy,
            symbols=symbol_list,
            target_timeframe=target_tf,
            config=cfg.portfolio,
            client=client,
            initial_capital=config.initial_capital,
            asset_weights=asset_weights,
            db_path=db_path,
        )
        await runner.run()


def app() -> None:
    """Docker ENTRYPOINT — 환경 변수 기반 LiveRunner 실행."""
    config = get_deployment_config()

    setup_logger(console_level="INFO")
    logger.info("MC Coin Bot starting (mode={})", config.execution_mode)

    try:
        asyncio.run(_run(config))
    except KeyboardInterrupt:
        logger.info("Shutdown requested (KeyboardInterrupt)")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
