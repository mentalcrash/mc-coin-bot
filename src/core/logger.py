"""Loguru logging configuration with dual sinks.

This module provides a centralized logging setup following the project's
logging standards (Rules #15). All logging in the application should use
the configured loguru logger.

Features:
    - Dual sinks: Console (human-readable) + File (JSON serialized)
    - Async-safe with enqueue=True
    - Rotation: 100MB / Retention: 30 days
    - Structured logging with context binding

Rules Applied:
    - #15 Logging Standards: Loguru, dual sinks, enqueue=True
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# 기본 핸들러 제거 (중복 로그 방지)
logger.remove()


def setup_logger(
    log_dir: Path | str = Path("logs"),
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """로거 설정 초기화.

    듀얼 싱크(Console + File)로 로거를 구성합니다.
    - Console: 사람이 읽기 쉬운 컬러 포맷
    - File: JSON 직렬화 (ELK 스택 연동 가능)

    Args:
        log_dir: 로그 파일 저장 디렉토리 (기본: "logs")
        console_level: 콘솔 출력 로그 레벨 (기본: "INFO")
        file_level: 파일 출력 로그 레벨 (기본: "DEBUG")
        rotation: 파일 로테이션 조건 (기본: "100 MB")
        retention: 파일 보관 기간 (기본: "30 days")

    Example:
        >>> from src.core.logger import setup_logger, logger
        >>> setup_logger(log_dir="logs", console_level="DEBUG")
        >>> logger.info("Application started")
    """
    # 기존 핸들러 모두 제거
    logger.remove()

    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 1. Console Handler (Human-readable)
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=console_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 2. File Handler (JSON, Async-safe)
    logger.add(
        log_path / "ingestion_{time:YYYY-MM-DD}.json",
        format="{message}",
        level=file_level,
        rotation=rotation,
        retention=retention,
        compression="gz",
        serialize=True,  # JSON 직렬화
        enqueue=True,  # [중요] 비동기 안전 - 이벤트 루프 차단 방지
        backtrace=True,
        diagnose=False,  # 프로덕션에서는 민감 정보 노출 방지
    )

    logger.info(
        "Logger initialized",
        extra={
            "log_dir": str(log_path),
            "console_level": console_level,
            "file_level": file_level,
        },
    )


def get_context_logger(
    *,
    symbol: str | None = None,
    exchange: str | None = None,
    operation: str | None = None,
    **extra: str,
) -> "Logger":
    """컨텍스트가 바인딩된 로거 반환.

    비동기 환경에서 로그가 뒤섞이는 것을 방지하기 위해
    logger.bind()를 사용하여 메타데이터를 주입합니다.

    Args:
        symbol: 거래 심볼 (예: "BTC/USDT")
        exchange: 거래소 이름 (예: "binance")
        operation: 작업 유형 (예: "fetch", "save")
        **extra: 추가 컨텍스트 키-값

    Returns:
        컨텍스트가 바인딩된 loguru logger

    Example:
        >>> ctx_logger = get_context_logger(symbol="BTC/USDT", operation="fetch")
        >>> ctx_logger.info("Fetching data...")
    """
    context: dict[str, str] = {}
    if symbol:
        context["symbol"] = symbol
    if exchange:
        context["exchange"] = exchange
    if operation:
        context["operation"] = operation
    context.update(extra)
    return logger.bind(**context)


# 모듈 레벨에서 logger 재export (편의성)
__all__ = ["get_context_logger", "logger", "setup_logger"]
