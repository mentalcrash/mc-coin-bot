"""Loguru logging configuration with production-ready features.

This module provides a centralized logging setup following the project's
logging standards (Rules #15). All logging in the application should use
the configured loguru logger.

Features:
    - Dual sinks: Console (human-readable) + File (JSON serialized)
    - Bounded queue for memory-safe async logging (solves Issue #1419)
    - Discord webhook integration for ERROR+ alerts
    - OpenTelemetry context injection for observability
    - Structured logging with context binding

Rules Applied:
    - #15 Logging Standards: Loguru, dual sinks, bounded queue
    - #22 Notification Standards: Discord webhooks for alerts
    - #23 Exception Handling: Alert hooks for critical errors
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.logging.config import LoggingConfig, get_logging_config
from src.logging.context import get_trading_logger
from src.logging.sinks.bounded_queue import BoundedQueueSink, DropPolicy
from src.logging.sinks.discord import DiscordWebhookSink
from src.logging.sinks.otel import create_otel_patcher, is_otel_available

if TYPE_CHECKING:
    from loguru import Logger

# =============================================================================
# Module-level logger (re-exported for convenience)
# =============================================================================

# Remove default handler to prevent duplicate logs
logger.remove()


# =============================================================================
# Console Format Templates
# =============================================================================

CONSOLE_FORMAT_DEFAULT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

CONSOLE_FORMAT_WITH_CONTEXT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<dim>[{extra[symbol]}/{extra[strategy]}]</dim> "
    "<level>{message}</level>"
)


# =============================================================================
# Sink Manager (encapsulates global state)
# =============================================================================


class _SinkManager:
    """Manages active sink references for cleanup.

    This class encapsulates the global state for sinks, avoiding
    the use of global statements which are discouraged by linters.
    """

    def __init__(self) -> None:
        self.active_sinks: list[BoundedQueueSink] = []
        self.discord_sink: DiscordWebhookSink | None = None

    def add_sink(self, sink: BoundedQueueSink) -> None:
        """Add a sink to the active list."""
        self.active_sinks.append(sink)

    def set_discord_sink(self, sink: DiscordWebhookSink) -> None:
        """Set the Discord sink reference."""
        self.discord_sink = sink

    def cleanup(self) -> None:
        """Clean up all active sinks."""
        for sink in self.active_sinks:
            sink.stop()
        self.active_sinks.clear()

        if self.discord_sink is not None:
            self.discord_sink.stop()
            self.discord_sink = None


# Module-level sink manager instance
_sink_manager = _SinkManager()


# =============================================================================
# Setup Functions
# =============================================================================


def setup_logger_from_config(config: LoggingConfig | None = None) -> None:
    """Initialize logger from Pydantic config model.

    This is the recommended way to set up the logger. It loads
    configuration from environment variables if not provided.

    Args:
        config: LoggingConfig instance (loads from env if None)

    Example:
        >>> from src.core.logger import setup_logger_from_config
        >>> setup_logger_from_config()  # Loads from LOG_* env vars
    """
    if config is None:
        config = get_logging_config()

    _setup_logger_internal(config)


def setup_logger(
    log_dir: Path | str = Path("logs"),
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    *,
    enable_discord: bool = False,
    discord_webhook_url: str | None = None,
) -> None:
    """Initialize the logger with minimal configuration.

    For full configuration options, use setup_logger_from_config()
    with a LoggingConfig instance.

    Args:
        log_dir: Directory for log files (default: "logs")
        console_level: Console output level (default: "INFO")
        file_level: File output level (default: "DEBUG")
        enable_discord: Enable Discord webhook alerts
        discord_webhook_url: Discord webhook URL for alerts

    Example:
        >>> from src.core.logger import setup_logger, logger
        >>> setup_logger(log_dir="logs", console_level="DEBUG")
        >>> logger.info("Application started")
    """
    config = LoggingConfig(
        log_dir=Path(log_dir),
        console_level=console_level,  # type: ignore[arg-type]
        file_level=file_level,  # type: ignore[arg-type]
        enable_discord=enable_discord,
        discord_webhook_url=discord_webhook_url,
    )
    _setup_logger_internal(config)


def _setup_logger_internal(config: LoggingConfig) -> None:
    """Internal logger setup using config object.

    Args:
        config: LoggingConfig instance with all settings
    """
    # Remove existing handlers and clean up sinks
    logger.remove()
    _sink_manager.cleanup()

    # Create log directory
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Apply OpenTelemetry patcher if available
    patched_logger = logger
    if is_otel_available():
        patcher = create_otel_patcher()
        patched_logger = logger.patch(patcher)

    # 1. Console Handler (Human-readable, synchronous)
    patched_logger.add(
        sys.stderr,
        format=CONSOLE_FORMAT_DEFAULT,
        level=config.console_level,
        colorize=True,
        backtrace=config.backtrace,
        diagnose=config.diagnose,
    )

    # 2. File Handler with Bounded Queue (JSON, async-safe)
    if config.json_logs:
        _setup_json_file_sink(patched_logger, log_path, config)
    else:
        _setup_text_file_sink(patched_logger, log_path, config)

    # 3. Discord Handler (optional, for ERROR+)
    if config.enable_discord and config.discord_webhook_url:
        _setup_discord_sink(config.discord_webhook_url, config.discord_min_level)

    logger.info(
        "Logger initialized",
        log_dir=str(log_path),
        console_level=config.console_level,
        file_level=config.file_level,
        discord_enabled=config.enable_discord,
        otel_enabled=is_otel_available(),
    )


def _setup_json_file_sink(
    patched_logger: Logger,
    log_path: Path,
    config: LoggingConfig,
) -> None:
    """Set up JSON file sink with bounded queue.

    Args:
        patched_logger: Logger instance (possibly with OTel patcher)
        log_path: Path to log directory
        config: Logging configuration
    """
    # Create file path template
    file_template = log_path / "trading_{time}.json"

    def file_sink_writer(msg: str) -> None:
        current_time = datetime.now(UTC).strftime("%Y-%m-%d_%H")
        file_name = str(file_template).replace("{time}", current_time)
        with Path(file_name).open("a", encoding="utf-8") as f:
            f.write(msg)

    bounded_file_sink = BoundedQueueSink(
        sink=file_sink_writer,
        maxsize=config.bounded_queue_size,
        drop_policy=DropPolicy.OLDEST,
        name="FileWriter",
    )
    _sink_manager.add_sink(bounded_file_sink)

    patched_logger.add(
        bounded_file_sink.write,
        format="{message}",
        level=config.file_level,
        serialize=True,
        backtrace=config.backtrace,
        diagnose=False,
    )


def _setup_text_file_sink(
    patched_logger: Logger,
    log_path: Path,
    config: LoggingConfig,
) -> None:
    """Set up text file sink with loguru's built-in rotation.

    Args:
        patched_logger: Logger instance
        log_path: Path to log directory
        config: Logging configuration
    """
    patched_logger.add(
        log_path / "trading_{time:YYYY-MM-DD}.log",
        format=CONSOLE_FORMAT_DEFAULT,
        level=config.file_level,
        rotation=config.rotation,
        retention=config.retention,
        compression=config.compression,
        enqueue=True,
        backtrace=config.backtrace,
        diagnose=False,
    )


def _setup_discord_sink(webhook_url: str, min_level: str = "ERROR") -> None:
    """Set up Discord webhook sink for alerts.

    Args:
        webhook_url: Discord webhook URL
        min_level: Minimum level to forward to Discord
    """
    discord_sink = DiscordWebhookSink(
        webhook_url=webhook_url,
        min_level=min_level,
    )
    _sink_manager.set_discord_sink(discord_sink)

    logger.add(
        discord_sink.write,
        level=min_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


# =============================================================================
# Context Logger Functions (Backward Compatibility)
# =============================================================================


def get_context_logger(
    *,
    symbol: str | None = None,
    exchange: str | None = None,
    operation: str | None = None,
    **extra: str,
) -> Logger:
    """Get a context-bound logger (backward compatible).

    This function is maintained for backward compatibility.
    For new code, use get_trading_logger from src.logging.context.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT")
        exchange: Exchange name (e.g., "binance")
        operation: Operation type (e.g., "fetch", "save")
        **extra: Additional context key-values

    Returns:
        Logger with context bound

    Example:
        >>> ctx_logger = get_context_logger(symbol="BTC/USDT", operation="fetch")
        >>> ctx_logger.info("Fetching data...")
    """
    return get_trading_logger(
        symbol=symbol,
        exchange=exchange,
        operation=operation,  # type: ignore[arg-type]
        **extra,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "get_context_logger",
    "get_trading_logger",
    "logger",
    "setup_logger",
    "setup_logger_from_config",
]
