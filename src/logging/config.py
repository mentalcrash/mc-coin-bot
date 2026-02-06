"""Logging configuration models using Pydantic.

This module defines the configuration schema for the logging service.
All settings can be loaded from environment variables.

Rules Applied:
    - #11 Pydantic Modeling: Settings management, strict types
    - #15 Logging Standards: Configurable dual sinks
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    """Configuration for the logging service.

    Loaded from environment variables with LOG_ prefix.

    Attributes:
        log_dir: Directory for log files
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        rotation: File rotation policy (size or time)
        retention: How long to keep rotated files
        compression: Compression format for rotated files
        json_logs: Enable JSON format for file logs
        enable_discord: Enable Discord webhook alerts
        discord_webhook_url: Discord webhook URL for alerts
        discord_min_level: Minimum level for Discord alerts
        bounded_queue_size: Size of bounded queue for async sinks
        drop_policy: Policy when queue is full
    """

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # File paths
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    # Log levels
    console_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = (
        Field(
            default="INFO",
            description="Minimum level for console output",
        )
    )
    file_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = (
        Field(
            default="DEBUG",
            description="Minimum level for file output",
        )
    )

    # File rotation settings
    rotation: str = Field(
        default="50 MB",
        description="File rotation policy (e.g., '100 MB', '1 day')",
    )
    retention: str = Field(
        default="7 days",
        description="How long to keep rotated files",
    )
    compression: str = Field(
        default="gz",
        description="Compression format (gz, bz2, xz, lzma, tar, tar.gz, etc.)",
    )
    json_logs: bool = Field(
        default=True,
        description="Enable JSON serialization for file logs",
    )

    # Discord integration
    enable_discord: bool = Field(
        default=False,
        description="Enable Discord webhook alerts",
    )
    discord_webhook_url: str | None = Field(
        default=None,
        description="Discord webhook URL for error alerts",
    )
    discord_min_level: Literal["WARNING", "ERROR", "CRITICAL"] = Field(
        default="ERROR",
        description="Minimum level for Discord alerts",
    )

    # Bounded queue settings
    bounded_queue_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Size of bounded queue for async sinks",
    )
    drop_policy: Literal["oldest", "newest", "block"] = Field(
        default="oldest",
        description="Policy when queue is full",
    )

    # Diagnostics (security)
    diagnose: bool = Field(
        default=False,
        description="Enable diagnostic info in tracebacks (disable in prod)",
    )
    backtrace: bool = Field(
        default=True,
        description="Enable full traceback",
    )


class DiscordChannelConfig(BaseSettings):
    """Configuration for Discord channel segmentation.

    Different webhook URLs for different alert types.

    Attributes:
        trade_webhook_url: Webhook for trade alerts (buy/sell)
        error_webhook_url: Webhook for error alerts
        report_webhook_url: Webhook for daily reports
    """

    model_config = SettingsConfigDict(
        env_prefix="DISCORD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    trade_webhook_url: str | None = Field(
        default=None,
        description="Webhook URL for trade alerts",
    )
    error_webhook_url: str | None = Field(
        default=None,
        description="Webhook URL for error alerts",
    )
    report_webhook_url: str | None = Field(
        default=None,
        description="Webhook URL for daily reports",
    )


def get_logging_config() -> LoggingConfig:
    """Load logging configuration from environment.

    Returns:
        LoggingConfig instance with values from env vars
    """
    return LoggingConfig()


def get_discord_config() -> DiscordChannelConfig:
    """Load Discord channel configuration from environment.

    Returns:
        DiscordChannelConfig instance with values from env vars
    """
    return DiscordChannelConfig()
