"""Pydantic Settings for configuration management.

This module provides centralized configuration management using
pydantic-settings. All settings are loaded from environment variables
and/or .env files with type validation.

Features:
    - SecretStr for API keys (auto-masking in logs)
    - Bronze/Silver directory path configuration
    - Rate limit and retry parameters
    - Environment variable loading from .env

Rules Applied:
    - #11 Pydantic Modeling: BaseSettings, SecretStr
    - #19 Git Security: No secrets in code
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """데이터 수집 파이프라인 설정.

    환경 변수 또는 .env 파일에서 설정을 로드합니다.
    API 키는 SecretStr로 보호되어 로그에 노출되지 않습니다.

    Environment Variables:
        - BINANCE_API_KEY: Binance API 키
        - BINANCE_SECRET: Binance Secret 키
        - BRONZE_DIR: Bronze 데이터 저장 경로 (기본: data/bronze)
        - SILVER_DIR: Silver 데이터 저장 경로 (기본: data/silver)
        - LOG_DIR: 로그 저장 경로 (기본: logs)

    Example:
        >>> settings = get_settings()
        >>> print(settings.bronze_dir)
        PosixPath('data/bronze')
        >>> print(settings.binance_api_key)
        SecretStr('**********')
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 알 수 없는 환경 변수 무시
    )

    # ==========================================================================
    # API Credentials (SecretStr for security)
    # ==========================================================================
    binance_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Binance API Key",
    )
    binance_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Binance Secret Key",
    )

    # ==========================================================================
    # Directory Paths (Medallion Architecture)
    # ==========================================================================
    bronze_dir: Path = Field(
        default=Path("data/bronze"),
        description="Bronze 계층 데이터 저장 경로 (원본)",
    )
    silver_dir: Path = Field(
        default=Path("data/silver"),
        description="Silver 계층 데이터 저장 경로 (정제)",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="로그 파일 저장 경로",
    )

    # ==========================================================================
    # API Rate Limiting
    # ==========================================================================
    rate_limit_per_minute: int = Field(
        default=1200,
        ge=1,
        description="분당 최대 API 요청 수",
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        description="API 요청 타임아웃 (초)",
    )

    # ==========================================================================
    # Retry Configuration
    # ==========================================================================
    max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="최대 재시도 횟수",
    )
    backoff_base: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="지수 백오프 기준 값",
    )
    backoff_max: float = Field(
        default=60.0,
        ge=10.0,
        description="최대 백오프 대기 시간 (초)",
    )

    # ==========================================================================
    # Data Fetching
    # ==========================================================================
    batch_size: int = Field(
        default=1000,
        ge=100,
        le=1500,
        description="배치당 캔들 수 (CCXT 기본 제한: 1000-1500)",
    )
    default_timeframe: str = Field(
        default="1m",
        description="기본 타임프레임",
    )

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("bronze_dir", "silver_dir", "log_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """문자열을 Path 객체로 변환.

        Args:
            v: 경로 문자열 또는 Path 객체

        Returns:
            Path 객체
        """
        return Path(v) if isinstance(v, str) else v

    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    def get_bronze_path(self, symbol: str, year: int) -> Path:
        """Bronze Parquet 파일 경로 생성.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 연도

        Returns:
            Bronze Parquet 파일 경로

        Example:
            >>> settings.get_bronze_path("BTC/USDT", 2025)
            PosixPath('data/bronze/BTC_USDT/2025.parquet')
        """
        safe_symbol = symbol.replace("/", "_")
        return self.bronze_dir / safe_symbol / f"{year}.parquet"

    def get_silver_path(self, symbol: str, year: int) -> Path:
        """Silver Parquet 파일 경로 생성.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 연도

        Returns:
            Silver Parquet 파일 경로

        Example:
            >>> settings.get_silver_path("BTC/USDT", 2025)
            PosixPath('data/silver/BTC_USDT/2025.parquet')
        """
        safe_symbol = symbol.replace("/", "_")
        return self.silver_dir / safe_symbol / f"{year}.parquet"

    def ensure_directories(self) -> None:
        """필요한 디렉토리들을 생성.

        Bronze, Silver, Log 디렉토리를 생성합니다.
        이미 존재하면 무시합니다.
        """
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def has_api_credentials(self) -> bool:
        """API 자격 증명이 설정되어 있는지 확인.

        Returns:
            API 키와 시크릿이 모두 설정되어 있으면 True
        """
        return bool(
            self.binance_api_key.get_secret_value()
            and self.binance_secret.get_secret_value()
        )


@lru_cache
def get_settings() -> IngestionSettings:
    """설정 싱글톤 인스턴스 반환.

    lru_cache를 사용하여 설정 객체를 캐싱합니다.
    애플리케이션 전체에서 동일한 설정 인스턴스를 사용합니다.

    Returns:
        IngestionSettings 인스턴스

    Example:
        >>> settings = get_settings()
        >>> settings.bronze_dir
        PosixPath('data/bronze')
    """
    return IngestionSettings()


def clear_settings_cache() -> None:
    """설정 캐시 초기화 (테스트용).

    테스트에서 설정을 재로드해야 할 때 사용합니다.
    """
    get_settings.cache_clear()
