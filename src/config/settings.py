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
    onchain_bronze_dir: Path = Field(
        default=Path("data/bronze/onchain"),
        description="On-chain Bronze 데이터 저장 경로",
    )
    onchain_silver_dir: Path = Field(
        default=Path("data/silver/onchain"),
        description="On-chain Silver 데이터 저장 경로",
    )
    etherscan_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Etherscan API Key (ETH supply 수집용, 무료 tier)",
    )
    macro_bronze_dir: Path = Field(
        default=Path("data/bronze/macro"),
        description="Macro Bronze 데이터 저장 경로",
    )
    macro_silver_dir: Path = Field(
        default=Path("data/silver/macro"),
        description="Macro Silver 데이터 저장 경로",
    )
    fred_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="FRED API Key (Federal Reserve Economic Data)",
    )
    options_bronze_dir: Path = Field(
        default=Path("data/bronze/options"),
        description="Options Bronze 데이터 저장 경로",
    )
    options_silver_dir: Path = Field(
        default=Path("data/silver/options"),
        description="Options Silver 데이터 저장 경로",
    )
    deriv_ext_bronze_dir: Path = Field(
        default=Path("data/bronze/deriv_ext"),
        description="Extended Derivatives Bronze 데이터 저장 경로",
    )
    deriv_ext_silver_dir: Path = Field(
        default=Path("data/silver/deriv_ext"),
        description="Extended Derivatives Silver 데이터 저장 경로",
    )
    coinalyze_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Coinalyze API Key (Extended Derivatives)",
    )
    coingecko_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="CoinGecko Demo API Key",
    )
    dune_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Dune Analytics API Key (research only)",
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
    @field_validator(
        "bronze_dir",
        "silver_dir",
        "log_dir",
        "onchain_bronze_dir",
        "onchain_silver_dir",
        "macro_bronze_dir",
        "macro_silver_dir",
        "options_bronze_dir",
        "options_silver_dir",
        "deriv_ext_bronze_dir",
        "deriv_ext_silver_dir",
        mode="before",
    )
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

    def get_bronze_deriv_path(self, symbol: str, year: int) -> Path:
        """Bronze Derivatives Parquet 파일 경로 생성.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 연도

        Returns:
            Bronze Derivatives Parquet 파일 경로

        Example:
            >>> settings.get_bronze_deriv_path("BTC/USDT", 2025)
            PosixPath('data/bronze/BTC_USDT/2025_deriv.parquet')
        """
        safe_symbol = symbol.replace("/", "_")
        return self.bronze_dir / safe_symbol / f"{year}_deriv.parquet"

    def get_silver_deriv_path(self, symbol: str, year: int) -> Path:
        """Silver Derivatives Parquet 파일 경로 생성.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 연도

        Returns:
            Silver Derivatives Parquet 파일 경로

        Example:
            >>> settings.get_silver_deriv_path("BTC/USDT", 2025)
            PosixPath('data/silver/BTC_USDT/2025_deriv.parquet')
        """
        safe_symbol = symbol.replace("/", "_")
        return self.silver_dir / safe_symbol / f"{year}_deriv.parquet"

    def get_onchain_bronze_path(self, source: str, name: str) -> Path:
        """On-chain Bronze Parquet 파일 경로 생성.

        Args:
            source: 데이터 소스 (예: "defillama")
            name: 데이터 이름 (예: "stablecoin_total")

        Returns:
            Bronze Parquet 파일 경로

        Example:
            >>> settings.get_onchain_bronze_path("defillama", "stablecoin_total")
            PosixPath('data/bronze/onchain/defillama/stablecoin_total.parquet')
        """
        return self.onchain_bronze_dir / source / f"{name}.parquet"

    def get_onchain_silver_path(self, source: str, name: str) -> Path:
        """On-chain Silver Parquet 파일 경로 생성.

        Args:
            source: 데이터 소스 (예: "defillama")
            name: 데이터 이름 (예: "stablecoin_total")

        Returns:
            Silver Parquet 파일 경로

        Example:
            >>> settings.get_onchain_silver_path("defillama", "stablecoin_total")
            PosixPath('data/silver/onchain/defillama/stablecoin_total.parquet')
        """
        return self.onchain_silver_dir / source / f"{name}.parquet"

    def get_macro_bronze_path(self, source: str, name: str) -> Path:
        """Macro Bronze Parquet 파일 경로 생성.

        Args:
            source: 데이터 소스 (예: "fred")
            name: 데이터 이름 (예: "dxy")

        Returns:
            Bronze Parquet 파일 경로

        Example:
            >>> settings.get_macro_bronze_path("fred", "dxy")
            PosixPath('data/bronze/macro/fred/dxy.parquet')
        """
        return self.macro_bronze_dir / source / f"{name}.parquet"

    def get_macro_silver_path(self, source: str, name: str) -> Path:
        """Macro Silver Parquet 파일 경로 생성.

        Args:
            source: 데이터 소스 (예: "fred")
            name: 데이터 이름 (예: "dxy")

        Returns:
            Silver Parquet 파일 경로

        Example:
            >>> settings.get_macro_silver_path("fred", "dxy")
            PosixPath('data/silver/macro/fred/dxy.parquet')
        """
        return self.macro_silver_dir / source / f"{name}.parquet"

    def get_options_bronze_path(self, source: str, name: str) -> Path:
        """Options Bronze Parquet 파일 경로 생성.

        Args:
            source: 데이터 소스 (e.g., "deribit")
            name: 데이터 이름 (e.g., "btc_dvol")

        Returns:
            Bronze Parquet 파일 경로

        Example:
            >>> settings.get_options_bronze_path("deribit", "btc_dvol")
            PosixPath('data/bronze/options/deribit/btc_dvol.parquet')
        """
        return self.options_bronze_dir / source / f"{name}.parquet"

    def get_options_silver_path(self, source: str, name: str) -> Path:
        """Options Silver Parquet 파일 경로 생성.

        Args:
            source: 데이터 소스 (e.g., "deribit")
            name: 데이터 이름 (e.g., "btc_dvol")

        Returns:
            Silver Parquet 파일 경로

        Example:
            >>> settings.get_options_silver_path("deribit", "btc_dvol")
            PosixPath('data/silver/options/deribit/btc_dvol.parquet')
        """
        return self.options_silver_dir / source / f"{name}.parquet"

    def get_deriv_ext_bronze_path(self, source: str, name: str) -> Path:
        """Extended Derivatives Bronze Parquet 파일 경로 생성.

        Example:
            >>> settings.get_deriv_ext_bronze_path("coinalyze", "btc_agg_oi")
            PosixPath('data/bronze/deriv_ext/coinalyze/btc_agg_oi.parquet')
        """
        return self.deriv_ext_bronze_dir / source / f"{name}.parquet"

    def get_deriv_ext_silver_path(self, source: str, name: str) -> Path:
        """Extended Derivatives Silver Parquet 파일 경로 생성.

        Example:
            >>> settings.get_deriv_ext_silver_path("coinalyze", "btc_agg_oi")
            PosixPath('data/silver/deriv_ext/coinalyze/btc_agg_oi.parquet')
        """
        return self.deriv_ext_silver_dir / source / f"{name}.parquet"

    def ensure_directories(self) -> None:
        """필요한 디렉토리들을 생성.

        Bronze, Silver, Log, On-chain, Macro 디렉토리를 생성합니다.
        이미 존재하면 무시합니다.
        """
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.onchain_bronze_dir.mkdir(parents=True, exist_ok=True)
        self.onchain_silver_dir.mkdir(parents=True, exist_ok=True)
        self.macro_bronze_dir.mkdir(parents=True, exist_ok=True)
        self.macro_silver_dir.mkdir(parents=True, exist_ok=True)
        self.options_bronze_dir.mkdir(parents=True, exist_ok=True)
        self.options_silver_dir.mkdir(parents=True, exist_ok=True)
        self.deriv_ext_bronze_dir.mkdir(parents=True, exist_ok=True)
        self.deriv_ext_silver_dir.mkdir(parents=True, exist_ok=True)

    def has_api_credentials(self) -> bool:
        """API 자격 증명이 설정되어 있는지 확인.

        Returns:
            API 키와 시크릿이 모두 설정되어 있으면 True
        """
        return bool(
            self.binance_api_key.get_secret_value() and self.binance_secret.get_secret_value()
        )


class DeploymentConfig(BaseSettings):
    """배포 환경 설정 (MC_ 접두사).

    Docker/Coolify 환경에서 환경 변수로 실행 모드를 제어합니다.
    기존 YAML config + 환경 변수 오버라이드 패턴 유지.

    Environment Variables:
        - MC_EXECUTION_MODE: 실행 모드 (paper | shadow | live)
        - MC_CONFIG_PATH: YAML 설정 파일 경로
        - MC_INITIAL_CAPITAL: 초기 자본 (USD)
        - MC_DB_PATH: SQLite DB 경로 (빈 문자열 = 비활성)
        - MC_ENABLE_PERSISTENCE: 상태 영속화 on/off
    """

    model_config = SettingsConfigDict(
        env_prefix="MC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    execution_mode: str = Field(
        default="paper",
        description="실행 모드 (paper | shadow | live)",
    )
    config_path: str = Field(
        default="config/paper.yaml",
        description="YAML 설정 파일 경로",
    )
    initial_capital: float = Field(
        default=10000.0,
        gt=0,
        description="초기 자본 (USD)",
    )
    db_path: str = Field(
        default="data/trading.db",
        description="SQLite DB 경로 (빈 문자열 = 비활성)",
    )
    enable_persistence: bool = Field(
        default=True,
        description="상태 영속화 on/off",
    )
    metrics_port: int = Field(
        default=8000,
        ge=0,
        description="Prometheus metrics 포트 (0=비활성)",
    )

    @field_validator("execution_mode")
    @classmethod
    def validate_execution_mode(cls, v: str) -> str:
        """실행 모드 검증."""
        allowed = {"paper", "shadow", "live"}
        if v not in allowed:
            msg = f"execution_mode must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v


@lru_cache
def get_deployment_config() -> DeploymentConfig:
    """배포 설정 싱글톤 인스턴스 반환.

    Returns:
        DeploymentConfig 인스턴스
    """
    return DeploymentConfig()


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
