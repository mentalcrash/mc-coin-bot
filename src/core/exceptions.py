"""Custom exception hierarchy for the trading system.

This module defines a domain-driven exception hierarchy following
Rules #23 (Exception Handling Standards). Exceptions are categorized
by their nature and expected handling behavior.

Exception Categories:
    - Recoverable (Retry): Network/Infrastructure errors
    - Unrecoverable (Fail Fast): Logic errors, Critical errors

Rules Applied:
    - #23 Exception Handling: Domain-driven hierarchy, add_note()
"""


class TradingError(Exception):
    """모든 트레이딩 관련 예외의 기본 클래스.

    이 예외를 직접 발생시키지 말고, 하위 클래스를 사용하세요.

    Attributes:
        message: 에러 메시지
        context: 추가 컨텍스트 정보 (디버깅용)
    """

    def __init__(
        self, message: str, *, context: dict[str, object] | None = None
    ) -> None:
        """TradingError 초기화.

        Args:
            message: 에러 메시지
            context: 추가 컨텍스트 정보 (선택)
        """
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context or {}

    def __str__(self) -> str:
        """에러 메시지와 컨텍스트를 포함한 문자열 반환."""
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# =============================================================================
# Infrastructure Errors (Recoverable - Retry)
# =============================================================================


class InfrastructureError(TradingError):
    """인프라 관련 오류 (파일 I/O, DB 연결 등).

    이 오류는 일시적일 수 있으므로 재시도 대상입니다.

    Example:
        >>> raise InfrastructureError(
        ...     "Failed to write parquet file",
        ...     context={"path": "/data/bronze/BTC_USDT/2025.parquet"}
        ... )
    """


class StorageError(InfrastructureError):
    """저장소 관련 오류 (Parquet 읽기/쓰기 실패 등)."""


# =============================================================================
# Exchange Errors (Recoverable - Retry with backoff)
# =============================================================================


class ExchangeError(TradingError):
    """거래소 API 관련 오류의 기본 클래스.

    네트워크 문제나 API 제한으로 인한 일시적 오류입니다.
    지수 백오프로 재시도해야 합니다.
    """


class NetworkError(ExchangeError):
    """네트워크 연결 오류 (타임아웃, 연결 실패 등).

    Example:
        >>> raise NetworkError(
        ...     "Connection timeout",
        ...     context={"exchange": "binance", "timeout": 30}
        ... )
    """


class RateLimitError(ExchangeError):
    """API 레이트 리밋 초과 (HTTP 429).

    이 오류 발생 시 반드시 대기 후 재시도해야 합니다.

    Attributes:
        retry_after: 재시도까지 대기 시간 (초)
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """RateLimitError 초기화.

        Args:
            message: 에러 메시지
            retry_after: 재시도까지 대기 시간 (초)
            context: 추가 컨텍스트 정보
        """
        super().__init__(message, context=context)
        self.retry_after = retry_after


class AuthenticationError(ExchangeError):
    """API 인증 오류 (잘못된 API 키, 만료된 키 등).

    이 오류는 재시도로 해결되지 않습니다.
    API 키 설정을 확인해야 합니다.
    """


# =============================================================================
# Data Validation Errors (Unrecoverable - Log and Skip)
# =============================================================================


class DataValidationError(TradingError):
    """데이터 검증 오류 (스키마 불일치, 유효하지 않은 값 등).

    Pydantic 검증 실패나 비즈니스 로직 검증 실패 시 발생합니다.

    Example:
        >>> raise DataValidationError(
        ...     "Price cannot be negative",
        ...     context={"symbol": "BTC/USDT", "price": -100}
        ... )
    """


class SchemaValidationError(DataValidationError):
    """Pydantic 스키마 검증 오류."""


class DataIntegrityError(DataValidationError):
    """데이터 무결성 오류 (결측치, 중복 등)."""


# =============================================================================
# Critical Errors (Unrecoverable - Immediate Stop)
# =============================================================================


class CriticalError(TradingError):
    """치명적 오류 (즉시 시스템 중단 필요).

    이 오류 발생 시 Kill Switch를 발동하고 관리자에게 알림을 보내야 합니다.

    Example:
        >>> raise CriticalError(
        ...     "API key possibly compromised",
        ...     context={"reason": "unexpected withdrawal attempt"}
        ... )
    """


# =============================================================================
# Utility Functions
# =============================================================================


def add_context_note(exc: Exception, note: str) -> None:
    """예외에 컨텍스트 노트 추가 (Python 3.11+ add_note).

    원본 Traceback을 보존하면서 디버깅 정보를 추가합니다.

    Args:
        exc: 예외 객체
        note: 추가할 노트 문자열

    Example:
        >>> try:
        ...     await fetch_data(symbol)
        ... except Exception as e:
        ...     add_context_note(e, f"Failed while fetching {symbol}")
        ...     raise
    """
    exc.add_note(note)
