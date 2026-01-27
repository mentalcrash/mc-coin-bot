"""OHLCV Pydantic schemas for CCXT response validation.

This module defines immutable data models for candlestick (OHLCV) data
following the Schema-First approach. All external API data must pass
through these models before processing.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for prices
    - #10 Python Standards: Modern typing (X | None, list[])
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OHLCVCandle(BaseModel):
    """단일 캔들 데이터 (CCXT fetch_ohlcv 응답 매핑).

    CCXT의 fetch_ohlcv()는 [timestamp, open, high, low, close, volume] 형태의
    리스트를 반환합니다. 이 모델은 해당 응답을 검증하고 불변 객체로 변환합니다.

    Attributes:
        timestamp: 캔들 시작 시간 (UTC)
        open: 시가 (Decimal, > 0)
        high: 고가 (Decimal, > 0)
        low: 저가 (Decimal, > 0)
        close: 종가 (Decimal, > 0)
        volume: 거래량 (Decimal, >= 0)

    Example:
        >>> candle = OHLCVCandle(
        ...     timestamp=1704067200000,  # Unix ms
        ...     open=Decimal("42000.50"),
        ...     high=Decimal("42100.00"),
        ...     low=Decimal("41900.00"),
        ...     close=Decimal("42050.25"),
        ...     volume=Decimal("123.456"),
        ... )
    """

    model_config = ConfigDict(frozen=True)  # 불변 객체

    timestamp: datetime
    open: Decimal = Field(..., gt=0, description="시가")
    high: Decimal = Field(..., gt=0, description="고가")
    low: Decimal = Field(..., gt=0, description="저가")
    close: Decimal = Field(..., gt=0, description="종가")
    volume: Decimal = Field(..., ge=0, description="거래량")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """CCXT Unix ms 타임스탬프를 UTC datetime으로 변환.

        Args:
            v: Unix milliseconds (int/float) 또는 datetime 객체

        Returns:
            UTC timezone이 적용된 datetime 객체
        """
        if isinstance(v, int | float):
            # Unix ms -> seconds -> UTC datetime
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        # v is datetime at this point (after type narrowing)
        # timezone이 없으면 UTC로 설정
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @model_validator(mode="after")
    def validate_price_consistency(self) -> Self:
        """가격 일관성 검증: high >= low, high >= open/close, low <= open/close.

        Returns:
            검증된 self

        Raises:
            ValueError: 가격 일관성 위반 시
        """
        if self.high < self.low:
            msg = f"high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        if self.high < self.open or self.high < self.close:
            msg = f"high ({self.high}) must be >= open ({self.open}) and close ({self.close})"
            raise ValueError(msg)
        if self.low > self.open or self.low > self.close:
            msg = f"low ({self.low}) must be <= open ({self.open}) and close ({self.close})"
            raise ValueError(msg)
        return self


class OHLCVBatch(BaseModel):
    """배치 수집 결과 (메타데이터 포함).

    DataFetcher가 수집한 OHLCV 데이터와 관련 메타데이터를 담는 컨테이너입니다.
    Bronze 계층에 저장할 때 이 모델의 메타데이터도 함께 보존됩니다.

    Attributes:
        symbol: 거래 심볼 (예: "BTC/USDT")
        timeframe: 타임프레임 (예: "1m", "1h")
        exchange: 거래소 이름 (예: "binance")
        candles: 캔들 데이터 튜플 (불변)
        fetched_at: 수집 시각 (UTC)

    Example:
        >>> batch = OHLCVBatch(
        ...     symbol="BTC/USDT",
        ...     timeframe="1m",
        ...     exchange="binance",
        ...     candles=(candle1, candle2, ...),
        ...     fetched_at=datetime.now(UTC),
        ... )
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="거래 심볼 (예: BTC/USDT)")
    timeframe: str = Field(..., description="타임프레임 (예: 1m, 1h)")
    exchange: str = Field(..., description="거래소 이름")
    candles: tuple[OHLCVCandle, ...] = Field(
        default_factory=tuple,
        description="캔들 데이터 (불변 튜플)",
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="수집 시각 (UTC)",
    )

    @field_validator("fetched_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """fetched_at에 UTC timezone 적용.

        Args:
            v: datetime 객체

        Returns:
            UTC timezone이 적용된 datetime
        """
        # timezone이 없으면 UTC로 설정
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @property
    def candle_count(self) -> int:
        """캔들 개수 반환."""
        return len(self.candles)

    @property
    def is_empty(self) -> bool:
        """배치가 비어있는지 확인."""
        return len(self.candles) == 0

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        """캔들의 시간 범위 반환 (start, end).

        Returns:
            (첫 캔들 시각, 마지막 캔들 시각) 또는 비어있으면 None
        """
        if self.is_empty:
            return None
        return (self.candles[0].timestamp, self.candles[-1].timestamp)
