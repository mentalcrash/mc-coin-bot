"""Derivatives data models for Binance Futures.

Funding Rate, Open Interest, Long/Short Ratio, Taker Buy/Sell Ratio,
Top Trader Account/Position Ratio 데이터를 위한 Pydantic V2 frozen 모델.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for financial values
    - #10 Python Standards: Modern typing (X | None)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FundingRateRecord(BaseModel):
    """Funding Rate 레코드.

    Attributes:
        symbol: 거래 심볼 (예: "BTC/USDT")
        timestamp: Funding 시각 (UTC)
        funding_rate: Funding Rate (예: 0.0001 = 0.01%)
        mark_price: Mark Price (USD)
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    funding_rate: Decimal
    mark_price: Decimal = Field(..., gt=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """Unix ms → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class OpenInterestRecord(BaseModel):
    """Open Interest 레코드.

    Attributes:
        symbol: 거래 심볼
        timestamp: 타임스탬프 (UTC)
        sum_open_interest: 총 미결제약정 (계약 수)
        sum_open_interest_value: 총 미결제약정 가치 (USD)
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    sum_open_interest: Decimal = Field(..., ge=0)
    sum_open_interest_value: Decimal = Field(..., ge=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """Unix ms → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class LongShortRatioRecord(BaseModel):
    """Global Long/Short Account Ratio 레코드.

    Attributes:
        symbol: 거래 심볼
        timestamp: 타임스탬프 (UTC)
        long_account: Long 계정 비율 (0~1)
        short_account: Short 계정 비율 (0~1)
        long_short_ratio: Long/Short 비율
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    long_account: Decimal = Field(..., ge=0, le=1)
    short_account: Decimal = Field(..., ge=0, le=1)
    long_short_ratio: Decimal = Field(..., ge=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """Unix ms → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class TakerRatioRecord(BaseModel):
    """Taker Buy/Sell Volume Ratio 레코드.

    Attributes:
        symbol: 거래 심볼
        timestamp: 타임스탬프 (UTC)
        buy_vol: Taker Buy Volume
        sell_vol: Taker Sell Volume
        buy_sell_ratio: Buy/Sell Ratio
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    buy_vol: Decimal = Field(..., ge=0)
    sell_vol: Decimal = Field(..., ge=0)
    buy_sell_ratio: Decimal = Field(..., ge=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """Unix ms → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class TopTraderAccountRatioRecord(BaseModel):
    """Top Trader Long/Short Account Ratio 레코드.

    Attributes:
        symbol: 거래 심볼
        timestamp: 타임스탬프 (UTC)
        long_account: Long 계정 비율 (0~1)
        short_account: Short 계정 비율 (0~1)
        long_short_ratio: Long/Short 비율
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    long_account: Decimal = Field(..., ge=0, le=1)
    short_account: Decimal = Field(..., ge=0, le=1)
    long_short_ratio: Decimal = Field(..., ge=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """Unix ms → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class TopTraderPositionRatioRecord(BaseModel):
    """Top Trader Long/Short Position Ratio 레코드.

    Attributes:
        symbol: 거래 심볼
        timestamp: 타임스탬프 (UTC)
        long_account: Long 포지션 비율 (0~1)
        short_account: Short 포지션 비율 (0~1)
        long_short_ratio: Long/Short 비율
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    long_account: Decimal = Field(..., ge=0, le=1)
    short_account: Decimal = Field(..., ge=0, le=1)
    long_short_ratio: Decimal = Field(..., ge=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | float | datetime) -> datetime:
        """Unix ms → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class DerivativesBatch(BaseModel):
    """파생상품 데이터 배치 (수집 결과 컨테이너).

    Attributes:
        symbol: 거래 심볼
        funding_rates: Funding Rate 레코드 튜플
        open_interest: Open Interest 레코드 튜플
        long_short_ratios: Long/Short Ratio 레코드 튜플
        taker_ratios: Taker Ratio 레코드 튜플
        top_acct_ratios: Top Trader Account Ratio 레코드 튜플
        top_pos_ratios: Top Trader Position Ratio 레코드 튜플
        fetched_at: 수집 시각 (UTC)
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    funding_rates: tuple[FundingRateRecord, ...] = Field(default_factory=tuple)
    open_interest: tuple[OpenInterestRecord, ...] = Field(default_factory=tuple)
    long_short_ratios: tuple[LongShortRatioRecord, ...] = Field(default_factory=tuple)
    taker_ratios: tuple[TakerRatioRecord, ...] = Field(default_factory=tuple)
    top_acct_ratios: tuple[TopTraderAccountRatioRecord, ...] = Field(default_factory=tuple)
    top_pos_ratios: tuple[TopTraderPositionRatioRecord, ...] = Field(default_factory=tuple)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("fetched_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """fetched_at에 UTC timezone 적용."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @property
    def is_empty(self) -> bool:
        """모든 데이터가 비어있는지 확인."""
        return (
            len(self.funding_rates) == 0
            and len(self.open_interest) == 0
            and len(self.long_short_ratios) == 0
            and len(self.taker_ratios) == 0
            and len(self.top_acct_ratios) == 0
            and len(self.top_pos_ratios) == 0
        )

    @property
    def total_records(self) -> int:
        """전체 레코드 수."""
        return (
            len(self.funding_rates)
            + len(self.open_interest)
            + len(self.long_short_ratios)
            + len(self.taker_ratios)
            + len(self.top_acct_ratios)
            + len(self.top_pos_ratios)
        )
