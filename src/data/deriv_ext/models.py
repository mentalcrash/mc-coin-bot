"""Coinalyze Extended Derivatives data models.

Aggregated OI, Funding Rate, Liquidation, CVD from multi-exchange sources.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for financial values
    - #10 Python Standards: Modern typing (X | None)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Threshold to distinguish Unix seconds vs milliseconds
_MS_THRESHOLD = 1e12


def _parse_date(v: str | int | float | datetime) -> datetime:
    """ISO string / Unix ms / Unix s → UTC datetime.

    Coinalyze timestamps are in seconds; detect by magnitude.
    """
    if isinstance(v, int | float):
        ts = v / 1000 if v > _MS_THRESHOLD else v
        return datetime.fromtimestamp(ts, tz=UTC)
    if isinstance(v, str):
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt
    if v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


class AggOIRecord(BaseModel):
    """Aggregated Open Interest (OHLC) 레코드.

    Attributes:
        date: UTC datetime
        symbol: Coinalyze symbol (e.g., "BTCUSDT.6")
        open: OI 시가
        high: OI 고가
        low: OI 저가
        close: OI 종가
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    source: str = "coinalyze"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix s → UTC datetime."""
        return _parse_date(v)


class AggFundingRecord(BaseModel):
    """Aggregated Funding Rate (OHLC) 레코드.

    Attributes:
        date: UTC datetime
        symbol: Coinalyze symbol
        open: Funding Rate 시가
        high: Funding Rate 고가
        low: Funding Rate 저가
        close: Funding Rate 종가
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    source: str = "coinalyze"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix s → UTC datetime."""
        return _parse_date(v)


class LiquidationRecord(BaseModel):
    """Liquidation (long/short volume) 레코드.

    Attributes:
        date: UTC datetime
        symbol: Coinalyze symbol
        long_volume: 롱 청산 볼륨
        short_volume: 숏 청산 볼륨
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    symbol: str
    long_volume: Decimal
    short_volume: Decimal
    source: str = "coinalyze"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix s → UTC datetime."""
        return _parse_date(v)


class CVDRecord(BaseModel):
    """Cumulative Volume Delta (OHLCV) 레코드.

    Attributes:
        date: UTC datetime
        symbol: Coinalyze symbol
        open: CVD 시가
        high: CVD 고가
        low: CVD 저가
        close: CVD 종가
        volume: 총 거래량
        buy_volume: 매수 거래량
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    buy_volume: Decimal
    source: str = "coinalyze"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix s → UTC datetime."""
        return _parse_date(v)


class HLAssetContextRecord(BaseModel):
    """Hyperliquid per-asset OI/Funding/Volume 스냅샷.

    Attributes:
        date: 수집 시점 (UTC)
        coin: 코인 심볼 (e.g., "BTC", "ETH")
        mark_price: 마크 가격
        open_interest: 미결제 약정
        funding: 현재 펀딩 레이트
        premium: 프리미엄 (선택)
        day_ntl_vlm: 24h notional 거래량
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    coin: str
    mark_price: Decimal
    open_interest: Decimal
    funding: Decimal
    premium: Decimal | None = None
    day_ntl_vlm: Decimal
    source: str = "hyperliquid"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix s → UTC datetime."""
        return _parse_date(v)


class HLPredictedFundingRecord(BaseModel):
    """Hyperliquid cross-venue predicted funding 비교.

    Attributes:
        date: 수집 시점 (UTC)
        coin: 코인 심볼 (e.g., "BTC", "ETH")
        venue: 거래소 (e.g., "Binance", "Bybit", "Hyperliquid")
        predicted_funding: 예상 펀딩 레이트
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    coin: str
    venue: str
    predicted_funding: Decimal
    source: str = "hyperliquid"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix s → UTC datetime."""
        return _parse_date(v)


class DerivExtBatch(BaseModel):
    """Extended Derivatives 데이터 배치 컨테이너.

    Attributes:
        source: 데이터 소스
        data_type: 데이터 유형
        records: 레코드 튜플
        fetched_at: 수집 시각 (UTC)
    """

    model_config = ConfigDict(frozen=True)

    source: str = "coinalyze"
    data_type: str
    records: tuple[
        AggOIRecord
        | AggFundingRecord
        | LiquidationRecord
        | CVDRecord
        | HLAssetContextRecord
        | HLPredictedFundingRecord,
        ...,
    ]
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("fetched_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """fetched_at에 UTC timezone 적용."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @property
    def count(self) -> int:
        """레코드 수."""
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        """레코드가 비어있는지 확인."""
        return len(self.records) == 0
