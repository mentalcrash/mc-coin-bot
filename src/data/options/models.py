"""Deribit Options data models (DVOL, P/C Ratio, Historical Vol, Term Structure, Max Pain).

Pydantic V2 frozen models with Decimal for financial values.

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

    Deribit timestamps are in milliseconds.
    """
    if isinstance(v, int | float):
        # Deribit uses milliseconds; detect by magnitude
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


class DVolRecord(BaseModel):
    """DVOL (30D Implied Volatility) 레코드.

    Attributes:
        date: UTC datetime
        currency: "BTC" or "ETH"
        open: DVOL 시가 (annualized %)
        high: DVOL 고가
        low: DVOL 저가
        close: DVOL 종가
        volume: 거래량
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    currency: str
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal
    volume: Decimal | None = None
    source: str = "deribit"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix ms → UTC datetime."""
        return _parse_date(v)


class PutCallRatioRecord(BaseModel):
    """Put/Call OI Ratio 스냅샷.

    Attributes:
        date: 스냅샷 시각 (UTC)
        currency: "BTC" or "ETH"
        put_oi: 총 Put OI (contracts)
        call_oi: 총 Call OI (contracts)
        pc_ratio: put_oi / call_oi
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    currency: str
    put_oi: Decimal
    call_oi: Decimal
    pc_ratio: Decimal
    source: str = "deribit"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix ms → UTC datetime."""
        return _parse_date(v)


class HistoricalVolRecord(BaseModel):
    """Realized Volatility (다중 윈도우).

    Attributes:
        date: UTC datetime
        currency: "BTC" or "ETH"
        vol_7d ~ vol_365d: 각 윈도우별 실현 변동성
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    currency: str
    vol_7d: Decimal | None = None
    vol_30d: Decimal | None = None
    vol_60d: Decimal | None = None
    vol_90d: Decimal | None = None
    vol_120d: Decimal | None = None
    vol_180d: Decimal | None = None
    vol_365d: Decimal | None = None
    source: str = "deribit"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix ms → UTC datetime."""
        return _parse_date(v)


class TermStructureRecord(BaseModel):
    """Futures Term Structure 스냅샷.

    Attributes:
        date: 스냅샷 시각 (UTC)
        currency: "BTC" or "ETH"
        near_expiry: 근월물 instrument 이름
        far_expiry: 원월물 instrument 이름
        near_basis_pct: (near_price - index) / index * 100
        far_basis_pct: (far_price - index) / index * 100
        slope: far_basis - near_basis (양수=contango, 음수=backwardation)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    currency: str
    near_expiry: str
    far_expiry: str
    near_basis_pct: Decimal
    far_basis_pct: Decimal
    slope: Decimal
    source: str = "deribit"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix ms → UTC datetime."""
        return _parse_date(v)


class MaxPainRecord(BaseModel):
    """Options Max Pain 스냅샷.

    Attributes:
        date: 스냅샷 시각 (UTC)
        currency: "BTC" or "ETH"
        expiry: 만기일
        max_pain_strike: Max Pain 행사가
        total_oi: 전체 OI
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    currency: str
    expiry: str
    max_pain_strike: Decimal
    total_oi: Decimal
    source: str = "deribit"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix ms → UTC datetime."""
        return _parse_date(v)


class OptionsBatch(BaseModel):
    """Options 데이터 배치 컨테이너.

    Attributes:
        source: 데이터 소스
        data_type: 데이터 유형
        records: 레코드 튜플
        fetched_at: 수집 시각 (UTC)
    """

    model_config = ConfigDict(frozen=True)

    source: str = "deribit"
    data_type: str
    records: tuple[
        DVolRecord | PutCallRatioRecord | HistoricalVolRecord | TermStructureRecord | MaxPainRecord,
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
