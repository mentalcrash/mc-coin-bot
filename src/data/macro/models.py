"""Macro economic data models (FRED + yfinance).

Pydantic V2 frozen models with Decimal for financial values.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for financial values
    - #10 Python Standards: Modern typing (X | None)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FREDObservationRecord(BaseModel):
    """FRED API observation record.

    Attributes:
        date: 관측일 (UTC)
        series_id: FRED 시리즈 ID (e.g., "DTWEXBGS")
        value: 관측값 (None = FRED의 "." 값)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    series_id: str
    value: Decimal | None = None
    source: str = "fred"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix seconds → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: str | float | int | Decimal | None) -> Decimal | None:
        """FRED "." → None, otherwise Decimal."""
        if v is None or v == ".":
            return None
        return Decimal(str(v))


class YFinanceRecord(BaseModel):
    """yfinance ETF price record.

    Attributes:
        date: 날짜 (UTC)
        ticker: 티커 심볼 (e.g., "SPY")
        open: 시가
        high: 고가
        low: 저가
        close: 종가
        volume: 거래량
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    ticker: str
    open: Decimal = Field(..., ge=0)
    high: Decimal = Field(..., ge=0)
    low: Decimal = Field(..., ge=0)
    close: Decimal = Field(..., ge=0)
    volume: Decimal = Field(..., ge=0)
    source: str = "yfinance"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix seconds → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class CoinGeckoGlobalRecord(BaseModel):
    """CoinGecko /global 스냅샷.

    Attributes:
        date: 수집 시점 (UTC)
        btc_dominance: BTC 시장 점유율 (%)
        eth_dominance: ETH 시장 점유율 (%)
        total_market_cap_usd: 전체 시가총액 (USD)
        total_volume_usd: 전체 24h 거래량 (USD)
        active_cryptocurrencies: 활성 암호화폐 수
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    btc_dominance: Decimal
    eth_dominance: Decimal
    total_market_cap_usd: Decimal
    total_volume_usd: Decimal
    active_cryptocurrencies: int
    source: str = "coingecko"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix seconds → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class CoinGeckoDefiRecord(BaseModel):
    """CoinGecko /global/decentralized_finance_defi 스냅샷.

    Attributes:
        date: 수집 시점 (UTC)
        defi_market_cap: DeFi 시가총액
        defi_to_eth_ratio: DeFi/ETH 비율
        defi_dominance: DeFi 시장 점유율 (%)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    defi_market_cap: Decimal
    defi_to_eth_ratio: Decimal
    defi_dominance: Decimal
    source: str = "coingecko"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """ISO string / Unix seconds → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class MacroBatch(BaseModel):
    """Macro 데이터 배치 (수집 결과 컨테이너).

    Attributes:
        source: 데이터 소스 (fred, yfinance)
        data_type: 데이터 유형 (e.g., "dxy", "spy")
        records: 레코드 튜플
        fetched_at: 수집 시각 (UTC)
    """

    model_config = ConfigDict(frozen=True)

    source: str
    data_type: str
    records: tuple[
        FREDObservationRecord | YFinanceRecord | CoinGeckoGlobalRecord | CoinGeckoDefiRecord,
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
