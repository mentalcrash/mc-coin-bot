"""Pydantic schemas for Binance market data."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RawBinanceKline(BaseModel):
    """Binance Kline 원본 데이터 (API 응답 스키마 그대로 저장).

    바이낸스 API 응답 형식을 그대로 유지하여 정밀도 손실을 방지합니다.
    가격과 수량 데이터는 문자열로 저장됩니다.
    """

    open_time: int = Field(..., description="Kline open time (ms timestamp)")
    open: str = Field(..., description="Open price")
    high: str = Field(..., description="High price")
    low: str = Field(..., description="Low price")
    close: str = Field(..., description="Close price")
    volume: str = Field(..., description="Base asset volume")
    close_time: int = Field(..., description="Kline close time (ms timestamp)")
    quote_volume: str = Field(..., description="Quote asset volume")
    trades: int = Field(..., description="Number of trades")
    taker_buy_volume: str = Field(..., description="Taker buy base asset volume")
    taker_buy_quote_volume: str = Field(..., description="Taker buy quote asset volume")

    model_config = {"frozen": True}

    @classmethod
    def from_binance_response(cls, data: list[Any]) -> "RawBinanceKline":
        """바이낸스 API 응답 리스트를 RawBinanceKline으로 변환.

        Args:
            data: 바이낸스 klines API 응답의 단일 캔들 데이터
                  [open_time, open, high, low, close, volume, close_time,
                   quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]

        Returns:
            RawBinanceKline 인스턴스
        """
        return cls(
            open_time=data[0],
            open=data[1],
            high=data[2],
            low=data[3],
            close=data[4],
            volume=data[5],
            close_time=data[6],
            quote_volume=data[7],
            trades=data[8],
            taker_buy_volume=data[9],
            taker_buy_quote_volume=data[10],
        )

    def to_dict(self) -> dict[str, Any]:
        """Parquet 저장용 딕셔너리로 변환."""
        return {
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "taker_buy_volume": self.taker_buy_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
        }


class CandleRecord(BaseModel):
    """로드 시 사용하는 정규화된 캔들 레코드.

    분석 및 전략 개발에 사용하기 위해 타입 변환된 캔들 데이터입니다.
    """

    timestamp: datetime = Field(..., description="UTC datetime (open_time 변환)")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., description="Base asset volume")
    quote_volume: float = Field(..., description="Quote asset volume")
    trades: int = Field(..., description="Number of trades")

    model_config = {"frozen": True}

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: int | datetime) -> datetime:
        """밀리초 타임스탬프 또는 datetime을 datetime으로 변환."""
        if isinstance(v, int):
            return datetime.fromtimestamp(v / 1000, tz=UTC)
        return v

    @classmethod
    def from_raw_kline(cls, raw: RawBinanceKline) -> "CandleRecord":
        """RawBinanceKline을 CandleRecord로 변환.

        Args:
            raw: 원본 바이낸스 캔들 데이터

        Returns:
            정규화된 CandleRecord 인스턴스
        """
        return cls(
            timestamp=datetime.fromtimestamp(raw.open_time / 1000, tz=UTC),
            open=float(raw.open),
            high=float(raw.high),
            low=float(raw.low),
            close=float(raw.close),
            volume=float(raw.volume),
            quote_volume=float(raw.quote_volume),
            trades=raw.trades,
        )


class TickerInfo(BaseModel):
    """티커 정보 (상위 종목 선정용)."""

    symbol: str = Field(..., description="Trading pair symbol (e.g., BTCUSDT)")
    quote_volume: float = Field(..., description="24h quote asset volume")

    model_config = {"frozen": True}
