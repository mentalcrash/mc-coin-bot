"""Market Data DTOs for data layer abstraction.

이 모듈은 데이터 접근 추상화를 위한 DTO(Data Transfer Objects)를 정의합니다.
MarketDataRequest로 요청하고, MarketDataSet으로 응답받는 일관된 인터페이스를 제공합니다.

Design Principles:
    - Immutable DTOs: frozen=True로 불변성 보장
    - Metadata + Data: 메타데이터와 실제 데이터를 함께 래핑
    - VectorBT Compatible: freq 프로퍼티로 VectorBT 주기 문자열 제공

Rules Applied:
    - #10 Python Standards: Modern typing, dataclass
    - #12 Data Engineering: UTC, DatetimeIndex
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class MarketDataRequest:
    """시장 데이터 요청 DTO.

    데이터 서비스에 데이터를 요청할 때 사용하는 불변 객체입니다.
    심볼, 타임프레임, 기간 정보를 포함합니다.

    Attributes:
        symbol: 거래 심볼 (예: "BTC/USDT")
        timeframe: 타임프레임 (예: "1m", "1h", "1D")
        start: 시작 시각 (UTC)
        end: 종료 시각 (UTC)

    Example:
        >>> request = MarketDataRequest(
        ...     symbol="BTC/USDT",
        ...     timeframe="1D",
        ...     start=datetime(2024, 1, 1, tzinfo=UTC),
        ...     end=datetime(2025, 12, 31, tzinfo=UTC),
        ... )
    """

    symbol: str
    timeframe: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """검증: 종료일이 시작일보다 커야 함."""
        if self.end <= self.start:
            msg = f"end ({self.end}) must be after start ({self.start})"
            raise ValueError(msg)


@dataclass
class MarketDataSet:
    """시장 데이터 응답 DTO.

    메타데이터와 OHLCV 데이터를 함께 래핑한 객체입니다.
    백테스트, EDA, 분석 등 다양한 컨텍스트에서 재사용됩니다.

    Attributes:
        symbol: 거래 심볼
        timeframe: 타임프레임
        start: 실제 데이터 시작 시각
        end: 실제 데이터 종료 시각
        ohlcv: OHLCV DataFrame (DatetimeIndex + OHLCV columns)

    Example:
        >>> data = MarketDataSet(
        ...     symbol="BTC/USDT",
        ...     timeframe="1D",
        ...     start=datetime(2024, 1, 1, tzinfo=UTC),
        ...     end=datetime(2025, 12, 31, tzinfo=UTC),
        ...     ohlcv=df,
        ... )
        >>> print(f"Total periods: {data.periods}")
        >>> print(f"VBT freq: {data.freq}")
    """

    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    ohlcv: pd.DataFrame

    @property
    def periods(self) -> int:
        """데이터 행 수 (캔들 수)."""
        return len(self.ohlcv)

    @property
    def freq(self) -> str:
        """VectorBT용 주기 문자열.

        VectorBT Portfolio 생성 시 freq 파라미터로 사용됩니다.
        연환산 수익률 계산에 필요합니다.

        Returns:
            VectorBT 호환 주기 문자열 (예: "1h", "1D")
        """
        # 타임프레임을 VectorBT 포맷으로 변환
        timeframe_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1D": "1D",
            "1d": "1D",
            "1W": "1W",
            "1w": "1W",
        }
        return timeframe_map.get(self.timeframe, self.timeframe)

    @property
    def duration_days(self) -> int:
        """데이터 기간 (일)."""
        return (self.end - self.start).days

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"MarketDataSet("
            f"symbol={self.symbol!r}, "
            f"timeframe={self.timeframe!r}, "
            f"periods={self.periods}, "
            f"start={self.start.date()}, "
            f"end={self.end.date()})"
        )
