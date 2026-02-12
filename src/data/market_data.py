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

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

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
        return _TIMEFRAME_MAP.get(self.timeframe, self.timeframe)

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


# =============================================================================
# VectorBT 타임프레임 변환 맵 (공유)
# =============================================================================

_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "1D": "1D",
    "1d": "1D",
    "1W": "1W",
    "1w": "1W",
}


@dataclass
class MultiSymbolData:
    """멀티 심볼 시장 데이터 컨테이너.

    여러 심볼의 OHLCV 데이터를 함께 래핑합니다.
    VectorBT `cash_sharing` 멀티에셋 백테스트의 입력으로 사용됩니다.

    Attributes:
        symbols: 심볼 목록 (예: ["BTC/USDT", "ETH/USDT", ...])
        timeframe: 타임프레임 (모든 심볼 동일)
        start: 데이터 시작 시각
        end: 데이터 종료 시각
        ohlcv: 심볼별 OHLCV DataFrame 딕셔너리

    Example:
        >>> data = MultiSymbolData(
        ...     symbols=["BTC/USDT", "ETH/USDT"],
        ...     timeframe="1D",
        ...     start=datetime(2020, 1, 1, tzinfo=UTC),
        ...     end=datetime(2025, 12, 31, tzinfo=UTC),
        ...     ohlcv={"BTC/USDT": btc_df, "ETH/USDT": eth_df},
        ... )
        >>> close_matrix = data.close_matrix  # DataFrame with symbol columns
    """

    symbols: list[str]
    timeframe: str
    start: datetime
    end: datetime
    ohlcv: dict[str, pd.DataFrame]

    def __post_init__(self) -> None:
        """검증."""
        if not self.symbols:
            msg = "symbols must not be empty"
            raise ValueError(msg)
        if set(self.symbols) != set(self.ohlcv.keys()):
            msg = f"symbols {self.symbols} do not match ohlcv keys {list(self.ohlcv.keys())}"
            raise ValueError(msg)

    @property
    def n_assets(self) -> int:
        """자산 수."""
        return len(self.symbols)

    @property
    def close_matrix(self) -> pd.DataFrame:
        """심볼별 close를 DataFrame으로 합성 (VectorBT 멀티에셋 입력용).

        Returns:
            DataFrame with DatetimeIndex, columns=symbols
        """
        return pd.DataFrame({s: self.ohlcv[s]["close"] for s in self.symbols})

    @property
    def freq(self) -> str:
        """VectorBT용 주기 문자열."""
        return _TIMEFRAME_MAP.get(self.timeframe, self.timeframe)

    @property
    def periods(self) -> int:
        """첫 번째 심볼 기준 데이터 행 수."""
        first = self.ohlcv[self.symbols[0]]
        return len(first)

    def get_single(self, symbol: str) -> MarketDataSet:
        """단일 심볼 MarketDataSet 추출 (기존 인터페이스 호환).

        Args:
            symbol: 추출할 심볼

        Returns:
            해당 심볼의 MarketDataSet

        Raises:
            KeyError: 존재하지 않는 심볼
        """
        if symbol not in self.ohlcv:
            msg = f"Symbol {symbol!r} not found. Available: {self.symbols}"
            raise KeyError(msg)
        return MarketDataSet(
            symbol=symbol,
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
            ohlcv=self.ohlcv[symbol],
        )

    def slice_time(self, start: datetime, end: datetime) -> MultiSymbolData:
        """시간 범위로 슬라이싱 (검증용 데이터 분할).

        Args:
            start: 시작 시각
            end: 종료 시각

        Returns:
            슬라이싱된 새 MultiSymbolData
        """
        sliced_ohlcv: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            df = self.ohlcv[symbol]
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            sliced_ohlcv[symbol] = df.loc[mask].copy()
        return MultiSymbolData(
            symbols=list(self.symbols),
            timeframe=self.timeframe,
            start=start,
            end=end,
            ohlcv=sliced_ohlcv,
        )

    def slice_iloc(self, start_idx: int, end_idx: int) -> MultiSymbolData:
        """정수 인덱스로 슬라이싱 (검증 splitter용).

        Args:
            start_idx: 시작 인덱스 (inclusive)
            end_idx: 종료 인덱스 (exclusive)

        Returns:
            슬라이싱된 새 MultiSymbolData
        """
        sliced_ohlcv: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            sliced_ohlcv[symbol] = self.ohlcv[symbol].iloc[start_idx:end_idx].copy()

        ref_df = sliced_ohlcv[self.symbols[0]]
        ref_index = ref_df.index
        new_start = ref_index[0].to_pydatetime().replace(tzinfo=UTC)  # type: ignore[union-attr]
        new_end = ref_index[-1].to_pydatetime().replace(tzinfo=UTC)  # type: ignore[union-attr]

        return MultiSymbolData(
            symbols=list(self.symbols),
            timeframe=self.timeframe,
            start=new_start,
            end=new_end,
            ohlcv=sliced_ohlcv,
        )

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"MultiSymbolData("
            f"symbols={self.symbols}, "
            f"timeframe={self.timeframe!r}, "
            f"periods={self.periods}, "
            f"start={self.start.date()}, "
            f"end={self.end.date()})"
        )
