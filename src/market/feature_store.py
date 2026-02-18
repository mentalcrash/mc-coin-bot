"""FeatureStore - 공통 지표 캐시 서비스.

전략 모듈 전반에서 동일한 기술적 지표(ATR, RSI 등)를 중복 계산하는 문제를 해결합니다.

Backtest: precompute(symbol, df) -> 전체 데이터 vectorized 사전 계산
Live: register(bus) -> BAR event 구독, 증분 업데이트
Query: enrich_dataframe(df, symbol) -> DataFrame에 캐시 컬럼 추가

RegimeService / DerivativesProvider와 동일한 패턴을 따릅니다.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.market import indicators
from src.market.indicators import log_returns

# 최소 버퍼 크기 (returns 계산에 최소 2개 바 필요)
_MIN_BUFFER_BARS = 2

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.core.events import AnyEvent

# ---------------------------------------------------------------------------
# Spec & Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorSpec:
    """단일 지표 사양.

    Args:
        name: indicators.py 함수명 (예: "atr", "rsi")
        params: 함수 파라미터 (OHLCV 시리즈 제외)
        column_name: 캐시 컬럼명 (None이면 자동 생성)
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    column_name: str | None = None


DEFAULT_SPECS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec("atr", {"period": 14}),
    IndicatorSpec("rsi", {"period": 14}),
    IndicatorSpec("adx", {"period": 14}),
    IndicatorSpec("realized_volatility", {"window": 30, "annualization_factor": 365.0}),
    IndicatorSpec("drawdown", {}),
    IndicatorSpec("parkinson_volatility", {}),
    IndicatorSpec("efficiency_ratio", {"period": 10}),
    IndicatorSpec("momentum", {"period": 10}),
)


class FeatureStoreConfig(BaseModel):
    """FeatureStore 설정."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    specs: tuple[IndicatorSpec, ...] = Field(default=DEFAULT_SPECS)
    target_timeframe: str = "1D"
    max_buffer_size: int = 500


# ---------------------------------------------------------------------------
# OHLCV parameter auto-mapping helpers
# ---------------------------------------------------------------------------

_OHLCV_PARAMS = frozenset({"close", "high", "low", "volume"})


def _resolve_column_name(spec: IndicatorSpec) -> str:
    """IndicatorSpec -> 캐시 컬럼명.

    column_name이 설정되면 그대로 사용.
    아니면 "{name}_{first_param_value}" 또는 "{name}" 자동 생성.
    """
    if spec.column_name is not None:
        return spec.column_name
    if spec.params:
        first_val = next(iter(spec.params.values()))
        return f"{spec.name}_{first_val}"
    return spec.name


def _call_indicator(spec: IndicatorSpec, df: pd.DataFrame) -> pd.Series:
    """IndicatorSpec + OHLCV DataFrame -> 계산된 지표 시리즈.

    함수 시그니처를 inspect하여 OHLCV 컬럼을 자동 매핑합니다.
    """
    fn = getattr(indicators, spec.name)
    sig = inspect.signature(fn)

    kwargs: dict[str, Any] = {}
    for param_name in sig.parameters:
        if param_name in spec.params:
            kwargs[param_name] = spec.params[param_name]
        elif param_name == "returns":
            close_series: pd.Series = df["close"]  # type: ignore[assignment]
            kwargs["returns"] = log_returns(close_series)
        elif param_name in _OHLCV_PARAMS:
            kwargs[param_name] = df[param_name]
        # else: function uses its default value

    result = fn(**kwargs)

    # Tuple returns (bollinger_bands 등): 첫 번째 시리즈만 사용
    if isinstance(result, tuple):
        return result[0]
    return result  # type: ignore[no-any-return]


# Public alias for _call_indicator (used by CLI ic-check)
compute_indicator = _call_indicator


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------


class FeatureStore:
    """공통 지표 캐시 서비스.

    RegimeService 패턴을 따릅니다:
    - Backtest: precompute(symbol, df) -> 전체 데이터 vectorized 사전 계산
    - Live: register(bus) -> BAR event 구독, 증분 업데이트
    - Query: enrich_dataframe(df, symbol) -> DataFrame에 캐시 컬럼 추가

    Args:
        config: FeatureStore 설정 (None이면 DEFAULT_SPECS 사용)
    """

    def __init__(self, config: FeatureStoreConfig | None = None) -> None:
        self._config = config or FeatureStoreConfig()
        # Backtest precomputed cache {symbol: DataFrame with indicator columns}
        self._cache: dict[str, pd.DataFrame] = {}
        # Live: 최신 지표값 {symbol: {col_name: value}}
        self._latest: dict[str, dict[str, float]] = {}
        # Live: OHLCV 버퍼 {symbol: list of bar dicts}
        self._buffers: dict[str, list[dict[str, float]]] = {}
        self._timestamps: dict[str, list[Any]] = {}

    # -------------------------------------------------------------------
    # Dynamic Spec Registration
    # -------------------------------------------------------------------

    def register_specs(self, extra: Sequence[IndicatorSpec]) -> None:
        """외부에서 추가 지표 Spec 등록. 중복(name 기준)은 무시.

        전략별로 필요한 추가 지표를 등록할 때 사용합니다.
        ``precompute()`` 호출 전에 등록이 완료되어야 합니다.

        Args:
            extra: 추가할 IndicatorSpec 시퀀스
        """
        existing = {s.name for s in self._config.specs}
        added: list[str] = []
        new_specs = list(self._config.specs)
        for spec in extra:
            if spec.name not in existing:
                new_specs.append(spec)
                existing.add(spec.name)
                added.append(spec.name)

        if added:
            # frozen config이므로 새 인스턴스로 교체
            self._config = FeatureStoreConfig(
                specs=tuple(new_specs),
                target_timeframe=self._config.target_timeframe,
                max_buffer_size=self._config.max_buffer_size,
            )
            logger.info("FeatureStore: registered {} specs: {}", len(added), added)

    # -------------------------------------------------------------------
    # Backtest
    # -------------------------------------------------------------------

    def precompute(self, symbol: str, df: pd.DataFrame) -> None:
        """전체 OHLCV 데이터에서 지표 사전 계산 + 캐시.

        Args:
            symbol: 거래 심볼
            df: 전체 OHLCV DataFrame (DatetimeIndex)
        """
        result = pd.DataFrame(index=df.index)
        for spec in self._config.specs:
            col_name = _resolve_column_name(spec)
            result[col_name] = _call_indicator(spec, df)

        self._cache[symbol] = result
        logger.info(
            "FeatureStore precomputed {} indicators for {} ({} bars)",
            len(self._config.specs),
            symbol,
            len(df),
        )

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """캐시된 지표 컬럼을 DataFrame에 timestamp join.

        Args:
            df: OHLCV DataFrame (DatetimeIndex)
            symbol: 거래 심볼

        Returns:
            지표 컬럼이 추가된 DataFrame (캐시 없으면 원본 반환)
        """
        cached = self._cache.get(symbol)
        if cached is None:
            return df

        subset = cached.reindex(df.index)
        new_cols = [c for c in subset.columns if c not in df.columns]
        if not new_cols:
            return df

        result = df.copy()
        for col in new_cols:
            result[col] = subset[col].to_numpy()
        return result

    # -------------------------------------------------------------------
    # Live
    # -------------------------------------------------------------------

    async def register(self, bus: EventBus) -> None:
        """EventBus에 BAR event 구독 등록.

        Args:
            bus: 이벤트 버스
        """
        from src.core.events import EventType

        bus.subscribe(EventType.BAR, self._on_bar)

    async def _on_bar(self, event: AnyEvent) -> None:
        """증분 지표 업데이트 (TF 필터 적용).

        target_timeframe 바만 처리. 버퍼에 누적 후 전체 재계산하여
        최신 지표값을 _latest에 저장합니다.
        """
        from src.core.events import BarEvent

        assert isinstance(event, BarEvent)
        if event.timeframe != self._config.target_timeframe:
            return

        symbol = event.symbol

        # 버퍼에 누적
        if symbol not in self._buffers:
            self._buffers[symbol] = []
            self._timestamps[symbol] = []

        self._buffers[symbol].append(
            {
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
            }
        )
        self._timestamps[symbol].append(event.bar_timestamp)

        # 버퍼 크기 제한
        max_buf = self._config.max_buffer_size
        if len(self._buffers[symbol]) > max_buf:
            trim = len(self._buffers[symbol]) - max_buf
            self._buffers[symbol] = self._buffers[symbol][trim:]
            self._timestamps[symbol] = self._timestamps[symbol][trim:]

        # 최소 바 수 체크 (returns 계산용)
        if len(self._buffers[symbol]) < _MIN_BUFFER_BARS:
            return

        # 버퍼로 DataFrame 구성 후 지표 재계산
        df = pd.DataFrame(
            self._buffers[symbol],
            index=pd.DatetimeIndex(self._timestamps[symbol]),
        )
        latest_values: dict[str, float] = {}
        for spec in self._config.specs:
            col_name = _resolve_column_name(spec)
            try:
                series = _call_indicator(spec, df)
                val = series.iloc[-1]
                if pd.notna(val):
                    latest_values[col_name] = float(val)
            except (ValueError, KeyError):
                pass  # 데이터 부족 시 스킵

        if latest_values:
            self._latest[symbol] = latest_values

    def get_feature_columns(self, symbol: str) -> dict[str, float] | None:
        """최신 캐시된 지표 값 반환 (live fallback).

        Args:
            symbol: 거래 심볼

        Returns:
            {column_name: value} 또는 None (데이터 없음)
        """
        return self._latest.get(symbol)

    def warmup(self, symbol: str, df: pd.DataFrame) -> None:
        """Live 시작 전 히스토리 데이터로 캐시 초기화.

        precompute로 캐시를 채우고, 이후 _on_bar에서 사용할
        버퍼도 초기화합니다.

        Args:
            symbol: 거래 심볼
            df: 히스토리 OHLCV DataFrame
        """
        self.precompute(symbol, df)

        # 버퍼 초기화 (이후 _on_bar에서 이어서 사용)
        max_buf = self._config.max_buffer_size
        tail = df.tail(max_buf)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        records: list[dict[str, float]] = tail[ohlcv_cols].to_dict("records")  # type: ignore[assignment]
        self._buffers[symbol] = [{k: float(v) for k, v in r.items()} for r in records]
        self._timestamps[symbol] = list(tail.index)

        # 최신 값 설정
        cached = self._cache.get(symbol)
        if cached is not None and len(cached) > 0:
            last_row = cached.iloc[-1]
            self._latest[symbol] = {
                col: float(val) for col, val in last_row.items() if pd.notna(val)
            }

    # -------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        """설정된 지표의 캐시 컬럼명 리스트."""
        return [_resolve_column_name(spec) for spec in self._config.specs]

    @property
    def config(self) -> FeatureStoreConfig:
        """설정."""
        return self._config
