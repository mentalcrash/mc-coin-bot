"""RegimeService — EDA 컴포넌트용 Regime 공유 서비스.

기존 EnsembleRegimeDetector를 래핑하여 모든 EDA 컴포넌트가
regime 정보를 투명하게 활용할 수 있게 합니다.

Modes:
    - Backtest: precompute()로 전체 데이터 vectorized 사전 계산
    - Live: _on_bar()로 BAR 이벤트 구독 → 증분 업데이트

Rules Applied:
    - EDA Component Pattern: register() → EventBus 구독
    - Backward Compatible: regime_service=None이면 기존 동작 100% 유지
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.regime.config import EnsembleRegimeDetectorConfig, RegimeLabel
from src.regime.ensemble import EnsembleRegimeDetector

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.core.events import AnyEvent


class RegimeServiceConfig(BaseModel):
    """RegimeService 설정.

    Attributes:
        ensemble: 앙상블 레짐 감지기 설정
        direction_window: 추세 방향 EMA 윈도우 (bar 수)
        direction_threshold: 최소 |normalized momentum| 임계값
        target_timeframe: BAR 이벤트 필터용 타임프레임
    """

    model_config = ConfigDict(frozen=True)

    ensemble: EnsembleRegimeDetectorConfig = Field(default_factory=EnsembleRegimeDetectorConfig)
    direction_window: int = Field(
        default=10,
        ge=3,
        le=60,
        description="추세 방향 EMA 윈도우 (bar 수)",
    )
    direction_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="최소 |normalized momentum| 임계값",
    )
    target_timeframe: str = Field(
        default="1D",
        description="BAR 이벤트 필터용 타임프레임",
    )


@dataclass
class EnrichedRegimeState:
    """방향 정보가 추가된 레짐 상태.

    Attributes:
        label: 현재 레짐 라벨 (TRENDING/RANGING/VOLATILE)
        probabilities: 각 레짐 확률 (합 = 1.0)
        bars_held: 현재 레짐 유지 bar 수
        raw_indicators: 원시 지표 값
        trend_direction: 추세 방향 (+1=상승, -1=하락, 0=중립)
        trend_strength: 추세 강도 (0.0~1.0)
    """

    label: RegimeLabel
    probabilities: dict[str, float]
    bars_held: int
    raw_indicators: dict[str, float] = field(default_factory=dict)
    trend_direction: int = 0
    trend_strength: float = 0.0


# DataFrame에 추가되는 regime 컬럼 목록
REGIME_COLUMNS = (
    "regime_label",
    "p_trending",
    "p_ranging",
    "p_volatile",
    "trend_direction",
    "trend_strength",
)

# 방향 계산에 필요한 최소 close 수
_MIN_DIRECTION_CLOSES = 3
_MIN_DIRECTION_BUFFER = 2
_MIN_LOG_RETURNS = 2


class RegimeService:
    """EDA 컴포넌트용 Regime 공유 서비스.

    Backtest: precompute() → enrich_dataframe() (vectorized, timestamp join)
    Live: register() → _on_bar() → get_regime() / get_regime_columns() (incremental)

    Args:
        config: RegimeServiceConfig
    """

    def __init__(self, config: RegimeServiceConfig | None = None) -> None:
        self._config = config or RegimeServiceConfig()
        self._detector = EnsembleRegimeDetector(self._config.ensemble)

        # 심볼별 최신 enriched regime state (live용)
        self._states: dict[str, EnrichedRegimeState] = {}

        # 방향 계산용 close 버퍼 (live용)
        self._close_buffers: dict[str, deque[float]] = {}

        # backtest 사전 계산 캐시 {symbol: DataFrame with regime columns}
        self._precomputed: dict[str, pd.DataFrame] = {}

    @property
    def config(self) -> RegimeServiceConfig:
        """현재 설정."""
        return self._config

    # ── EDA Registration ──

    async def register(self, bus: EventBus) -> None:
        """EventBus에 BAR 이벤트 구독 등록."""
        from src.core.events import EventType

        self._bus = bus
        bus.subscribe(EventType.BAR, self._on_bar)
        logger.info("RegimeService registered (tf={})", self._config.target_timeframe)

    async def _on_bar(self, event: AnyEvent) -> None:
        """BAR 이벤트 핸들러 — 증분 regime 업데이트 (live용)."""
        from src.core.events import BarEvent

        assert isinstance(event, BarEvent)
        bar = event

        # TF 필터
        if bar.timeframe != self._config.target_timeframe:
            return

        symbol = bar.symbol
        close = bar.close

        # 앙상블 detector incremental 업데이트
        state = self._detector.update(symbol, close)
        if state is None:
            return  # warmup 중

        # 방향 계산
        direction, strength = self._update_direction(symbol, close)

        enriched = EnrichedRegimeState(
            label=state.label,
            probabilities=dict(state.probabilities),
            bars_held=state.bars_held,
            raw_indicators=dict(state.raw_indicators),
            trend_direction=direction,
            trend_strength=strength,
        )
        self._states[symbol] = enriched

    # ── Query API ──

    def get_regime(self, symbol: str) -> EnrichedRegimeState | None:
        """현재 regime 상태 조회.

        Args:
            symbol: 거래 심볼

        Returns:
            EnrichedRegimeState 또는 미등록/warmup 시 None
        """
        return self._states.get(symbol)

    def get_regime_columns(self, symbol: str) -> dict[str, object] | None:
        """현재 regime 상태를 DataFrame 컬럼-값 dict로 반환.

        StrategyEngine에서 Live fallback 시 사용.
        사전 계산이 없을 때 현재 state를 broadcast합니다.

        Returns:
            {"regime_label": "trending", "p_trending": 0.6, ...} 또는 None
        """
        state = self._states.get(symbol)
        if state is None:
            return None

        return {
            "regime_label": state.label.value,
            "p_trending": state.probabilities.get("trending", 0.0),
            "p_ranging": state.probabilities.get("ranging", 0.0),
            "p_volatile": state.probabilities.get("volatile", 0.0),
            "trend_direction": state.trend_direction,
            "trend_strength": state.trend_strength,
        }

    # ── Backtest: Vectorized Precomputation ──

    def precompute(self, symbol: str, closes: pd.Series) -> pd.DataFrame:
        """전체 데이터에서 regime을 vectorized 사전 계산.

        Args:
            symbol: 거래 심볼
            closes: 전체 종가 시리즈 (DatetimeIndex)

        Returns:
            DataFrame with regime columns (same index as closes)
        """
        # 앙상블 감지기로 regime 분류
        regime_df = self._detector.classify_series(closes)

        # 방향/강도 vectorized 계산
        direction_df = self._compute_direction_vectorized(closes)

        # 결합
        result = pd.DataFrame(
            {
                "regime_label": regime_df["regime_label"],
                "p_trending": regime_df["p_trending"],
                "p_ranging": regime_df["p_ranging"],
                "p_volatile": regime_df["p_volatile"],
                "trend_direction": direction_df["trend_direction"],
                "trend_strength": direction_df["trend_strength"],
            },
            index=closes.index,
        )

        self._precomputed[symbol] = result
        logger.info(
            "RegimeService precomputed {} bars for {} (warmup NaN: {})",
            len(result),
            symbol,
            int(result["regime_label"].isna().sum()),
        )
        return result

    def enrich_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """사전 계산된 regime 컬럼을 df에 join.

        Backtest 모드에서 StrategyEngine이 호출합니다.
        사전 계산이 없으면 df를 변경 없이 반환합니다.

        Args:
            df: OHLCV DataFrame (DatetimeIndex)
            symbol: 거래 심볼

        Returns:
            regime 컬럼이 추가된 DataFrame (또는 원본)
        """
        precomputed = self._precomputed.get(symbol)
        if precomputed is None:
            return df

        # DatetimeIndex join — df의 인덱스에 맞춰 reindex
        regime_subset = precomputed.reindex(df.index)

        # 기존 regime 컬럼이 있으면 덮어쓰지 않음
        new_cols = [c for c in REGIME_COLUMNS if c not in df.columns]
        if not new_cols:
            return df

        # 원본 df를 변경하지 않도록 copy
        result = df.copy()
        for col in new_cols:
            if col in regime_subset.columns:
                result[col] = regime_subset[col].to_numpy()

        return result

    # ── Live: Warmup ──

    def warmup(self, symbol: str, closes: list[float]) -> None:
        """과거 close 데이터로 detector를 초기화 (live warmup용).

        Args:
            symbol: 거래 심볼
            closes: 과거 종가 리스트 (시간 순)
        """
        for close in closes:
            self._detector.update(symbol, close)
            self._update_direction(symbol, close)

        # 마지막 상태를 _states에 저장
        base_state = self._detector.get_regime(symbol)
        if base_state is not None:
            buf = self._close_buffers.get(symbol)
            if buf and len(buf) >= _MIN_DIRECTION_BUFFER:
                direction, strength = self._compute_direction_from_buffer(buf)
            else:
                direction, strength = 0, 0.0

            self._states[symbol] = EnrichedRegimeState(
                label=base_state.label,
                probabilities=dict(base_state.probabilities),
                bars_held=base_state.bars_held,
                raw_indicators=dict(base_state.raw_indicators),
                trend_direction=direction,
                trend_strength=strength,
            )

        logger.info(
            "RegimeService warmup: {} bars for {} → {}",
            len(closes),
            symbol,
            self._states.get(symbol, "None (warmup insufficient)"),
        )

    # ── Direction Computation ──

    def _update_direction(self, symbol: str, close: float) -> tuple[int, float]:
        """Close 버퍼에 추가 + 방향/강도 계산.

        Returns:
            (trend_direction, trend_strength)
        """
        cfg = self._config
        max_buf = cfg.direction_window + 5

        if symbol not in self._close_buffers:
            self._close_buffers[symbol] = deque(maxlen=max_buf)

        buf = self._close_buffers[symbol]
        buf.append(close)

        if len(buf) < _MIN_DIRECTION_CLOSES:
            return 0, 0.0

        return self._compute_direction_from_buffer(buf)

    def _compute_direction_from_buffer(self, buf: deque[float]) -> tuple[int, float]:
        """버퍼에서 방향/강도 계산.

        log_returns → EWM momentum → normalized → direction/strength

        Returns:
            (trend_direction, trend_strength)
        """
        cfg = self._config
        prices = np.array(buf, dtype=np.float64)

        # log returns
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < _MIN_LOG_RETURNS:
            return 0, 0.0

        # EWM momentum (pandas로 계산)
        returns_series = pd.Series(log_returns)
        ema_momentum = float(
            returns_series.ewm(span=min(cfg.direction_window, len(returns_series)), adjust=False)
            .mean()
            .iloc[-1]
        )

        # rolling std for normalization (match vectorized: window=direction_window)
        window = min(cfg.direction_window, len(returns_series))
        min_periods = max(2, window // 2)
        rolling_std = returns_series.rolling(window=window, min_periods=min_periods).std()
        std = float(rolling_std.iloc[-1])
        if std <= 0 or math.isnan(std):
            return 0, 0.0

        normalized = ema_momentum / std

        # direction + strength
        if abs(normalized) <= cfg.direction_threshold:
            return 0, 0.0

        direction = 1 if normalized > 0 else -1
        strength = min(abs(normalized), 1.0)

        return direction, strength

    def _compute_direction_vectorized(self, closes: pd.Series) -> pd.DataFrame:
        """Vectorized 방향/강도 계산 (backtest용).

        Args:
            closes: 전체 종가 시리즈

        Returns:
            DataFrame with trend_direction, trend_strength columns
        """
        cfg = self._config

        log_returns = np.log(closes / closes.shift(1))

        # EWM momentum
        ema_momentum = log_returns.ewm(span=cfg.direction_window, adjust=False).mean()

        # rolling std for normalization
        rolling_std = log_returns.rolling(
            window=cfg.direction_window, min_periods=max(2, cfg.direction_window // 2)
        ).std()
        rolling_std = rolling_std.replace(0, np.nan)

        normalized: pd.Series = ema_momentum / rolling_std  # type: ignore[assignment]

        # direction
        trend_direction = pd.Series(0, index=closes.index, dtype=int)
        above = normalized > cfg.direction_threshold
        below = normalized < -cfg.direction_threshold
        trend_direction = trend_direction.where(~above, 1)
        trend_direction = trend_direction.where(~below, -1)

        # NaN 처리 — normalized가 NaN인 곳은 direction=0
        nan_mask = normalized.isna()
        trend_direction = trend_direction.where(~nan_mask, 0)

        # strength = min(|normalized|, 1.0), NaN → 0.0
        trend_strength = normalized.abs().clip(upper=1.0).fillna(0.0)
        # threshold 미만은 0
        trend_strength = trend_strength.where(normalized.abs() > cfg.direction_threshold, 0.0)

        return pd.DataFrame(
            {
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
            },
            index=closes.index,
        )
