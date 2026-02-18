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
import numpy.typing as npt
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
    """방향 + 신뢰도 + 전환확률 + cascade 정보가 추가된 레짐 상태.

    Attributes:
        label: 현재 레짐 라벨 (TRENDING/RANGING/VOLATILE)
        probabilities: 각 레짐 확률 (합 = 1.0)
        bars_held: 현재 레짐 유지 bar 수
        raw_indicators: 원시 지표 값
        trend_direction: 추세 방향 (+1=상승, -1=하락, 0=중립)
        trend_strength: 추세 강도 (0.0~1.0)
        confidence: detector agreement (0~1)
        transition_prob: 다음 bar 레짐 전환 확률 (0~1)
        cascade_risk: derivatives 기반 급락 위험도 (0~1)
    """

    label: RegimeLabel
    probabilities: dict[str, float]
    bars_held: int
    raw_indicators: dict[str, float] = field(default_factory=dict)
    trend_direction: int = 0
    trend_strength: float = 0.0
    confidence: float = 0.0
    transition_prob: float = 0.0
    cascade_risk: float = 0.0


@dataclass(frozen=True)
class RegimeContext:
    """전략 소비용 regime 정보 패키지.

    Attributes:
        label: 현재 레짐 라벨
        p_trending: trending 확률
        p_ranging: ranging 확률
        p_volatile: volatile 확률
        confidence: detector agreement (0~1)
        transition_prob: 다음 bar 전환 확률 (0~1)
        cascade_risk: derivatives 기반 급락 위험도 (0~1)
        trend_direction: 추세 방향 (+1/-1/0)
        trend_strength: 추세 강도 (0~1)
        bars_in_regime: 현재 레짐 유지 bar 수
        suggested_vol_scalar: regime 기반 vol 스케일러 (0.1~1.0)
    """

    label: RegimeLabel
    p_trending: float
    p_ranging: float
    p_volatile: float
    confidence: float
    transition_prob: float
    cascade_risk: float
    trend_direction: int
    trend_strength: float
    bars_in_regime: int
    suggested_vol_scalar: float


# DataFrame에 추가되는 regime 컬럼 목록
REGIME_COLUMNS = (
    "regime_label",
    "p_trending",
    "p_ranging",
    "p_volatile",
    "trend_direction",
    "trend_strength",
    "regime_confidence",
    "regime_transition_prob",
    "cascade_risk",
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
        derivatives_provider: derivatives 데이터 제공자 (None이면 비활성)
    """

    def __init__(
        self,
        config: RegimeServiceConfig | None = None,
        derivatives_provider: object | None = None,
    ) -> None:
        self._config = config or RegimeServiceConfig()
        self._detector = EnsembleRegimeDetector(self._config.ensemble)
        self._derivatives_provider = derivatives_provider

        # 심볼별 최신 enriched regime state (live용)
        self._states: dict[str, EnrichedRegimeState] = {}

        # 방향 계산용 close 버퍼 (live용)
        self._close_buffers: dict[str, deque[float]] = {}

        # backtest 사전 계산 캐시 {symbol: DataFrame with regime columns}
        self._precomputed: dict[str, pd.DataFrame] = {}

        # Transition matrix per symbol: 3x3 (trending/ranging/volatile)
        self._transition_matrices: dict[str, npt.NDArray[np.float64]] = {}
        self._transition_counts: dict[str, npt.NDArray[np.float64]] = {}

        # EventBus 참조 (register 시 설정)
        self._bus: EventBus | None = None

    @property
    def config(self) -> RegimeServiceConfig:
        """현재 설정."""
        return self._config

    # ── Transition Matrix ──

    _IDX_MAP: dict[str, int] = {"trending": 0, "ranging": 1, "volatile": 2}

    def _estimate_transition_matrix(self, labels: pd.Series) -> npt.NDArray[np.float64]:
        """Label 시퀀스에서 3x3 전환 행렬 추정 (Laplace smoothing).

        Args:
            labels: regime label 시퀀스

        Returns:
            3x3 row-normalized transition matrix
        """
        matrix = np.ones((3, 3))  # Laplace smoothing
        prev: str | None = None
        for label in labels.dropna():
            lbl = str(label)
            if prev is not None and lbl in self._IDX_MAP and prev in self._IDX_MAP:
                matrix[self._IDX_MAP[prev], self._IDX_MAP[lbl]] += 1
            prev = lbl
        row_sums = matrix.sum(axis=1, keepdims=True)
        return matrix / row_sums

    def _get_transition_prob(self, symbol: str, current_label: RegimeLabel) -> float:
        """현재 label에서 다른 label로 전환될 확률.

        Args:
            symbol: 거래 심볼
            current_label: 현재 레짐 라벨

        Returns:
            전환 확률 (0~1). matrix 없으면 0.0
        """
        mat = self._transition_matrices.get(symbol)
        if mat is None:
            return 0.0
        idx = self._IDX_MAP.get(current_label.value, 0)
        return 1.0 - float(mat[idx, idx])

    def _update_transition_counts(self, symbol: str, prev_label: str, new_label: str) -> None:
        """Incremental transition count 업데이트 + matrix 재계산.

        Args:
            symbol: 거래 심볼
            prev_label: 이전 레짐 라벨
            new_label: 현재 레짐 라벨
        """
        if symbol not in self._transition_counts:
            self._transition_counts[symbol] = np.ones((3, 3))  # Laplace

        if prev_label in self._IDX_MAP and new_label in self._IDX_MAP:
            self._transition_counts[symbol][
                self._IDX_MAP[prev_label], self._IDX_MAP[new_label]
            ] += 1

        counts = self._transition_counts[symbol]
        row_sums = counts.sum(axis=1, keepdims=True)
        self._transition_matrices[symbol] = counts / row_sums

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

        # Derivatives 데이터 조회 (provider가 있으면)
        derivatives: dict[str, float] | None = None
        cascade_risk = 0.0
        if self._derivatives_provider is not None:
            derivatives = self._get_derivatives_for_bar(symbol)

        # 앙상블 detector incremental 업데이트
        state = self._detector.update(symbol, close, derivatives=derivatives)
        if state is None:
            return  # warmup 중

        # Cascade risk 조회 (derivatives detector가 있을 때)
        if self._detector.has_derivatives_detector:
            cascade_risk = self._detector.get_cascade_risk(symbol)

        # Transition probability
        prev_state = self._states.get(symbol)
        if prev_state is not None:
            self._update_transition_counts(symbol, prev_state.label.value, state.label.value)
        transition_prob = self._get_transition_prob(symbol, state.label)

        # 방향 계산
        direction, strength = self._update_direction(symbol, close)

        enriched = EnrichedRegimeState(
            label=state.label,
            probabilities=dict(state.probabilities),
            bars_held=state.bars_held,
            raw_indicators=dict(state.raw_indicators),
            trend_direction=direction,
            trend_strength=strength,
            confidence=state.confidence,
            transition_prob=transition_prob,
            cascade_risk=cascade_risk,
        )

        # REGIME_CHANGE 이벤트 발행
        if prev_state is not None and enriched.label != prev_state.label:
            await self._publish_regime_change(symbol, prev_state, enriched)

        self._states[symbol] = enriched

    def _get_derivatives_for_bar(self, symbol: str) -> dict[str, float] | None:
        """derivatives_provider에서 현재 bar의 derivatives 데이터 조회.

        Returns:
            derivatives dict 또는 None
        """
        provider = self._derivatives_provider
        if provider is None:
            return None

        # BacktestDerivativesProvider.get_derivatives_columns(symbol) 호출
        get_fn = getattr(provider, "get_derivatives_columns", None)
        if get_fn is not None and callable(get_fn):
            result = get_fn(symbol)
            if isinstance(result, dict):
                return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        return None

    async def _publish_regime_change(
        self,
        symbol: str,
        prev_state: EnrichedRegimeState,
        new_state: EnrichedRegimeState,
    ) -> None:
        """REGIME_CHANGE 이벤트 발행.

        Args:
            symbol: 거래 심볼
            prev_state: 이전 레짐 상태
            new_state: 새 레짐 상태
        """
        if self._bus is None:
            return

        from src.core.events import RegimeChangeEvent

        event = RegimeChangeEvent(
            symbol=symbol,
            prev_label=prev_state.label.value,
            new_label=new_state.label.value,
            confidence=new_state.confidence,
            cascade_risk=new_state.cascade_risk,
            transition_prob=new_state.transition_prob,
        )
        await self._bus.publish(event)

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
            "regime_confidence": state.confidence,
            "regime_transition_prob": state.transition_prob,
            "cascade_risk": state.cascade_risk,
        }

    def get_regime_context(self, symbol: str) -> RegimeContext | None:
        """전략 소비용 rich regime 정보 패키지 반환.

        Args:
            symbol: 거래 심볼

        Returns:
            RegimeContext 또는 미등록/warmup 시 None
        """
        state = self._states.get(symbol)
        if state is None:
            return None

        # suggested_vol_scalar 계산
        base = (
            state.probabilities.get("trending", 0.0) * 1.0
            + state.probabilities.get("ranging", 0.0) * 0.4
            + state.probabilities.get("volatile", 0.0) * 0.2
        )

        deriv_cfg = self._config.ensemble.derivatives
        cascade_threshold = deriv_cfg.cascade_risk_threshold if deriv_cfg is not None else 0.7
        if state.cascade_risk > cascade_threshold:
            vol_scalar = 0.1
        else:
            vol_scalar = round(base * (0.5 + 0.5 * state.confidence), 2)

        return RegimeContext(
            label=state.label,
            p_trending=state.probabilities.get("trending", 0.0),
            p_ranging=state.probabilities.get("ranging", 0.0),
            p_volatile=state.probabilities.get("volatile", 0.0),
            confidence=state.confidence,
            transition_prob=state.transition_prob,
            cascade_risk=state.cascade_risk,
            trend_direction=state.trend_direction,
            trend_strength=state.trend_strength,
            bars_in_regime=state.bars_held,
            suggested_vol_scalar=vol_scalar,
        )

    # ── Backtest: Vectorized Precomputation ──

    def precompute(
        self,
        symbol: str,
        closes: pd.Series,
        deriv_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """전체 데이터에서 regime을 vectorized 사전 계산.

        Args:
            symbol: 거래 심볼
            closes: 전체 종가 시리즈 (DatetimeIndex)
            deriv_df: derivatives DataFrame (DerivativesDetector용)

        Returns:
            DataFrame with regime columns (same index as closes)
        """
        # 앙상블 감지기로 regime 분류 (derivatives 포함)
        regime_df = self._detector.classify_series(closes, deriv_df=deriv_df)

        # 방향/강도 vectorized 계산
        direction_df = self._compute_direction_vectorized(closes)

        # Transition matrix 추정
        # NOTE: Backtest에서는 전체 label 시퀀스로 matrix를 추정합니다 (look-ahead).
        # Live에서는 expanding window로 _update_transition_counts()가 누적합니다.
        # 정보용(transition_prob)이므로 전략 시그널에 영향을 주지 않습니다.
        regime_labels: pd.Series = regime_df["regime_label"]  # type: ignore[assignment]
        self._transition_matrices[symbol] = self._estimate_transition_matrix(regime_labels)
        logger.debug(
            "Transition matrix estimated from full label sequence ({} bars, look-ahead)",
            len(regime_labels.dropna()),
        )

        # Transition probability per bar
        transition_prob = self._compute_transition_prob_vectorized(symbol, regime_labels)

        # Confidence (ensemble에서 이미 계산됨)
        confidence = regime_df.get("confidence", pd.Series(0.0, index=closes.index))

        # Cascade risk (derivatives detector가 있으면)
        cascade_risk_col: pd.Series
        if self._detector.has_derivatives_detector and deriv_df is not None:
            cascade_risk_col = self._detector.get_cascade_risk_series(deriv_df)
        else:
            cascade_risk_col = pd.Series(0.0, index=closes.index)
            if self._detector.has_derivatives_detector and deriv_df is None:
                logger.warning(
                    "DerivativesDetector active but deriv_df=None — cascade_risk will be 0.0"
                )

        # 결합
        result = pd.DataFrame(
            {
                "regime_label": regime_df["regime_label"],
                "p_trending": regime_df["p_trending"],
                "p_ranging": regime_df["p_ranging"],
                "p_volatile": regime_df["p_volatile"],
                "trend_direction": direction_df["trend_direction"],
                "trend_strength": direction_df["trend_strength"],
                "regime_confidence": confidence,
                "regime_transition_prob": transition_prob,
                "cascade_risk": cascade_risk_col,
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

    def _compute_transition_prob_expanding(self, labels: pd.Series) -> pd.Series:
        """Expanding-window transition probability 계산.

        각 bar까지의 label 시퀀스로 transition matrix를 누적 추정하여
        look-ahead bias를 제거합니다.

        Args:
            labels: regime label 시리즈

        Returns:
            transition_prob 시리즈 (0~1)
        """
        result = pd.Series(0.0, index=labels.index, dtype=float)
        counts = np.ones((3, 3))  # Laplace smoothing
        prev_label: str | None = None

        for i, label in enumerate(labels):
            if pd.isna(label):
                prev_label = None
                continue

            lbl = str(label)
            if prev_label is not None and lbl in self._IDX_MAP and prev_label in self._IDX_MAP:
                counts[self._IDX_MAP[prev_label], self._IDX_MAP[lbl]] += 1

            # Current matrix
            row_sums = counts.sum(axis=1, keepdims=True)
            mat = counts / row_sums

            if lbl in self._IDX_MAP:
                idx = self._IDX_MAP[lbl]
                result.iloc[i] = 1.0 - float(mat[idx, idx])

            prev_label = lbl

        return result

    def _compute_transition_prob_vectorized(self, symbol: str, labels: pd.Series) -> pd.Series:
        """Expanding-window transition probability 계산 (look-ahead 제거).

        Args:
            symbol: 거래 심볼
            labels: regime label 시리즈

        Returns:
            transition_prob 시리즈
        """
        return self._compute_transition_prob_expanding(labels)

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
        # Build label sequence for transition matrix
        label_sequence: list[str] = []

        for close in closes:
            state = self._detector.update(symbol, close)
            self._update_direction(symbol, close)
            if state is not None:
                label_sequence.append(state.label.value)

        # Build transition matrix from warmup labels
        if len(label_sequence) >= _MIN_DIRECTION_BUFFER:
            labels_series = pd.Series(label_sequence)
            self._transition_matrices[symbol] = self._estimate_transition_matrix(labels_series)

        # 마지막 상태를 _states에 저장
        base_state = self._detector.get_regime(symbol)
        if base_state is not None:
            buf = self._close_buffers.get(symbol)
            if buf and len(buf) >= _MIN_DIRECTION_BUFFER:
                direction, strength = self._compute_direction_from_buffer(buf)
            else:
                direction, strength = 0, 0.0

            transition_prob = self._get_transition_prob(symbol, base_state.label)

            self._states[symbol] = EnrichedRegimeState(
                label=base_state.label,
                probabilities=dict(base_state.probabilities),
                bars_held=base_state.bars_held,
                raw_indicators=dict(base_state.raw_indicators),
                trend_direction=direction,
                trend_strength=strength,
                confidence=base_state.confidence,
                transition_prob=transition_prob,
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
