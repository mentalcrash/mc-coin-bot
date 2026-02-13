"""Regime Detector (공유 인프라).

RV Ratio와 Efficiency Ratio 기반 시장 레짐 분류기.
전략에서 import하여 레짐 컬럼을 DataFrame에 추가하는 유틸리티입니다.

Vectorized API: classify_series() — VBT 전략 preprocessor용
Incremental API: update() — EDA / 라이브용

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops in classify_series)
    - #10 Python Standards: Modern typing, dataclass
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.regime.config import RegimeDetectorConfig, RegimeLabel


@dataclass
class RegimeState:
    """개별 bar의 레짐 상태.

    Attributes:
        label: 현재 레짐 라벨
        probabilities: 각 레짐의 확률 (합 = 1.0)
        bars_held: 현재 레짐 유지 bar 수
        raw_indicators: 원시 지표 값
    """

    label: RegimeLabel
    probabilities: dict[str, float]
    bars_held: int
    raw_indicators: dict[str, float]


def _sigmoid(
    x: npt.NDArray[np.float64] | pd.Series,
    center: float,
    scale: float = 10.0,
) -> npt.NDArray[np.float64]:
    """Sigmoid 변환 (soft classification).

    Args:
        x: 입력 값
        center: 중심점 (x=center에서 0.5)
        scale: 기울기 (높을수록 급격한 전환)

    Returns:
        0~1 범위의 변환된 값
    """
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scale * (arr - center)))


def apply_hysteresis(
    raw_labels: pd.Series,
    min_hold_bars: int,
) -> pd.Series:
    """Hysteresis 적용: 최소 bar 수 이상 유지 후에만 레짐 전환.

    Args:
        raw_labels: 원시 레짐 라벨 시리즈
        min_hold_bars: 최소 유지 bar 수

    Returns:
        Hysteresis가 적용된 레짐 라벨 시리즈
    """
    if min_hold_bars <= 1:
        return raw_labels

    result = raw_labels.copy()
    values = result.to_numpy()
    n = len(values)

    # NaN이 아닌 첫 번째 인덱스 찾기
    start_idx = 0
    for i in range(n):
        if pd.notna(values[i]):
            start_idx = i
            break

    current_label = values[start_idx]
    hold_count = 1
    pending_label = None
    pending_count = 0

    for i in range(start_idx + 1, n):
        if pd.isna(values[i]):
            continue

        if values[i] == current_label:
            hold_count += 1
            pending_label = None
            pending_count = 0
        elif values[i] == pending_label:
            pending_count += 1
            if pending_count >= min_hold_bars:
                current_label = pending_label
                hold_count = pending_count
                pending_label = None
                pending_count = 0
            else:
                values[i] = current_label
        else:
            pending_label = values[i]
            pending_count = 1
            values[i] = current_label

    result = pd.Series(values, index=raw_labels.index)
    return result


class RegimeDetector:
    """시장 레짐 감지기.

    RV Ratio (단기/장기 변동성 비율)와 Efficiency Ratio를 사용하여
    시장을 trending, ranging, volatile 세 가지 레짐으로 분류합니다.

    Args:
        config: 레짐 감지 설정

    Example:
        >>> detector = RegimeDetector()
        >>> regime_df = detector.classify_series(close_prices)
        >>> regime_df["regime_label"]
    """

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self._config = config or RegimeDetectorConfig()
        # Incremental state: symbol → deque of close prices
        self._buffers: dict[str, deque[float]] = {}
        self._states: dict[str, RegimeState] = {}
        self._hold_counters: dict[str, int] = {}
        self._pending_labels: dict[str, RegimeLabel | None] = {}

    @property
    def config(self) -> RegimeDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (bar 수)."""
        return self._config.rv_long_window + 5

    # ── Vectorized API (VBT 전략 preprocessor용) ──

    def classify_series(self, closes: pd.Series) -> pd.DataFrame:
        """전체 시리즈에서 레짐 분류 (벡터화).

        Args:
            closes: 종가 시리즈 (DatetimeIndex 권장)

        Returns:
            DataFrame with columns:
                regime_label (str), p_trending, p_ranging, p_volatile,
                rv_ratio, efficiency_ratio
        """
        cfg = self._config

        # 로그 수익률
        log_returns = np.log(closes / closes.shift(1))

        # RV (realized volatility) — rolling std of log returns
        rv_short = log_returns.rolling(
            window=cfg.rv_short_window, min_periods=cfg.rv_short_window
        ).std()
        rv_long = log_returns.rolling(
            window=cfg.rv_long_window, min_periods=cfg.rv_long_window
        ).std()

        # RV Ratio: 단기/장기 변동성 비율
        rv_ratio: pd.Series = rv_short / rv_long.replace(0, np.nan)  # type: ignore[assignment]

        # Efficiency Ratio: |net_change| / sum(|changes|)
        net_change = closes.diff(cfg.er_window).abs()
        sum_changes = (
            closes.diff().abs().rolling(window=cfg.er_window, min_periods=cfg.er_window).sum()
        )
        er: pd.Series = net_change / sum_changes.replace(0, np.nan)  # type: ignore[assignment]

        # Independent scores (각 레짐 독립 계산)
        # 1. TRENDING: ER 기반 방향성 강도
        s_trending = pd.Series(
            _sigmoid(er, cfg.er_trending_threshold, scale=6.0),
            index=closes.index,
        )

        # 2. VOLATILE: RV ratio 기반 변동성 확장 (ER 결합 제거)
        s_volatile = pd.Series(
            _sigmoid(rv_ratio, cfg.rv_expansion_threshold, scale=8.0),
            index=closes.index,
        )

        # 3. RANGING: 변동성 수축 * 저방향성 (독립 지표)
        contraction = pd.Series(
            1.0 - _sigmoid(rv_ratio, 1.0, scale=5.0),
            index=closes.index,
        )
        s_ranging = contraction * (1.0 - s_trending)

        # NaN 마스크: warmup 기간 중 NaN 유지
        nan_mask = rv_ratio.isna() | er.isna()
        s_trending = s_trending.where(~nan_mask, np.nan)
        s_volatile = s_volatile.where(~nan_mask, np.nan)
        s_ranging = s_ranging.where(~nan_mask, np.nan)

        # Normalize: 3개 독립 점수를 확률 분포로 정규화
        total = s_trending + s_ranging + s_volatile
        total = total.replace(0, np.nan)
        p_trending = s_trending / total
        p_ranging = s_ranging / total
        p_volatile = s_volatile / total

        # Hard label (argmax) — NaN 행은 NaN 유지
        probs = pd.DataFrame(
            {
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
            }
        )
        valid_mask: pd.Series = probs.notna().all(axis=1)  # type: ignore[assignment]
        label_map = {
            "p_trending": RegimeLabel.TRENDING,
            "p_ranging": RegimeLabel.RANGING,
            "p_volatile": RegimeLabel.VOLATILE,
        }
        raw_labels = pd.Series(np.nan, index=closes.index, dtype=object)
        if valid_mask.any():  # type: ignore[truthy-bool]
            idx_max: pd.Series = probs[valid_mask].idxmax(axis=1)  # type: ignore[assignment]
            raw_labels[valid_mask] = idx_max.map(label_map)

        # Hysteresis: min_hold_bars 동안 레짐 유지
        regime_labels = self._apply_hysteresis(raw_labels, cfg.min_hold_bars)

        result = pd.DataFrame(
            {
                "regime_label": regime_labels,
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "rv_ratio": rv_ratio,
                "efficiency_ratio": er,
            },
            index=closes.index,
        )

        return result

    @staticmethod
    def _apply_hysteresis(
        raw_labels: pd.Series,
        min_hold_bars: int,
    ) -> pd.Series:
        """Hysteresis 적용 (모듈 레벨 함수 위임)."""
        return apply_hysteresis(raw_labels, min_hold_bars)

    # ── Incremental API (EDA / 라이브용) ──

    def update(self, symbol: str, close: float) -> RegimeState | None:
        """Bar 단위 incremental 업데이트.

        Args:
            symbol: 거래 심볼
            close: 현재 종가

        Returns:
            RegimeState 또는 warmup 중 None
        """
        cfg = self._config
        max_buf = cfg.rv_long_window + 5

        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=max_buf)
            self._hold_counters[symbol] = 0
            self._pending_labels[symbol] = None

        buf = self._buffers[symbol]
        buf.append(close)

        # warmup 체크
        if len(buf) < cfg.rv_long_window + 1:
            return None

        # 최근 close들로 Series 생성
        prices = pd.Series(list(buf))
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # RV 계산
        rv_short = float(log_returns.iloc[-cfg.rv_short_window :].std())
        rv_long = float(log_returns.iloc[-cfg.rv_long_window :].std())
        rv_ratio = rv_short / rv_long if rv_long > 0 else 1.0

        # ER 계산
        recent_prices = prices.iloc[-cfg.er_window - 1 :]
        if len(recent_prices) > cfg.er_window:
            net_change = abs(float(recent_prices.iloc[-1] - recent_prices.iloc[-cfg.er_window - 1]))
            sum_changes = float(recent_prices.diff().abs().sum())
            er = net_change / sum_changes if sum_changes > 0 else 0.0
        else:
            er = 0.0

        # Independent scores
        s_trending = float(_sigmoid(np.array([er]), cfg.er_trending_threshold, scale=6.0)[0])
        s_volatile = float(_sigmoid(np.array([rv_ratio]), cfg.rv_expansion_threshold, scale=8.0)[0])
        contraction = 1.0 - float(_sigmoid(np.array([rv_ratio]), 1.0, scale=5.0)[0])
        s_ranging = contraction * (1.0 - s_trending)

        # Normalize
        total = s_trending + s_ranging + s_volatile
        if total > 0:
            p_trending = s_trending / total
            p_ranging = s_ranging / total
            p_volatile = s_volatile / total
        else:
            p_trending = 0.0
            p_ranging = 1.0
            p_volatile = 0.0

        # Hard label (argmax)
        probs = {"trending": p_trending, "ranging": p_ranging, "volatile": p_volatile}
        raw_label = RegimeLabel(max(probs, key=probs.get))  # type: ignore[arg-type]

        # Hysteresis (matches vectorized apply_hysteresis: same pending label required)
        prev_state = self._states.get(symbol)
        if prev_state is not None and raw_label != prev_state.label:
            pending = self._pending_labels[symbol]
            if raw_label == pending:
                # Same pending label — increment counter
                self._hold_counters[symbol] += 1
                if self._hold_counters[symbol] >= cfg.min_hold_bars:
                    label = raw_label
                    bars_held = 1
                    self._hold_counters[symbol] = 0
                    self._pending_labels[symbol] = None
                else:
                    label = prev_state.label
                    bars_held = prev_state.bars_held + 1
            else:
                # New pending label — reset counter to 1
                self._pending_labels[symbol] = raw_label
                self._hold_counters[symbol] = 1
                label = prev_state.label
                bars_held = prev_state.bars_held + 1
        else:
            label = raw_label
            bars_held = (prev_state.bars_held + 1) if prev_state is not None else 1
            self._hold_counters[symbol] = 0
            self._pending_labels[symbol] = None

        state = RegimeState(
            label=label,
            probabilities={
                "trending": p_trending,
                "ranging": p_ranging,
                "volatile": p_volatile,
            },
            bars_held=bars_held,
            raw_indicators={"rv_ratio": rv_ratio, "er": er},
        )
        self._states[symbol] = state
        return state

    def get_regime(self, symbol: str) -> RegimeState | None:
        """현재 레짐 상태 조회.

        Args:
            symbol: 거래 심볼

        Returns:
            RegimeState 또는 미등록 시 None
        """
        return self._states.get(symbol)


def add_regime_columns(
    df: pd.DataFrame,
    config: RegimeDetectorConfig | None = None,
) -> pd.DataFrame:
    """DataFrame에 레짐 컬럼을 추가하는 편의 함수.

    전략의 preprocess()에서 한 줄로 사용:
        df = add_regime_columns(df, config)

    추가되는 컬럼:
        regime_label, p_trending, p_ranging, p_volatile,
        rv_ratio, efficiency_ratio

    Args:
        df: OHLCV DataFrame (close 컬럼 필수)
        config: 레짐 감지 설정 (None이면 기본값)

    Returns:
        레짐 컬럼이 추가된 새 DataFrame

    Raises:
        ValueError: close 컬럼 누락 시
    """
    if "close" not in df.columns:
        msg = "DataFrame must contain 'close' column"
        raise ValueError(msg)

    detector = RegimeDetector(config)
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    regime_df = detector.classify_series(close_series)
    return pd.concat([df, regime_df], axis=1)
