"""Derivatives Regime Detector.

Funding rate, OI 변화율, funding persistence 기반으로
레버리지 축적/해소 및 cascade risk를 감지합니다.

Dual API:
    - classify_series(): Vectorized (backtest)
    - update(): Incremental (live)

Rules Applied:
    - #12 Data Engineering: Vectorization (classify_series)
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.regime.config import DerivativesDetectorConfig, RegimeLabel
from src.regime.detector import RegimeState


def _sigmoid_scalar(x: float, center: float, scale: float = 10.0) -> float:
    """Scalar sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))


def _sigmoid_array(x: pd.Series, center: float, scale: float = 10.0) -> pd.Series:
    """Vectorized sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))  # type: ignore[return-value]


@dataclass
class _SymbolState:
    """Per-symbol incremental state."""

    funding_buf: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    oi_buf: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    persistence_count: int = 0
    last_funding_sign: int = 0
    cascade_risk: float = 0.0


class DerivativesDetector:
    """Derivatives 기반 보조 레짐 감지기.

    funding_rate + oi 기반으로 레버리지 상태를 추론하여
    TRENDING/RANGING/VOLATILE 확률을 제공합니다.

    Args:
        config: DerivativesDetectorConfig
    """

    def __init__(self, config: DerivativesDetectorConfig | None = None) -> None:
        self._config = config or DerivativesDetectorConfig()
        self._states: dict[str, _SymbolState] = {}
        self._regime_states: dict[str, RegimeState] = {}

    @property
    def config(self) -> DerivativesDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods

    # ── Vectorized API ──

    def classify_series(self, deriv_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized derivatives regime 분류.

        Args:
            deriv_df: DataFrame with at least 'funding_rate' and 'oi' columns.
                      Optional: 'ls_ratio', 'taker_ratio', 'liquidation_volume'

        Returns:
            DataFrame with columns:
                regime_label, p_trending, p_ranging, p_volatile,
                rv_ratio (placeholder 0), efficiency_ratio (placeholder 0),
                confidence (always 1.0 for single detector)
        """
        cfg = self._config
        n = len(deriv_df)

        # Funding rate z-score
        fr = deriv_df.get("funding_rate")
        if fr is None:
            return self._empty_result(deriv_df.index, n)
        fr = fr.astype(float)

        fr_mean = fr.rolling(
            window=cfg.funding_zscore_window, min_periods=cfg.funding_zscore_window
        ).mean()
        fr_std = fr.rolling(
            window=cfg.funding_zscore_window, min_periods=cfg.funding_zscore_window
        ).std()
        fr_std = fr_std.replace(0, np.nan)
        funding_zscore: pd.Series = (fr - fr_mean) / fr_std  # type: ignore[assignment]

        # OI change rate
        oi = deriv_df.get("oi")
        if oi is not None:
            oi = oi.astype(float)
            oi_shifted = oi.shift(cfg.oi_change_window)
            oi_change: pd.Series = (oi - oi_shifted) / oi_shifted.replace(0, np.nan)  # type: ignore[assignment]
        else:
            oi_change = pd.Series(0.0, index=deriv_df.index)

        # Funding persistence (consecutive same-sign funding bars)
        funding_sign = np.sign(fr)
        persistence = self._compute_persistence_vectorized(funding_sign)

        # Cascade risk: weighted combination
        cascade_risk = self._compute_cascade_risk_vectorized(
            funding_zscore, oi_change, persistence, cfg
        )

        # Classification scores
        abs_fz = funding_zscore.abs()
        s_volatile = _sigmoid_array(abs_fz, center=1.5, scale=4.0) * _sigmoid_array(
            cascade_risk, center=0.4, scale=6.0
        )
        s_trending = _sigmoid_array(abs_fz, center=0.5, scale=3.0) * (1.0 - s_volatile)
        s_ranging: pd.Series = (1.0 - s_trending - s_volatile).clip(lower=0.0)  # type: ignore[assignment]

        # Normalize
        total = s_trending + s_ranging + s_volatile
        total = total.replace(0, np.nan)
        p_trending = s_trending / total
        p_ranging = s_ranging / total
        p_volatile = s_volatile / total

        # NaN mask
        nan_mask = funding_zscore.isna()
        p_trending = p_trending.where(~nan_mask, np.nan)
        p_ranging = p_ranging.where(~nan_mask, np.nan)
        p_volatile = p_volatile.where(~nan_mask, np.nan)

        # Hard labels
        probs_df = pd.DataFrame(
            {"p_trending": p_trending, "p_ranging": p_ranging, "p_volatile": p_volatile}
        )
        valid_mask: pd.Series = probs_df.notna().all(axis=1)  # type: ignore[assignment]
        label_map = {
            "p_trending": RegimeLabel.TRENDING,
            "p_ranging": RegimeLabel.RANGING,
            "p_volatile": RegimeLabel.VOLATILE,
        }
        raw_labels = pd.Series(np.nan, index=deriv_df.index, dtype=object)
        if valid_mask.any():  # type: ignore[truthy-bool]
            idx_max: pd.Series = probs_df[valid_mask].idxmax(axis=1)  # type: ignore[assignment]
            raw_labels[valid_mask] = idx_max.map(label_map)

        return pd.DataFrame(
            {
                "regime_label": raw_labels,
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "rv_ratio": 0.0,
                "efficiency_ratio": 0.0,
                "confidence": 1.0,
            },
            index=deriv_df.index,
        )

    def get_cascade_risk_series(self, deriv_df: pd.DataFrame) -> pd.Series:
        """Vectorized cascade risk 시리즈 반환.

        Args:
            deriv_df: derivatives DataFrame

        Returns:
            cascade risk 시리즈 (0~1)
        """
        cfg = self._config

        fr = deriv_df.get("funding_rate")
        if fr is None:
            return pd.Series(0.0, index=deriv_df.index)
        fr = fr.astype(float)

        fr_mean = fr.rolling(
            window=cfg.funding_zscore_window, min_periods=cfg.funding_zscore_window
        ).mean()
        fr_std = fr.rolling(
            window=cfg.funding_zscore_window, min_periods=cfg.funding_zscore_window
        ).std()
        fr_std = fr_std.replace(0, np.nan)
        funding_zscore: pd.Series = (fr - fr_mean) / fr_std  # type: ignore[assignment]

        oi = deriv_df.get("oi")
        if oi is not None:
            oi = oi.astype(float)
            oi_shifted = oi.shift(cfg.oi_change_window)
            oi_change: pd.Series = (oi - oi_shifted) / oi_shifted.replace(0, np.nan)  # type: ignore[assignment]
        else:
            oi_change = pd.Series(0.0, index=deriv_df.index)

        funding_sign = np.sign(fr)
        persistence = self._compute_persistence_vectorized(funding_sign)

        return self._compute_cascade_risk_vectorized(funding_zscore, oi_change, persistence, cfg)

    @staticmethod
    def _compute_persistence_vectorized(funding_sign: pd.Series) -> pd.Series:
        """Consecutive same-sign funding bar count."""
        result = np.zeros(len(funding_sign))
        count = 0
        prev_sign = 0.0
        for i, sign in enumerate(funding_sign):
            if np.isnan(sign):
                count = 0
            elif sign == prev_sign:
                count += 1
            else:
                count = 1
            prev_sign = sign if not np.isnan(sign) else 0.0
            result[i] = count
        return pd.Series(result, index=funding_sign.index)

    @staticmethod
    def _compute_cascade_risk_vectorized(
        funding_zscore: pd.Series,
        oi_change: pd.Series,
        persistence: pd.Series,
        cfg: DerivativesDetectorConfig,
    ) -> pd.Series:
        """Cascade risk = weighted combination of 3 features, clipped 0~1."""
        abs_fz = funding_zscore.abs().fillna(0.0)
        abs_oi = oi_change.abs().fillna(0.0)
        norm_persistence = (persistence / cfg.funding_persistence_window).clip(upper=1.0)

        # Weighted sum: funding zscore (40%), OI change (30%), persistence (30%)
        raw = (
            0.4 * _sigmoid_array(abs_fz, center=1.5, scale=3.0)
            + 0.3 * _sigmoid_array(abs_oi, center=0.1, scale=10.0)
            + 0.3 * norm_persistence
        )

        return raw.clip(lower=0.0, upper=1.0)

    def _empty_result(self, index: pd.Index, n: int) -> pd.DataFrame:  # type: ignore[type-arg]
        """빈 결과 DataFrame (funding_rate 누락 시)."""
        return pd.DataFrame(
            {
                "regime_label": pd.Series(np.nan, index=index, dtype=object),
                "p_trending": np.nan,
                "p_ranging": np.nan,
                "p_volatile": np.nan,
                "rv_ratio": 0.0,
                "efficiency_ratio": 0.0,
                "confidence": 1.0,
            },
            index=index,
        )

    # ── Incremental API ──

    def update(
        self,
        symbol: str,
        funding_rate: float,
        oi: float = 0.0,
        **_kwargs: float,
    ) -> RegimeState | None:
        """Bar 단위 incremental 업데이트.

        Args:
            symbol: 거래 심볼
            funding_rate: 현재 funding rate
            oi: 현재 open interest
            **_kwargs: 추가 derivatives 데이터 (ls_ratio 등, 현재 미사용)

        Returns:
            RegimeState 또는 warmup 중 None
        """
        cfg = self._config

        if symbol not in self._states:
            self._states[symbol] = _SymbolState(
                funding_buf=deque(
                    maxlen=max(cfg.funding_zscore_window, cfg.funding_persistence_window) + 1
                ),
                oi_buf=deque(maxlen=cfg.oi_change_window + 1),
            )

        ss = self._states[symbol]
        ss.funding_buf.append(funding_rate)
        ss.oi_buf.append(oi)

        # Warmup check
        if len(ss.funding_buf) < cfg.funding_zscore_window:
            return None

        # Funding z-score
        fr_arr = np.array(ss.funding_buf, dtype=np.float64)
        window_data = fr_arr[-cfg.funding_zscore_window :]
        fr_mean = float(window_data.mean())
        fr_std = float(window_data.std())
        funding_zscore = (funding_rate - fr_mean) / fr_std if fr_std > 0 else 0.0

        # OI change rate
        if len(ss.oi_buf) > cfg.oi_change_window and ss.oi_buf[-cfg.oi_change_window - 1] > 0:
            prev_oi = ss.oi_buf[-cfg.oi_change_window - 1]
            oi_change = (oi - prev_oi) / prev_oi
        else:
            oi_change = 0.0

        # Funding persistence
        current_sign = 1 if funding_rate > 0 else (-1 if funding_rate < 0 else 0)
        if current_sign == ss.last_funding_sign and current_sign != 0:
            ss.persistence_count += 1
        else:
            ss.persistence_count = 1 if current_sign != 0 else 0
        ss.last_funding_sign = current_sign

        norm_persistence = min(ss.persistence_count / cfg.funding_persistence_window, 1.0)

        # Cascade risk
        cascade_risk = (
            0.4 * _sigmoid_scalar(abs(funding_zscore), center=1.5, scale=3.0)
            + 0.3 * _sigmoid_scalar(abs(oi_change), center=0.1, scale=10.0)
            + 0.3 * norm_persistence
        )
        cascade_risk = max(0.0, min(1.0, cascade_risk))
        ss.cascade_risk = cascade_risk

        # Classification
        abs_fz = abs(funding_zscore)
        s_volatile = _sigmoid_scalar(abs_fz, 1.5, 4.0) * _sigmoid_scalar(cascade_risk, 0.4, 6.0)
        s_trending = _sigmoid_scalar(abs_fz, 0.5, 3.0) * (1.0 - s_volatile)
        s_ranging = max(0.0, 1.0 - s_trending - s_volatile)

        total = s_trending + s_ranging + s_volatile
        if total > 0:
            p_trending = s_trending / total
            p_ranging = s_ranging / total
            p_volatile = s_volatile / total
        else:
            p_trending = 0.0
            p_ranging = 1.0
            p_volatile = 0.0

        probs = {"trending": p_trending, "ranging": p_ranging, "volatile": p_volatile}
        label = RegimeLabel(max(probs, key=probs.get))  # type: ignore[arg-type]

        prev = self._regime_states.get(symbol)
        bars_held = (prev.bars_held + 1) if prev is not None and prev.label == label else 1

        state = RegimeState(
            label=label,
            probabilities=probs,
            bars_held=bars_held,
            raw_indicators={
                "funding_zscore": funding_zscore,
                "oi_change": oi_change,
                "cascade_risk": cascade_risk,
            },
            confidence=1.0,
        )
        self._regime_states[symbol] = state
        return state

    def get_regime(self, symbol: str) -> RegimeState | None:
        """현재 레짐 상태 조회."""
        return self._regime_states.get(symbol)

    def get_cascade_risk(self, symbol: str) -> float:
        """현재 cascade risk score (0~1)."""
        ss = self._states.get(symbol)
        return ss.cascade_risk if ss is not None else 0.0
