"""Vol-Structure Based Regime Detector.

단기/장기 변동성 비율과 정규화된 모멘텀을 사용하여 레짐을 분류합니다.
Sigmoid 함수로 soft probability를 출력합니다.

Vectorized API: classify_series() — VBT 전략 preprocessor용
Incremental API: update() — EDA / 라이브용

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.regime.config import RegimeLabel, VolStructureDetectorConfig
from src.regime.detector import RegimeState


def _sigmoid(
    x: npt.NDArray[np.float64] | pd.Series,
    center: float,
    scale: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Sigmoid 변환 (soft classification)."""
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-scale * (arr - center)))


@dataclass
class _SymbolBuffer:
    """심볼별 incremental 상태."""

    closes: deque[float]
    last_state: RegimeState | None = None


class VolStructureDetector:
    """Vol-Structure 기반 시장 레짐 감지기.

    vol_ratio (단기/장기 변동성 비율)와 norm_momentum (정규화 모멘텀)을
    sigmoid 확률로 변환하여 TRENDING/RANGING/VOLATILE을 분류합니다.

    알고리즘:
        expansion_score = sigmoid(vol_ratio - 1.0)  # 변동성 확장
        momentum_score = sigmoid(|norm_mom|)          # 방향성 강도
        p_trending = expansion_score * momentum_score
        p_volatile = expansion_score * (1 - momentum_score)
        p_ranging  = 1 - p_trending - p_volatile

    Args:
        config: Vol-Structure 감지기 설정
    """

    def __init__(self, config: VolStructureDetectorConfig | None = None) -> None:
        self._config = config or VolStructureDetectorConfig()
        self._buffers: dict[str, _SymbolBuffer] = {}

    @property
    def config(self) -> VolStructureDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods

    # ── Vectorized API ──

    def classify_series(self, closes: pd.Series) -> pd.DataFrame:
        """전체 시리즈에서 Vol-Structure 레짐 분류 (벡터화).

        Args:
            closes: 종가 시리즈

        Returns:
            DataFrame with columns:
                p_trending, p_ranging, p_volatile, vol_ratio, norm_momentum
        """
        cfg = self._config
        log_returns = np.log(closes / closes.shift(1))

        # 단기/장기 변동성
        vol_short = log_returns.rolling(
            window=cfg.vol_short_window, min_periods=cfg.vol_short_window
        ).std()
        vol_long = log_returns.rolling(
            window=cfg.vol_long_window, min_periods=cfg.vol_long_window
        ).std()

        vol_ratio: pd.Series = vol_short / vol_long.replace(0, np.nan)  # type: ignore[assignment]

        # Normalized momentum
        returns_sum = log_returns.rolling(window=cfg.mom_window, min_periods=cfg.mom_window).sum()
        returns_std = log_returns.rolling(window=cfg.mom_window, min_periods=cfg.mom_window).std()
        norm_mom: pd.Series = returns_sum / returns_std.replace(0, np.nan)  # type: ignore[assignment]

        # Soft probabilities via sigmoid
        expansion_score = pd.Series(
            _sigmoid(vol_ratio - 1.0, center=0.0, scale=5.0),
            index=closes.index,
        )
        momentum_score = pd.Series(
            _sigmoid(norm_mom.abs(), center=1.0, scale=3.0),
            index=closes.index,
        )

        p_trending = expansion_score * momentum_score
        p_volatile = expansion_score * (1.0 - momentum_score)

        # NaN 마스크
        nan_mask = vol_ratio.isna() | norm_mom.isna()
        p_trending = p_trending.where(~nan_mask, np.nan)
        p_volatile = p_volatile.where(~nan_mask, np.nan)

        # Normalize: p_trending + p_volatile > 1.0 방지
        total_tv = p_trending + p_volatile
        excess_mask = total_tv > 1.0
        scale_factor = 1.0 / total_tv.where(excess_mask, 1.0)
        p_trending = p_trending * scale_factor.where(excess_mask, 1.0)
        p_volatile = p_volatile * scale_factor.where(excess_mask, 1.0)

        p_ranging = (1.0 - p_trending - p_volatile).clip(lower=0.0)

        return pd.DataFrame(
            {
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "vol_ratio": vol_ratio,
                "norm_momentum": norm_mom,
            },
            index=closes.index,
        )

    # ── Incremental API ──

    def update(self, symbol: str, close: float) -> RegimeState | None:
        """Bar 단위 incremental 업데이트.

        Args:
            symbol: 거래 심볼
            close: 현재 종가

        Returns:
            RegimeState 또는 warmup 중 None
        """
        cfg = self._config
        max_buf = cfg.vol_long_window + cfg.mom_window + 10

        if symbol not in self._buffers:
            self._buffers[symbol] = _SymbolBuffer(closes=deque(maxlen=max_buf))

        buf = self._buffers[symbol]
        buf.closes.append(close)

        if len(buf.closes) < cfg.vol_long_window + 2:
            return None

        prices = pd.Series(list(buf.closes))
        log_returns = np.log(prices / prices.shift(1)).dropna()

        if len(log_returns) < cfg.vol_long_window:
            return None

        # Vol ratio
        vol_short = float(log_returns.iloc[-cfg.vol_short_window :].std())
        vol_long = float(log_returns.iloc[-cfg.vol_long_window :].std())
        vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

        # Norm momentum
        if len(log_returns) >= cfg.mom_window:
            recent = log_returns.iloc[-cfg.mom_window :]
            r_sum = float(recent.sum())
            r_std = float(recent.std())
            norm_mom = r_sum / r_std if r_std > 0 else 0.0
        else:
            norm_mom = 0.0

        # Sigmoid probabilities
        expansion_score = float(_sigmoid(np.array([vol_ratio - 1.0]), center=0.0, scale=5.0)[0])
        momentum_score = float(_sigmoid(np.array([abs(norm_mom)]), center=1.0, scale=3.0)[0])

        p_trending = expansion_score * momentum_score
        p_volatile = expansion_score * (1.0 - momentum_score)

        # Normalize
        total = p_trending + p_volatile
        if total > 1.0:
            p_trending /= total
            p_volatile /= total
        p_ranging = max(0.0, 1.0 - p_trending - p_volatile)

        # Hard label
        probs = {"trending": p_trending, "ranging": p_ranging, "volatile": p_volatile}
        label = RegimeLabel(max(probs, key=probs.get))  # type: ignore[arg-type]

        state = RegimeState(
            label=label,
            probabilities=probs,
            bars_held=1,
            raw_indicators={"vol_ratio": vol_ratio, "norm_momentum": norm_mom},
        )
        buf.last_state = state
        return state
