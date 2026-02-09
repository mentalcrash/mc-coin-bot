"""MSAR (Markov-Switching AutoRegression) Regime Detector.

statsmodels MarkovAutoregression을 사용하여 AR 계수 기반으로
TRENDING/RANGING/VOLATILE 레짐을 분류합니다.

Regime 매핑:
    - AR 계수 합이 양수 + 큼 → TRENDING (momentum persistence)
    - AR 계수 합이 음수/작음 + 저분산 → RANGING (mean-reversion)
    - 고분산 → VOLATILE

Rules Applied:
    - #12 Data Engineering: ML training은 Zero Loop Policy 예외
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.regime.config import MSARDetectorConfig, RegimeLabel

if TYPE_CHECKING:
    from src.regime.detector import RegimeState

logger = logging.getLogger(__name__)

# Lazy import: statsmodels is optional
_msar_available = False
_MarkovAutoregression: type[Any] | None = None
try:
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression as _MARCls,  # pyright: ignore[reportMissingImports]
    )

    _MarkovAutoregression = _MARCls
    _msar_available = True
except ImportError:
    pass

# Public alias for external use
MSAR_AVAILABLE: bool = _msar_available

# Constants
_TWO_REGIMES = 2
_MAX_AR_LAG_SCAN = 10
_MAX_FIT_ITER = 200


def _map_regime_from_model(
    model_result: Any,
    k_regimes: int,
) -> dict[int, RegimeLabel]:
    """모델 결과로부터 variance 기반 레짐 매핑 생성.

    Args:
        model_result: MarkovAutoregression fit result
        k_regimes: 레짐 수

    Returns:
        state→RegimeLabel 매핑
    """
    params = model_result.params
    variances = np.zeros(k_regimes)

    for regime_idx in range(k_regimes):
        try:
            sigma_key = f"sigma2.{regime_idx}"
            if hasattr(params, "index") and sigma_key in params.index:
                variances[regime_idx] = float(params[sigma_key])
        except Exception:
            variances[regime_idx] = 0.0

    mapping: dict[int, RegimeLabel] = {}
    sorted_var = np.argsort(variances)

    if k_regimes == _TWO_REGIMES:
        mapping[int(sorted_var[-1])] = RegimeLabel.VOLATILE
        mapping[int(sorted_var[0])] = RegimeLabel.TRENDING
    else:
        mapping[int(sorted_var[-1])] = RegimeLabel.VOLATILE
        mapping[int(sorted_var[0])] = RegimeLabel.RANGING
        for idx in range(1, k_regimes - 1):
            mapping[int(sorted_var[idx])] = RegimeLabel.TRENDING

    return mapping


def _aggregate_regime_probs(
    smoothed_probs: np.ndarray,  # type: ignore[type-arg]
    mapping: dict[int, RegimeLabel],
    k_regimes: int,
) -> tuple[float, float, float]:
    """Smoothed probabilities를 regime별로 집계.

    Returns:
        (p_trending, p_ranging, p_volatile)
    """
    pt = 0.0
    pr = 0.0
    pv = 0.0

    for state_idx in range(k_regimes):
        prob_val = float(smoothed_probs[state_idx]) if state_idx < len(smoothed_probs) else 0.0
        label = mapping.get(state_idx, RegimeLabel.RANGING)
        if label == RegimeLabel.TRENDING:
            pt += prob_val
        elif label == RegimeLabel.RANGING:
            pr += prob_val
        else:
            pv += prob_val

    return pt, pr, pv


class MSARDetector:
    """MSAR 기반 시장 레짐 감지기.

    MarkovAutoregression으로 AR 계수 기반 레짐을 분류하고
    TRENDING/RANGING/VOLATILE 확률을 출력합니다.

    Args:
        config: MSAR 감지기 설정
    """

    def __init__(self, config: MSARDetectorConfig | None = None) -> None:
        if not _msar_available or _MarkovAutoregression is None:
            msg = "statsmodels is required for MSARDetector. Install with: uv add statsmodels"
            raise ImportError(msg)
        self._config = config or MSARDetectorConfig()
        self._msar_cls = _MarkovAutoregression
        # Incremental state per symbol
        self._buffers: dict[str, deque[float]] = {}
        self._models: dict[str, Any] = {}
        self._mappings: dict[str, dict[int, RegimeLabel]] = {}
        self._bar_counts: dict[str, int] = {}

    @property
    def config(self) -> MSARDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods

    def _fit_model(self, train_data: np.ndarray) -> tuple[Any, dict[int, RegimeLabel]] | None:  # type: ignore[type-arg]
        """MSAR 모델 학습.

        Returns:
            (model_result, mapping) 또는 실패 시 None
        """
        cfg = self._config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = self._msar_cls(
                    train_data,
                    k_regimes=cfg.k_regimes,
                    order=cfg.order,
                    switching_ar=cfg.switching_ar,
                    switching_variance=cfg.switching_variance,
                )
                result = model.fit(maxiter=_MAX_FIT_ITER, disp=False)
                mapping = _map_regime_from_model(result, cfg.k_regimes)
            except Exception:
                return None
            else:
                return result, mapping

    def _predict_bar(
        self,
        model_result: Any,
        mapping: dict[int, RegimeLabel],
        ret_idx: int,
    ) -> tuple[float, float, float, int] | None:
        """단일 bar의 레짐 확률 예측.

        Returns:
            (pt, pr, pv, state) 또는 실패 시 None
        """
        cfg = self._config
        smoothed = model_result.smoothed_marginal_probabilities
        if ret_idx < 0 or ret_idx >= len(smoothed):
            return None

        probs = smoothed[ret_idx]
        pt, pr, pv = _aggregate_regime_probs(probs, mapping, cfg.k_regimes)

        max_prob = max(pt, pr, pv)
        if max_prob == pt:
            state = 1
        elif max_prob == pv:
            state = -1
        else:
            state = 0

        return pt, pr, pv, state

    # ── Vectorized API ──

    def classify_series(self, closes: pd.Series) -> pd.DataFrame:
        """전체 시리즈에서 MSAR 레짐 분류.

        Args:
            closes: 종가 시리즈

        Returns:
            DataFrame with columns:
                p_trending, p_ranging, p_volatile, msar_state
        """
        cfg = self._config
        n = len(closes)

        returns = (
            np.log(closes / closes.shift(1)).dropna()
            if cfg.use_log_returns
            else closes.pct_change().dropna()
        )

        p_trending = np.full(n, np.nan)
        p_ranging = np.full(n, np.nan)
        p_volatile = np.full(n, np.nan)
        msar_state = np.full(n, -1, dtype=int)

        last_model_result: Any = None
        last_mapping: dict[int, RegimeLabel] = {}

        for i in range(cfg.min_train_window, n):
            if (i - cfg.min_train_window) % cfg.retrain_interval == 0:
                start_idx = max(0, i - cfg.sliding_window) if cfg.sliding_window > 0 else 0
                train_returns = returns.iloc[start_idx:i]

                if len(train_returns) >= cfg.min_train_window:
                    fit_result = self._fit_model(train_returns.to_numpy())
                    if fit_result is not None:
                        last_model_result, last_mapping = fit_result
                    else:
                        logger.debug("MSAR training failed at bar %d", i)

            if last_model_result is not None and last_mapping:
                try:
                    ret_idx = min(i - 1, len(last_model_result.smoothed_marginal_probabilities) - 1)
                    prediction = self._predict_bar(last_model_result, last_mapping, ret_idx)
                    if prediction is not None:
                        p_trending[i], p_ranging[i], p_volatile[i], msar_state[i] = prediction
                except Exception:
                    logger.debug("MSAR prediction failed at bar %d", i)

        return pd.DataFrame(
            {
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "msar_state": msar_state,
            },
            index=closes.index,
        )

    # ── Incremental API ──

    def _retrain_incremental(self, symbol: str, returns: pd.Series) -> None:
        """Incremental 학습."""
        cfg = self._config
        train_returns = (
            returns.iloc[-cfg.sliding_window :] if cfg.sliding_window > 0 else returns.iloc[:-1]
        )

        if len(train_returns) < cfg.min_train_window:
            return

        fit_result = self._fit_model(train_returns.to_numpy())
        if fit_result is not None:
            self._models[symbol] = fit_result[0]
            self._mappings[symbol] = fit_result[1]
        else:
            logger.debug("MSAR training failed for %s", symbol)

    def update(self, symbol: str, close: float) -> RegimeState | None:
        """Bar 단위 incremental 업데이트.

        Args:
            symbol: 거래 심볼
            close: 현재 종가

        Returns:
            RegimeState 또는 warmup 중 None
        """
        from src.regime.detector import RegimeState

        cfg = self._config
        max_buf = max(cfg.sliding_window, cfg.min_train_window) + cfg.order + 10

        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=max_buf)
            self._bar_counts[symbol] = 0

        buf = self._buffers[symbol]
        buf.append(close)
        self._bar_counts[symbol] += 1

        if len(buf) < cfg.min_train_window + 1:
            return None

        prices = pd.Series(list(buf))
        returns = (
            np.log(prices / prices.shift(1)).dropna()
            if cfg.use_log_returns
            else prices.pct_change().dropna()
        )

        bar_count = self._bar_counts[symbol]
        if (
            symbol not in self._models
            or (bar_count - cfg.min_train_window) % cfg.retrain_interval == 0
        ):
            self._retrain_incremental(symbol, returns)

        model_result = self._models.get(symbol)
        mapping = self._mappings.get(symbol, {})
        if model_result is None or not mapping:
            return None

        try:
            smoothed = model_result.smoothed_marginal_probabilities
            probs = smoothed[len(smoothed) - 1]
            pt, pr, pv = _aggregate_regime_probs(probs, mapping, cfg.k_regimes)

            regime_probs = {"trending": pt, "ranging": pr, "volatile": pv}
            hard_label = RegimeLabel(max(regime_probs, key=regime_probs.get))  # type: ignore[arg-type]
        except Exception:
            return None

        return RegimeState(
            label=hard_label,
            probabilities=regime_probs,
            bars_held=1,
            raw_indicators={"msar_state": 1.0 if hard_label == RegimeLabel.TRENDING else 0.0},
        )
