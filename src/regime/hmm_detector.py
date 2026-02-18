"""HMM-Based Regime Detector.

GaussianHMM expanding window training으로 Bull/Bear/Sideways를 분류하여
TRENDING/RANGING 확률을 출력합니다. HMM은 VOLATILE을 직접 감지하지 않으며,
Rule-Based(RV ratio)와 Vol-Structure가 담당합니다.

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

from src.regime.config import HMMDetectorConfig, RegimeLabel

if TYPE_CHECKING:
    from src.regime.detector import RegimeState

logger = logging.getLogger(__name__)

# Lazy import: hmmlearn is optional
_hmm_available = False
_GaussianHMM: type[Any] | None = None
try:
    from hmmlearn.hmm import (
        GaussianHMM as _GaussianHMM_cls,  # pyright: ignore[reportMissingImports]
    )

    _GaussianHMM = _GaussianHMM_cls
    _hmm_available = True
except ImportError:
    pass

# Public alias for external use
HMM_AVAILABLE: bool = _hmm_available

# Constant for 2-state HMM
_TWO_STATES = 2


def _map_state_to_regime(
    means: np.ndarray,  # type: ignore[type-arg]
    state: int,
    n_states: int,
) -> int:
    """Map HMM state to regime label based on mean return ordering.

    Bull (highest mean) -> 1, Bear (lowest mean) -> -1, Middle -> 0
    """
    sorted_states = np.argsort(means)
    if n_states == _TWO_STATES:
        mapping = {int(sorted_states[0]): -1, int(sorted_states[1]): 1}
    else:
        mapping = {}
        mapping[int(sorted_states[0])] = -1
        mapping[int(sorted_states[-1])] = 1
        for idx in range(1, n_states - 1):
            mapping[int(sorted_states[idx])] = 0
    return mapping.get(state, 0)


class HMMDetector:
    """HMM 기반 시장 레짐 감지기.

    GaussianHMM expanding window로 Bull/Bear/Sideways를 분류하고,
    이를 TRENDING/RANGING 확률로 매핑합니다.

    Bull/Bear -> TRENDING (양방향 추세)
    Sideways -> RANGING (횡보)
    VOLATILE -> 0.0 (HMM은 감지 불가, 다른 감지기가 담당)

    Args:
        config: HMM 감지기 설정
    """

    def __init__(self, config: HMMDetectorConfig | None = None) -> None:
        if not _hmm_available or _GaussianHMM is None:
            msg = "hmmlearn is required for HMMDetector. Install with: uv add hmmlearn"
            raise ImportError(msg)
        self._config = config or HMMDetectorConfig()
        self._hmm_cls = _GaussianHMM
        # Incremental state per symbol
        self._buffers: dict[str, deque[float]] = {}
        self._models: dict[str, Any] = {}  # last trained GaussianHMM
        self._bar_counts: dict[str, int] = {}
        # Convergence tracking
        self._fit_attempts: int = 0
        self._fit_failures: int = 0

    @property
    def config(self) -> HMMDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods

    @property
    def convergence_rate(self) -> float:
        """학습 수렴률 (0~1). 1.0이면 모든 학습 시도 성공."""
        if self._fit_attempts == 0:
            return 1.0
        return 1.0 - (self._fit_failures / self._fit_attempts)

    # ── Vectorized API ──

    def classify_series(self, closes: pd.Series) -> pd.DataFrame:
        """전체 시리즈에서 HMM 레짐 분류 (벡터화 불가 -- expanding window).

        Args:
            closes: 종가 시리즈

        Returns:
            DataFrame with columns:
                p_trending, p_ranging, p_volatile, hmm_state, hmm_prob
        """
        cfg = self._config
        n = len(closes)

        # 수익률 + 롤링 변동성
        returns = np.log(closes / closes.shift(1)) if cfg.use_log_returns else closes.pct_change()
        rolling_vol = returns.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()

        returns_arr = returns.to_numpy().astype(np.float64)
        vol_arr = rolling_vol.to_numpy().astype(np.float64)

        # Output arrays
        p_trending = np.full(n, np.nan)
        p_ranging = np.full(n, np.nan)
        p_volatile = np.full(n, 0.0)
        hmm_state = np.full(n, -1)
        hmm_prob = np.full(n, 0.0)

        last_model: Any = None

        rng = np.random.default_rng(42)

        for i in range(cfg.min_train_window, n):
            # Retrain at intervals
            if (i - cfg.min_train_window) % cfg.retrain_interval == 0:
                # Sliding window: 최근 N bar만 사용
                start_idx = max(0, i - cfg.sliding_window) if cfg.sliding_window > 0 else 0

                features = np.column_stack([returns_arr[start_idx:i], vol_arr[start_idx:i]])
                valid_mask = ~np.isnan(features).any(axis=1)
                features_clean = features[valid_mask]

                if len(features_clean) < cfg.n_states * 10:
                    continue

                # Decay weighting via bootstrap resampling
                if cfg.decay_half_life > 0 and len(features_clean) > 1:
                    n_samples = len(features_clean)
                    decay_weights = np.exp(
                        -np.log(2.0) * np.arange(n_samples)[::-1] / cfg.decay_half_life
                    )
                    prob = decay_weights / decay_weights.sum()
                    indices = rng.choice(n_samples, size=n_samples, p=prob)
                    features_clean = features_clean[indices]

                self._fit_attempts += 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        model = self._hmm_cls(
                            n_components=cfg.n_states,
                            n_iter=cfg.n_iter,
                            covariance_type="full",
                            random_state=42,
                        )
                        model.fit(features_clean)
                        last_model = model
                    except Exception:
                        self._fit_failures += 1
                        logger.debug("HMM training failed at bar %d", i)
                        continue

            if last_model is not None:
                feat_i = np.array([[returns_arr[i], vol_arr[i]]])
                if not np.isnan(feat_i).any():
                    try:
                        state = int(last_model.predict(feat_i)[0])
                        posteriors = last_model.predict_proba(feat_i)[0]
                        means = last_model.means_[:, 0]

                        regime = _map_state_to_regime(means, state, cfg.n_states)
                        hmm_state[i] = regime
                        hmm_prob[i] = float(posteriors[state])

                        # Posterior -> TRENDING/RANGING probability
                        sorted_states = np.argsort(means)
                        bull_idx = int(sorted_states[-1])
                        bear_idx = int(sorted_states[0])

                        pt = float(posteriors[bull_idx] + posteriors[bear_idx])
                        if cfg.n_states == _TWO_STATES:
                            pr = 0.0
                        else:
                            pr = float(
                                sum(
                                    posteriors[int(sorted_states[k])]
                                    for k in range(1, cfg.n_states - 1)
                                )
                            )

                        p_trending[i] = pt
                        p_ranging[i] = pr
                        p_volatile[i] = 0.0
                    except Exception:
                        logger.debug("HMM prediction failed at bar %d", i)

        return pd.DataFrame(
            {
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "hmm_state": hmm_state,
                "hmm_prob": hmm_prob,
            },
            index=closes.index,
        )

    # ── Incremental API ──

    def _retrain_incremental(
        self,
        symbol: str,
        returns_arr: np.ndarray,  # type: ignore[type-arg]
        vol_arr: np.ndarray,  # type: ignore[type-arg]
    ) -> None:
        """Incremental 학습 (sliding window + decay 적용)."""
        cfg = self._config
        arr_len = len(returns_arr) - 1  # exclude last (current bar)

        start_idx = max(0, arr_len - cfg.sliding_window) if cfg.sliding_window > 0 else 0

        features = np.column_stack([returns_arr[start_idx:arr_len], vol_arr[start_idx:arr_len]])
        valid_mask = ~np.isnan(features).any(axis=1)
        features_clean = features[valid_mask]

        if len(features_clean) < cfg.n_states * 10:
            return

        # Decay weighting via bootstrap
        if cfg.decay_half_life > 0 and len(features_clean) > 1:
            rng = np.random.default_rng(42)
            n_samples = len(features_clean)
            decay_weights = np.exp(-np.log(2.0) * np.arange(n_samples)[::-1] / cfg.decay_half_life)
            prob = decay_weights / decay_weights.sum()
            indices = rng.choice(n_samples, size=n_samples, p=prob)
            features_clean = features_clean[indices]

        self._fit_attempts += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = self._hmm_cls(
                    n_components=cfg.n_states,
                    n_iter=cfg.n_iter,
                    covariance_type="full",
                    random_state=42,
                )
                model.fit(features_clean)
                self._models[symbol] = model
            except Exception:
                self._fit_failures += 1
                logger.debug("HMM training failed for %s", symbol)

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
        max_buf = cfg.min_train_window + cfg.vol_window + 10

        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=max_buf)
            self._bar_counts[symbol] = 0

        buf = self._buffers[symbol]
        buf.append(close)
        self._bar_counts[symbol] += 1

        if len(buf) < cfg.min_train_window + 1:
            return None

        # Build series
        prices = pd.Series(list(buf))
        returns = np.log(prices / prices.shift(1)) if cfg.use_log_returns else prices.pct_change()
        rolling_vol = returns.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()

        returns_arr = returns.to_numpy().astype(np.float64)
        vol_arr = rolling_vol.to_numpy().astype(np.float64)

        # Retrain if needed
        bar_count = self._bar_counts[symbol]
        if (
            symbol not in self._models
            or (bar_count - cfg.min_train_window) % cfg.retrain_interval == 0
        ):
            self._retrain_incremental(symbol, returns_arr, vol_arr)

        model = self._models.get(symbol)
        if model is None:
            return None

        feat_i = np.array([[returns_arr[-1], vol_arr[-1]]])
        if np.isnan(feat_i).any():
            return None

        try:
            state = int(model.predict(feat_i)[0])
            posteriors = model.predict_proba(feat_i)[0]
            means = model.means_[:, 0]

            sorted_states = np.argsort(means)
            bull_idx = int(sorted_states[-1])
            bear_idx = int(sorted_states[0])

            pt = float(posteriors[bull_idx] + posteriors[bear_idx])
            if cfg.n_states == _TWO_STATES:
                pr = 0.0
            else:
                pr = float(
                    sum(posteriors[int(sorted_states[k])] for k in range(1, cfg.n_states - 1))
                )

            regime = _map_state_to_regime(means, state, cfg.n_states)
            label = RegimeLabel.TRENDING if regime != 0 else RegimeLabel.RANGING
        except Exception:
            return None

        return RegimeState(
            label=label,
            probabilities={"trending": pt, "ranging": pr, "volatile": 0.0},
            bars_held=1,
            raw_indicators={"hmm_state": float(regime), "hmm_prob": float(posteriors[state])},
        )
