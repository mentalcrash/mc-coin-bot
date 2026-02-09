"""Ensemble Regime Detector.

Rule-Based + HMM + Vol-Structure 감지기의 가중 확률 블렌딩으로
TRENDING/RANGING/VOLATILE 레짐을 분류합니다.

Weighted Probability Blending:
    blended[regime] = sum(weight_i * p_i[regime]) / sum(weight_i)

Graceful Degradation:
    - HMM 비활성(hmmlearn 미설치 또는 config=None) 시 자동 제외
    - Warmup 완료된 감지기만 블렌딩 참여, 가중치 자동 재정규화

Rules Applied:
    - #12 Data Engineering: Vectorization (classify_series)
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.regime.config import (
    EnsembleRegimeDetectorConfig,
    RegimeLabel,
)
from src.regime.detector import RegimeDetector, RegimeState, apply_hysteresis
from src.regime.vol_detector import VolStructureDetector

logger = logging.getLogger(__name__)

# HMM availability check (hmmlearn is optional)
_hmm_detector_available = False
_HMMDetectorCls: type | None = None
try:
    from src.regime.hmm_detector import HMM_AVAILABLE as _HMM_LIB, HMMDetector as _HMMDetCls

    _hmm_detector_available = _HMM_LIB
    _HMMDetectorCls = _HMMDetCls
except ImportError:
    pass


class EnsembleRegimeDetector:
    """앙상블 레짐 감지기.

    Rule-Based, HMM, Vol-Structure 3종 감지기의 확률을
    가중 평균하여 최종 레짐을 결정합니다.

    Args:
        config: 앙상블 감지기 설정
    """

    def __init__(self, config: EnsembleRegimeDetectorConfig | None = None) -> None:
        self._config = config or EnsembleRegimeDetectorConfig()

        # Rule-Based detector (항상 활성)
        self._rule_detector = RegimeDetector(self._config.rule_based)

        # HMM detector (optional)
        self._hmm_detector: Any = None
        if self._config.hmm is not None and _hmm_detector_available and _HMMDetectorCls is not None:
            self._hmm_detector = _HMMDetectorCls(self._config.hmm)
        elif self._config.hmm is not None and not _hmm_detector_available:
            logger.warning("hmmlearn not available -- HMM detector disabled")

        # Vol-Structure detector (optional)
        self._vol_detector: VolStructureDetector | None = None
        if self._config.vol_structure is not None:
            self._vol_detector = VolStructureDetector(self._config.vol_structure)

        # Incremental hysteresis state per symbol
        self._hold_counters: dict[str, int] = {}
        self._states: dict[str, RegimeState] = {}

    @property
    def config(self) -> EnsembleRegimeDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (Rule-Based의 warmup = 가장 짧음)."""
        return self._rule_detector.warmup_periods

    # ── Vectorized API ──

    def classify_series(self, closes: pd.Series) -> pd.DataFrame:
        """전체 시리즈에서 앙상블 레짐 분류.

        각 감지기의 확률을 가중 평균하고, hysteresis를 적용합니다.

        Args:
            closes: 종가 시리즈

        Returns:
            DataFrame with columns:
                regime_label, p_trending, p_ranging, p_volatile,
                rv_ratio, efficiency_ratio
        """
        n = len(closes)
        cfg = self._config

        # 1. Rule-Based 감지 (항상)
        rule_df = self._rule_detector.classify_series(closes)

        # 2. HMM 감지 (optional)
        hmm_df: pd.DataFrame | None = None
        if self._hmm_detector is not None:
            hmm_df = self._hmm_detector.classify_series(closes)

        # 3. Vol-Structure 감지 (optional)
        vol_df: pd.DataFrame | None = None
        if self._vol_detector is not None:
            vol_df = self._vol_detector.classify_series(closes)

        # 4. Weighted blending (bar-by-bar)
        p_trending = np.full(n, np.nan)
        p_ranging = np.full(n, np.nan)
        p_volatile = np.full(n, np.nan)

        rule_pt = rule_df["p_trending"].to_numpy()
        rule_pr = rule_df["p_ranging"].to_numpy()
        rule_pv = rule_df["p_volatile"].to_numpy()

        hmm_pt = hmm_df["p_trending"].to_numpy() if hmm_df is not None else None
        hmm_pr = hmm_df["p_ranging"].to_numpy() if hmm_df is not None else None
        hmm_pv = hmm_df["p_volatile"].to_numpy() if hmm_df is not None else None

        vol_pt = vol_df["p_trending"].to_numpy() if vol_df is not None else None
        vol_pr = vol_df["p_ranging"].to_numpy() if vol_df is not None else None
        vol_pv = vol_df["p_volatile"].to_numpy() if vol_df is not None else None

        for i in range(n):
            total_weight = 0.0
            bt = 0.0
            br = 0.0
            bv = 0.0

            # Rule-Based
            if not np.isnan(rule_pt[i]):
                w = cfg.weight_rule_based
                bt += w * rule_pt[i]
                br += w * rule_pr[i]
                bv += w * rule_pv[i]
                total_weight += w

            # HMM
            if (
                hmm_pt is not None
                and hmm_pr is not None
                and hmm_pv is not None
                and not np.isnan(hmm_pt[i])
            ):
                w = cfg.weight_hmm
                bt += w * hmm_pt[i]
                br += w * hmm_pr[i]
                bv += w * hmm_pv[i]
                total_weight += w

            # Vol-Structure
            if (
                vol_pt is not None
                and vol_pr is not None
                and vol_pv is not None
                and not np.isnan(vol_pt[i])
            ):
                w = cfg.weight_vol_structure
                bt += w * vol_pt[i]
                br += w * vol_pr[i]
                bv += w * vol_pv[i]
                total_weight += w

            if total_weight > 0:
                p_trending[i] = bt / total_weight
                p_ranging[i] = br / total_weight
                p_volatile[i] = bv / total_weight

        # 5. Hard labels (argmax)
        probs_df = pd.DataFrame(
            {
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
            },
            index=closes.index,
        )
        valid_mask: pd.Series = probs_df.notna().all(axis=1)  # type: ignore[assignment]
        label_map = {
            "p_trending": RegimeLabel.TRENDING,
            "p_ranging": RegimeLabel.RANGING,
            "p_volatile": RegimeLabel.VOLATILE,
        }
        raw_labels = pd.Series(np.nan, index=closes.index, dtype=object)
        if valid_mask.any():  # type: ignore[truthy-bool]
            idx_max: pd.Series = probs_df[valid_mask].idxmax(axis=1)  # type: ignore[assignment]
            raw_labels[valid_mask] = idx_max.map(label_map)

        # 6. Hysteresis (앙상블 이후 한 번만)
        regime_labels = apply_hysteresis(raw_labels, cfg.min_hold_bars)

        return pd.DataFrame(
            {
                "regime_label": regime_labels,
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "rv_ratio": rule_df["rv_ratio"],
                "efficiency_ratio": rule_df["efficiency_ratio"],
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

        # 각 감지기 업데이트
        rule_state = self._rule_detector.update(symbol, close)
        hmm_state = (
            self._hmm_detector.update(symbol, close) if self._hmm_detector is not None else None
        )
        vol_state = (
            self._vol_detector.update(symbol, close) if self._vol_detector is not None else None
        )

        # Rule-Based가 warmup 중이면 전체 None
        if rule_state is None:
            return None

        # Weighted blending
        total_weight = 0.0
        bt = 0.0
        br = 0.0
        bv = 0.0

        # Rule-Based (항상 참여)
        w = cfg.weight_rule_based
        bt += w * rule_state.probabilities["trending"]
        br += w * rule_state.probabilities["ranging"]
        bv += w * rule_state.probabilities["volatile"]
        total_weight += w

        # HMM
        if hmm_state is not None:
            w = cfg.weight_hmm
            bt += w * hmm_state.probabilities["trending"]
            br += w * hmm_state.probabilities["ranging"]
            bv += w * hmm_state.probabilities["volatile"]
            total_weight += w

        # Vol-Structure
        if vol_state is not None:
            w = cfg.weight_vol_structure
            bt += w * vol_state.probabilities["trending"]
            br += w * vol_state.probabilities["ranging"]
            bv += w * vol_state.probabilities["volatile"]
            total_weight += w

        # Normalize
        if total_weight > 0:
            bt /= total_weight
            br /= total_weight
            bv /= total_weight

        # Hard label
        probs = {"trending": bt, "ranging": br, "volatile": bv}
        raw_label = RegimeLabel(max(probs, key=probs.get))  # type: ignore[arg-type]

        # Hysteresis
        if symbol not in self._hold_counters:
            self._hold_counters[symbol] = 0

        prev_state = self._states.get(symbol)
        if prev_state is not None and raw_label != prev_state.label:
            self._hold_counters[symbol] += 1
            if self._hold_counters[symbol] < cfg.min_hold_bars:
                label = prev_state.label
                bars_held = prev_state.bars_held + 1
            else:
                label = raw_label
                bars_held = 1
                self._hold_counters[symbol] = 0
        else:
            label = raw_label
            bars_held = (prev_state.bars_held + 1) if prev_state is not None else 1
            self._hold_counters[symbol] = 0

        state = RegimeState(
            label=label,
            probabilities=probs,
            bars_held=bars_held,
            raw_indicators={
                "rv_ratio": rule_state.raw_indicators.get("rv_ratio", 0.0),
                "er": rule_state.raw_indicators.get("er", 0.0),
            },
        )
        self._states[symbol] = state
        return state

    def get_regime(self, symbol: str) -> RegimeState | None:
        """현재 레짐 상태 조회."""
        return self._states.get(symbol)


def add_ensemble_regime_columns(
    df: pd.DataFrame,
    config: EnsembleRegimeDetectorConfig | None = None,
) -> pd.DataFrame:
    """DataFrame에 앙상블 레짐 컬럼을 추가하는 편의 함수.

    추가되는 컬럼:
        regime_label, p_trending, p_ranging, p_volatile,
        rv_ratio, efficiency_ratio

    Args:
        df: OHLCV DataFrame (close 컬럼 필수)
        config: 앙상블 감지기 설정 (None이면 기본값)

    Returns:
        레짐 컬럼이 추가된 새 DataFrame

    Raises:
        ValueError: close 컬럼 누락 시
    """
    if "close" not in df.columns:
        msg = "DataFrame must contain 'close' column"
        raise ValueError(msg)

    detector = EnsembleRegimeDetector(config)
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    regime_df = detector.classify_series(close_series)
    return pd.concat([df, regime_df], axis=1)
