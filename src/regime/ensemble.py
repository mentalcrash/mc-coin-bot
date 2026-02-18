"""Ensemble Regime Detector.

Rule-Based + HMM + Vol-Structure + MSAR 감지기의 가중 확률 블렌딩으로
TRENDING/RANGING/VOLATILE 레짐을 분류합니다.

Ensemble Methods:
    - weighted_average: 고정 가중치 블렌딩 (기본)
    - meta_learner: LogisticRegression walk-forward stacking

Graceful Degradation:
    - HMM/MSAR 비활성(라이브러리 미설치 또는 config=None) 시 자동 제외
    - Warmup 완료된 감지기만 블렌딩 참여, 가중치 자동 재정규화

Rules Applied:
    - #12 Data Engineering: Vectorization (classify_series)
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from numpy import floating

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

# MSAR availability check (statsmodels is optional)
_msar_detector_available = False
_MSARDetectorCls: type | None = None
try:
    from src.regime.msar_detector import MSAR_AVAILABLE as _MSAR_LIB, MSARDetector as _MSARDetCls

    _msar_detector_available = _MSAR_LIB
    _MSARDetectorCls = _MSARDetCls
except ImportError:
    pass

# sklearn availability check (optional for meta-learner)
_sklearn_available = False
_LogisticRegressionCls: type | None = None
try:
    from sklearn.linear_model import (
        LogisticRegression as _LRCls,  # pyright: ignore[reportMissingImports]
    )

    _LogisticRegressionCls = _LRCls
    _sklearn_available = True
except ImportError:
    pass

# Public alias
SKLEARN_AVAILABLE: bool = _sklearn_available

_PROB_COLS = ("p_trending", "p_ranging", "p_volatile")
_MIN_META_TRAIN_SAMPLES = 30
_MIN_META_CLASSES = 2


def _collect_detector_probs(
    detector_dfs: list[tuple[str, pd.DataFrame, float]],
    n: int,
) -> tuple[
    npt.NDArray[floating[Any]],
    npt.NDArray[floating[Any]],
    npt.NDArray[floating[Any]],
    npt.NDArray[floating[Any]],
]:
    """감지기별 확률을 가중 평균 + confidence 계산.

    Args:
        detector_dfs: (name, df, weight) 튜플 리스트
        n: bar 수

    Returns:
        (p_trending, p_ranging, p_volatile, confidence) numpy arrays
    """
    p_trending = np.full(n, np.nan)
    p_ranging = np.full(n, np.nan)
    p_volatile = np.full(n, np.nan)
    confidence = np.full(n, np.nan)

    # Pre-extract numpy arrays per detector
    det_arrays: list[
        tuple[
            npt.NDArray[floating[Any]],
            npt.NDArray[floating[Any]],
            npt.NDArray[floating[Any]],
            float,
        ]
    ] = []
    for _name, df, weight in detector_dfs:
        det_arrays.append(
            (
                df["p_trending"].to_numpy(),
                df["p_ranging"].to_numpy(),
                df["p_volatile"].to_numpy(),
                weight,
            )
        )

    for i in range(n):
        total_weight = 0.0
        bt = 0.0
        br = 0.0
        bv = 0.0

        for pt_arr, pr_arr, pv_arr, w in det_arrays:
            if not np.isnan(pt_arr[i]):
                bt += w * pt_arr[i]
                br += w * pr_arr[i]
                bv += w * pv_arr[i]
                total_weight += w

        if total_weight > 0:
            p_trending[i] = bt / total_weight
            p_ranging[i] = br / total_weight
            p_volatile[i] = bv / total_weight

    return p_trending, p_ranging, p_volatile, confidence


def _compute_confidence(
    detector_dfs: list[tuple[str, pd.DataFrame, float]],
    final_labels: pd.Series,
    n: int,
) -> npt.NDArray[floating[Any]]:
    """각 bar에서 최종 label에 동의하는 detector 비율 계산.

    각 detector의 argmax(p_trending, p_ranging, p_volatile) label을
    최종 앙상블 label과 비교하여 동의 비율을 산출합니다.

    Args:
        detector_dfs: (name, prob_df, weight) 리스트
        final_labels: 최종 앙상블 regime_label 시리즈
        n: bar 수

    Returns:
        confidence 배열 (0~1)
    """
    confidence = np.full(n, np.nan)
    final_arr = final_labels.to_numpy()

    label_map = {
        "p_trending": RegimeLabel.TRENDING.value,
        "p_ranging": RegimeLabel.RANGING.value,
        "p_volatile": RegimeLabel.VOLATILE.value,
    }

    # Pre-compute argmax label per detector per bar
    det_argmax_labels: list[npt.NDArray[np.object_]] = []
    for _name, df, _w in detector_dfs:
        pt = df["p_trending"].to_numpy()
        pr = df["p_ranging"].to_numpy()
        pv = df["p_volatile"].to_numpy()
        labels_arr = np.full(n, np.nan, dtype=object)
        for i in range(n):
            if not np.isnan(pt[i]):
                pairs = [
                    ("p_trending", float(pt[i])),
                    ("p_ranging", float(pr[i])),
                    ("p_volatile", float(pv[i])),
                ]
                max_col = max(pairs, key=lambda x: x[1])[0]
                labels_arr[i] = label_map[max_col]
        det_argmax_labels.append(labels_arr)

    for i in range(n):
        if pd.isna(final_arr[i]):
            continue
        active = 0
        agree = 0
        for det_labels in det_argmax_labels:
            if pd.notna(det_labels[i]):
                active += 1
                if det_labels[i] == final_arr[i]:
                    agree += 1
        if active > 0:
            confidence[i] = agree / active

    return confidence


def _apply_hard_labels(
    p_trending: npt.NDArray[floating[Any]],
    p_ranging: npt.NDArray[floating[Any]],
    p_volatile: npt.NDArray[floating[Any]],
    index: pd.Index,  # type: ignore[type-arg]
    min_hold_bars: int,
) -> pd.Series:
    """확률 → hard labels + hysteresis."""
    probs_df = pd.DataFrame(
        {"p_trending": p_trending, "p_ranging": p_ranging, "p_volatile": p_volatile},
        index=index,
    )
    valid_mask: pd.Series = probs_df.notna().all(axis=1)  # type: ignore[assignment]
    label_map = {
        "p_trending": RegimeLabel.TRENDING,
        "p_ranging": RegimeLabel.RANGING,
        "p_volatile": RegimeLabel.VOLATILE,
    }
    raw_labels = pd.Series(np.nan, index=index, dtype=object)
    if valid_mask.any():  # type: ignore[truthy-bool]
        idx_max: pd.Series = probs_df[valid_mask].idxmax(axis=1)  # type: ignore[assignment]
        raw_labels[valid_mask] = idx_max.map(label_map)

    return apply_hysteresis(raw_labels, min_hold_bars)


class EnsembleRegimeDetector:
    """앙상블 레짐 감지기.

    Rule-Based, HMM, Vol-Structure, MSAR 감지기의 확률을
    가중 평균 또는 meta-learner로 결합하여 최종 레짐을 결정합니다.

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

        # MSAR detector (optional)
        self._msar_detector: Any = None
        if (
            self._config.msar is not None
            and _msar_detector_available
            and _MSARDetectorCls is not None
        ):
            self._msar_detector = _MSARDetectorCls(self._config.msar)
        elif self._config.msar is not None and not _msar_detector_available:
            logger.warning("statsmodels not available -- MSAR detector disabled")

        # Derivatives detector (optional)
        self._deriv_detector: Any = None
        if self._config.derivatives is not None:
            from src.regime.derivatives_detector import DerivativesDetector

            self._deriv_detector = DerivativesDetector(self._config.derivatives)

        # Meta-learner model (lazy init)
        self._meta_model: Any = None

        # Incremental hysteresis state per symbol
        self._hold_counters: dict[str, int] = {}
        self._pending_labels: dict[str, RegimeLabel | None] = {}
        self._states: dict[str, RegimeState] = {}

    @property
    def config(self) -> EnsembleRegimeDetectorConfig:
        """현재 설정."""
        return self._config

    @property
    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (Rule-Based의 warmup = 가장 짧음)."""
        return self._rule_detector.warmup_periods

    # ── Internal: detector probabilities collection ──

    def _run_all_detectors(
        self,
        closes: pd.Series,
        deriv_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame, float]]]:
        """모든 활성 감지기 실행.

        Args:
            closes: 종가 시리즈
            deriv_df: derivatives 데이터 (DerivativesDetector용, None이면 비활성)

        Returns:
            (rule_df, list of (name, prob_df, weight))
        """
        cfg = self._config

        rule_df = self._rule_detector.classify_series(closes)
        detector_results: list[tuple[str, pd.DataFrame, float]] = [
            ("rule", rule_df, cfg.weight_rule_based),
        ]

        if self._hmm_detector is not None:
            hmm_df = self._hmm_detector.classify_series(closes)
            detector_results.append(("hmm", hmm_df, cfg.weight_hmm))

        if self._vol_detector is not None:
            vol_df = self._vol_detector.classify_series(closes)
            detector_results.append(("vol", vol_df, cfg.weight_vol_structure))

        if self._msar_detector is not None:
            msar_df = self._msar_detector.classify_series(closes)
            detector_results.append(("msar", msar_df, cfg.weight_msar))

        if self._deriv_detector is not None and deriv_df is not None:
            deriv_result_df = self._deriv_detector.classify_series(deriv_df)
            detector_results.append(("derivatives", deriv_result_df, cfg.weight_derivatives))

        return rule_df, detector_results

    # ── Vectorized API ──

    def classify_series(
        self,
        closes: pd.Series,
        deriv_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """전체 시리즈에서 앙상블 레짐 분류.

        각 감지기의 확률을 가중 평균하고, hysteresis를 적용합니다.

        Args:
            closes: 종가 시리즈
            deriv_df: derivatives 데이터 (None이면 derivatives detector 비활성)

        Returns:
            DataFrame with columns:
                regime_label, p_trending, p_ranging, p_volatile,
                rv_ratio, efficiency_ratio, confidence
        """
        cfg = self._config
        rule_df, detector_results = self._run_all_detectors(closes, deriv_df)

        if cfg.ensemble_method == "meta_learner":
            return self._classify_meta_learner(closes, rule_df, detector_results)

        return self._classify_weighted_average(closes, rule_df, detector_results)

    def _classify_weighted_average(
        self,
        closes: pd.Series,
        rule_df: pd.DataFrame,
        detector_results: list[tuple[str, pd.DataFrame, float]],
    ) -> pd.DataFrame:
        """Weighted average 앙상블."""
        n = len(closes)
        cfg = self._config

        p_trending, p_ranging, p_volatile, _unused_conf = _collect_detector_probs(
            detector_results,
            n,
        )

        regime_labels = _apply_hard_labels(
            p_trending,
            p_ranging,
            p_volatile,
            closes.index,
            cfg.min_hold_bars,
        )

        # Confidence: detector agreement with final ensemble label
        confidence = _compute_confidence(detector_results, regime_labels, n)

        return pd.DataFrame(
            {
                "regime_label": regime_labels,
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "rv_ratio": rule_df["rv_ratio"],
                "efficiency_ratio": rule_df["efficiency_ratio"],
                "confidence": confidence,
            },
            index=closes.index,
        )

    @staticmethod
    def _train_meta_model(
        features_arr: npt.NDArray[floating[Any]],
        labels: pd.Series,
        label_encode: dict[str, int],
        train_start: int,
        train_end: int,
        regularization: float,
    ) -> Any | None:
        """Meta-learner 학습 (walk-forward 단일 윈도우)."""
        if _LogisticRegressionCls is None:
            return None

        x_train = features_arr[train_start:train_end]
        y_labels = labels.iloc[train_start:train_end]

        valid = ~np.isnan(x_train).any(axis=1) & y_labels.notna().to_numpy()
        x_valid = x_train[valid]
        y_valid = y_labels[valid].map(label_encode).to_numpy().astype(int)

        if len(x_valid) < _MIN_META_TRAIN_SAMPLES or len(np.unique(y_valid)) < _MIN_META_CLASSES:
            return None

        try:
            lr = _LogisticRegressionCls(
                C=regularization,
                max_iter=500,
                solver="lbfgs",
                random_state=42,
            )
            lr.fit(x_valid, y_valid)
        except Exception:
            return None
        else:
            return lr

    @staticmethod
    def _predict_meta_bar(
        model: Any,
        features: npt.NDArray[floating[Any]],
        label_decode: dict[int, RegimeLabel],
    ) -> tuple[float, float, float] | None:
        """Meta-learner 단일 bar 예측."""
        if np.isnan(features).any():
            return None

        try:
            proba = model.predict_proba(features.reshape(1, -1))[0]
            pt, pr, pv = 0.0, 0.0, 0.0
            for cls_idx, cls_val in enumerate(model.classes_):
                decoded = label_decode.get(int(cls_val))
                if decoded == RegimeLabel.TRENDING:
                    pt = float(proba[cls_idx])
                elif decoded == RegimeLabel.RANGING:
                    pr = float(proba[cls_idx])
                elif decoded == RegimeLabel.VOLATILE:
                    pv = float(proba[cls_idx])
        except Exception:
            return None
        else:
            return pt, pr, pv

    def _classify_meta_learner(
        self,
        closes: pd.Series,
        rule_df: pd.DataFrame,
        detector_results: list[tuple[str, pd.DataFrame, float]],
    ) -> pd.DataFrame:
        """Meta-learner stacking 앙상블.

        walk-forward 방식으로 LogisticRegression을 학습하여
        감지기 확률 → 레짐 라벨을 예측합니다.
        """
        if not _sklearn_available or _LogisticRegressionCls is None:
            logger.warning("sklearn not available -- falling back to weighted_average")
            return self._classify_weighted_average(closes, rule_df, detector_results)

        cfg = self._config
        ml_cfg = cfg.meta_learner
        if ml_cfg is None:
            return self._classify_weighted_average(closes, rule_df, detector_results)

        n = len(closes)

        # 1. Stack detector probabilities as features
        feature_dfs: list[pd.DataFrame] = []
        for name, df, _weight in detector_results:
            cols = ["p_trending", "p_ranging", "p_volatile"]
            col_map = {
                "p_trending": f"{name}_pt",
                "p_ranging": f"{name}_pr",
                "p_volatile": f"{name}_pv",
            }
            sub_df: pd.DataFrame = df[cols]  # type: ignore[assignment]
            renamed = sub_df.rename(columns=col_map)
            feature_dfs.append(renamed)

        features_df = pd.concat(feature_dfs, axis=1)
        features_arr = features_df.to_numpy()

        # 2. Forward return labels (사후)
        fwd_returns = closes.pct_change(ml_cfg.forward_return_window).shift(
            -ml_cfg.forward_return_window
        )
        labels = pd.Series(np.nan, index=closes.index, dtype=object)
        abs_fwd = fwd_returns.abs()
        labels[abs_fwd <= ml_cfg.trending_threshold] = "ranging"
        labels[(abs_fwd > ml_cfg.trending_threshold) & (abs_fwd <= ml_cfg.volatile_threshold)] = (
            "trending"
        )
        labels[abs_fwd > ml_cfg.volatile_threshold] = "volatile"

        label_encode = {"trending": 0, "ranging": 1, "volatile": 2}
        label_decode = {0: RegimeLabel.TRENDING, 1: RegimeLabel.RANGING, 2: RegimeLabel.VOLATILE}

        # 3. Walk-forward training + prediction
        p_trending = np.full(n, np.nan)
        p_ranging = np.full(n, np.nan)
        p_volatile = np.full(n, np.nan)

        model: Any = None

        for i in range(n):
            # Retrain at intervals
            if i >= ml_cfg.train_window + ml_cfg.forward_return_window and (
                model is None or i % ml_cfg.retrain_interval == 0
            ):
                train_end = i - ml_cfg.forward_return_window
                train_start = max(0, train_end - ml_cfg.train_window)
                new_model = self._train_meta_model(
                    features_arr,
                    labels,
                    label_encode,
                    train_start,
                    train_end,
                    ml_cfg.regularization,
                )
                if new_model is not None:
                    model = new_model

            if model is not None:
                prediction = self._predict_meta_bar(model, features_arr[i], label_decode)
                if prediction is not None:
                    p_trending[i], p_ranging[i], p_volatile[i] = prediction

        regime_labels = _apply_hard_labels(
            p_trending,
            p_ranging,
            p_volatile,
            closes.index,
            cfg.min_hold_bars,
        )

        # Meta-learner confidence: detector agreement
        confidence = _compute_confidence(detector_results, regime_labels, n)

        return pd.DataFrame(
            {
                "regime_label": regime_labels,
                "p_trending": p_trending,
                "p_ranging": p_ranging,
                "p_volatile": p_volatile,
                "rv_ratio": rule_df["rv_ratio"],
                "efficiency_ratio": rule_df["efficiency_ratio"],
                "confidence": confidence,
            },
            index=closes.index,
        )

    # ── Incremental API ──

    def _blend_detector_states(
        self,
        rule_state: RegimeState,
        hmm_state: RegimeState | None,
        vol_state: RegimeState | None,
        msar_state: RegimeState | None,
        deriv_state: RegimeState | None = None,
    ) -> tuple[dict[str, float], float]:
        """감지기 상태를 가중 블렌딩하여 확률 dict + confidence 반환.

        Returns:
            (probabilities dict, confidence)
        """
        cfg = self._config
        total_weight = 0.0
        bt, br, bv = 0.0, 0.0, 0.0

        all_states: list[tuple[RegimeState | None, float]] = [
            (rule_state, cfg.weight_rule_based),
            (hmm_state, cfg.weight_hmm),
            (vol_state, cfg.weight_vol_structure),
            (msar_state, cfg.weight_msar),
            (deriv_state, cfg.weight_derivatives),
        ]

        for state, weight in all_states:
            if state is not None:
                bt += weight * state.probabilities["trending"]
                br += weight * state.probabilities["ranging"]
                bv += weight * state.probabilities["volatile"]
                total_weight += weight

        if total_weight > 0:
            bt /= total_weight
            br /= total_weight
            bv /= total_weight

        probs = {"trending": bt, "ranging": br, "volatile": bv}

        # Confidence: argmax label 대비 동의 비율
        blended_label = max(probs, key=probs.get)  # type: ignore[arg-type]
        active_count = 0
        agree_count = 0
        for state, _weight in all_states:
            if state is not None:
                active_count += 1
                det_label = max(state.probabilities, key=state.probabilities.get)  # type: ignore[arg-type]
                if det_label == blended_label:
                    agree_count += 1
        confidence = agree_count / active_count if active_count > 0 else 1.0

        return probs, confidence

    def update(
        self,
        symbol: str,
        close: float,
        derivatives: dict[str, float] | None = None,
    ) -> RegimeState | None:
        """Bar 단위 incremental 업데이트.

        Note: meta_learner 모드에서는 forward return이 필요하므로
        weighted_average로 자동 폴백합니다.

        Args:
            symbol: 거래 심볼
            close: 현재 종가
            derivatives: derivatives 데이터 (None이면 derivatives detector 스킵)

        Returns:
            RegimeState 또는 warmup 중 None
        """
        cfg = self._config

        if cfg.ensemble_method == "meta_learner":
            msg = "meta_learner does not support incremental update -- falling back to weighted_average"
            logger.warning(msg)

        # 각 감지기 업데이트
        rule_state = self._rule_detector.update(symbol, close)
        hmm_state = (
            self._hmm_detector.update(symbol, close) if self._hmm_detector is not None else None
        )
        vol_state = (
            self._vol_detector.update(symbol, close) if self._vol_detector is not None else None
        )
        msar_state = (
            self._msar_detector.update(symbol, close) if self._msar_detector is not None else None
        )
        deriv_state: RegimeState | None = None
        if self._deriv_detector is not None and derivatives is not None:
            deriv_state = self._deriv_detector.update(symbol, **derivatives)

        # Rule-Based가 warmup 중이면 전체 None
        if rule_state is None:
            return None

        # Weighted blending + confidence
        probs, confidence = self._blend_detector_states(
            rule_state, hmm_state, vol_state, msar_state, deriv_state
        )
        raw_label = RegimeLabel(max(probs, key=probs.get))  # type: ignore[arg-type]

        # Hysteresis (matches vectorized apply_hysteresis: same pending label required)
        if symbol not in self._hold_counters:
            self._hold_counters[symbol] = 0
            self._pending_labels[symbol] = None

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
            probabilities=probs,
            bars_held=bars_held,
            raw_indicators={
                "rv_ratio": rule_state.raw_indicators.get("rv_ratio", 0.0),
                "er": rule_state.raw_indicators.get("er", 0.0),
            },
            confidence=confidence,
        )
        self._states[symbol] = state
        return state

    def get_regime(self, symbol: str) -> RegimeState | None:
        """현재 레짐 상태 조회."""
        return self._states.get(symbol)

    @property
    def has_derivatives_detector(self) -> bool:
        """Derivatives detector가 활성인지 여부."""
        return self._deriv_detector is not None

    def get_cascade_risk(self, symbol: str) -> float:
        """Derivatives detector의 cascade risk 조회.

        Returns:
            cascade risk (0~1). detector 없으면 0.0
        """
        if self._deriv_detector is None:
            return 0.0
        return self._deriv_detector.get_cascade_risk(symbol)

    def get_cascade_risk_series(self, deriv_df: pd.DataFrame) -> pd.Series:
        """Derivatives detector의 cascade risk 시리즈 반환.

        Returns:
            cascade risk 시리즈. detector 없으면 0.0 시리즈
        """
        if self._deriv_detector is None:
            return pd.Series(0.0, index=deriv_df.index)
        return self._deriv_detector.get_cascade_risk_series(deriv_df)


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
