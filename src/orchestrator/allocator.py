"""Capital Allocator — 멀티 전략 자본 배분 엔진.

4가지 배분 알고리즘(EW, InvVol, Risk Parity, Adaptive Kelly)과
Lifecycle 상태별 가중치 제한(clamp)을 제공합니다.

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - #11 Pydantic Modeling: OrchestratorConfig 활용
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from numpy import floating
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from src.orchestrator.config import OrchestratorConfig, PodConfig

from src.orchestrator.models import AllocationMethod, LifecycleState

# ── Constants ─────────────────────────────────────────────────────

_MIN_WEIGHT = 1e-8
_COV_REGULARIZATION = 1e-6
_OPTIMIZER_FTOL = 1e-12
_OPTIMIZER_MAXITER = 1000
_WARNING_SCALE = 0.5
_ANNUALIZATION_FACTOR = 365.0
_NAN_MAJORITY_THRESHOLD = 0.5
_CONDITION_NUMBER_THRESHOLD = 1e10


# ── Spinu Objective (Module-level pure functions) ─────────────────


def _spinu_objective(
    w: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    budgets: npt.NDArray[np.float64],
) -> float:
    """Spinu(2013) Risk Parity convex objective.

    f(w) = 0.5 * w^T cov w - sum(b_i log(w_i))
    """
    portfolio_var: float = 0.5 * float(w @ cov @ w)
    log_barrier: float = float(np.sum(budgets * np.log(w)))
    return portfolio_var - log_barrier


def _spinu_gradient(
    w: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    budgets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gradient of Spinu objective: grad = cov @ w - b/w."""
    result: npt.NDArray[np.float64] = cov @ w - budgets / w
    return result


# ── Covariance Utilities ──────────────────────────────────────────


def _is_cov_valid(cov: npt.NDArray[np.float64]) -> bool:
    """공분산 행렬 유효성: finite, 양수 대각, 대칭."""
    if not np.all(np.isfinite(cov)):
        return False
    if not np.all(np.diag(cov) > 0):
        return False
    return bool(np.allclose(cov, cov.T))


def _regularize_cov(cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Ridge 정규화: cov + epsilon * I."""
    result: npt.NDArray[np.float64] = cov + _COV_REGULARIZATION * np.eye(cov.shape[0])
    return result


def _compute_cov_matrix(pod_returns: pd.DataFrame, lookback: int) -> npt.NDArray[np.float64]:
    """수익률 DataFrame에서 공분산 행렬 계산 (lookback 적용, 연간화)."""
    tail = pod_returns.tail(lookback)
    cov: npt.NDArray[np.float64] = (
        tail.cov().to_numpy() * _ANNUALIZATION_FACTOR  # type: ignore[union-attr]
    )
    # NaN → 0
    cov = np.where(np.isfinite(cov), cov, 0.0)
    # 대각 원소가 0이면 최소값 설정
    diag: npt.NDArray[np.float64] = np.diag(cov).copy()
    diag[diag <= 0] = _MIN_WEIGHT
    np.fill_diagonal(cov, diag)
    # 대칭 보장
    cov = (cov + cov.T) / 2.0
    return cov


def _compute_confidence(live_days: int, ramp_days: int) -> float:
    """Linear ramp: 0.0 ~ 1.0."""
    if ramp_days <= 0:
        return 1.0
    return min(live_days / ramp_days, 1.0)


# ── CapitalAllocator ──────────────────────────────────────────────


class CapitalAllocator:
    """멀티 전략 자본 배분 엔진.

    4가지 배분 알고리즘과 Lifecycle 상태별 가중치 제한을 제공합니다.

    Args:
        config: OrchestratorConfig (pods, allocation_method 등)
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config
        self._pod_configs: dict[str, PodConfig] = {p.pod_id: p for p in config.pods}
        self._dispatch: dict[AllocationMethod, Callable[..., dict[str, float]]] = {
            AllocationMethod.EQUAL_WEIGHT: self._equal_weight,
            AllocationMethod.INVERSE_VOLATILITY: self._inverse_volatility,
            AllocationMethod.RISK_PARITY: self._risk_parity,
            AllocationMethod.ADAPTIVE_KELLY: self._adaptive_kelly,
        }

    # ── Public API ────────────────────────────────────────────────

    def compute_weights(
        self,
        pod_returns: pd.DataFrame,
        pod_states: dict[str, LifecycleState],
        lookback: int = 90,
        pod_live_days: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """자본 배분 가중치 계산.

        Args:
            pod_returns: columns=pod_ids, index=dates 일별 수익률
            pod_states: pod_id -> LifecycleState 매핑
            lookback: 수익률 lookback 기간 (일)
            pod_live_days: pod_id -> 실제 live_days 매핑 (None이면 heuristic)

        Returns:
            pod_id -> fraction 매핑, sum <= 1.0, 모든 값 >= 0.0
        """
        all_pod_ids = list(self._pod_configs.keys())

        # 1. RETIRED 필터링
        active_ids = [pid for pid in all_pod_ids if pod_states.get(pid) != LifecycleState.RETIRED]

        # 전부 RETIRED
        if not active_ids:
            return dict.fromkeys(all_pod_ids, 0.0)

        # 2. 단일 Pod
        if len(active_ids) == 1:
            pid = active_ids[0]
            w = self._pod_configs[pid].initial_fraction
            result = dict.fromkeys(all_pod_ids, 0.0)
            result[pid] = w
            return self._apply_lifecycle_clamps(result, pod_states)

        # 3. 수익률 데이터 필터링
        available = [pid for pid in active_ids if pid in pod_returns.columns]
        if not available:
            return dict.fromkeys(all_pod_ids, 0.0)

        returns_tail: pd.DataFrame = pod_returns[available].tail(lookback)  # type: ignore[assignment]
        # NaN 과반 컬럼 drop
        nan_ratio: Any = returns_tail.isna().mean()
        valid_cols = [c for c in available if float(nan_ratio[c]) < _NAN_MAJORITY_THRESHOLD]
        if not valid_cols:
            return dict.fromkeys(all_pod_ids, 0.0)

        returns_clean: pd.DataFrame = returns_tail[valid_cols].fillna(0.0)  # type: ignore[assignment]

        # 4. 배분 알고리즘 호출
        method = self._config.allocation_method
        raw_weights = self._dispatch[method](returns_clean, pod_states, pod_live_days)

        # 5. 비활성 pod는 0.0
        result = dict.fromkeys(all_pod_ids, 0.0)
        for pid, w in raw_weights.items():
            result[pid] = w

        # 6. Lifecycle clamps
        result = self._apply_lifecycle_clamps(result, pod_states)

        # 7. 합계 > 1.0이면 비례 축소
        total = sum(result.values())
        if total > 1.0:
            for pid in result:
                result[pid] /= total

        return result

    # ── Allocation Algorithms ─────────────────────────────────────

    def _equal_weight(
        self,
        returns: pd.DataFrame,
        pod_states: dict[str, LifecycleState],
        pod_live_days: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """균등 배분."""
        n = len(returns.columns)
        w = 1.0 / n
        return {str(col): w for col in returns.columns}

    def _inverse_volatility(
        self,
        returns: pd.DataFrame,
        pod_states: dict[str, LifecycleState],
        pod_live_days: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """변동성 역비례 배분."""
        vol: Any = returns.std()
        if vol.isna().all() or (vol == 0).all():
            return self._equal_weight(returns, pod_states)

        inv_vol: Any = 1.0 / vol.clip(lower=_MIN_WEIGHT)
        if inv_vol.isna().any():
            return self._equal_weight(returns, pod_states)

        weights: Any = inv_vol / inv_vol.sum()
        return {str(col): float(weights[col]) for col in returns.columns}

    def _risk_parity(
        self,
        returns: pd.DataFrame,
        pod_states: dict[str, LifecycleState],
        pod_live_days: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Spinu(2013) Risk Parity via L-BFGS-B."""
        cov = _compute_cov_matrix(returns, len(returns))
        return self._risk_parity_weights(cov, list(returns.columns), returns, pod_states)

    def _risk_parity_weights(
        self,
        cov: npt.NDArray[np.float64],
        col_names: list[Any],
        returns: pd.DataFrame,
        pod_states: dict[str, LifecycleState],
    ) -> dict[str, float]:
        """Risk Parity 최적화 실행."""
        n = len(col_names)
        budgets = np.ones(n) / n

        # 공분산 유효성
        if not _is_cov_valid(cov):
            cov = _regularize_cov(cov)
            if not _is_cov_valid(cov):
                logger.warning("Covariance invalid after regularization, falling back to InvVol")
                return self._inverse_volatility(returns, pod_states)

        # Condition number 체크
        cond = float(np.linalg.cond(cov))
        if cond > _CONDITION_NUMBER_THRESHOLD:
            cov = _regularize_cov(cov)

        w0 = np.ones(n) / n
        bounds = [(_MIN_WEIGHT, None) for _ in range(n)]

        opt_result = minimize(
            _spinu_objective,
            w0,
            args=(cov, budgets),
            jac=_spinu_gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": _OPTIMIZER_FTOL, "maxiter": _OPTIMIZER_MAXITER},
        )

        if not opt_result.success:
            logger.warning(
                "Risk Parity optimizer failed: {}, falling back to InvVol", opt_result.message
            )
            return self._inverse_volatility(returns, pod_states)

        w: npt.NDArray[np.float64] = np.maximum(opt_result.x, 0.0)
        total = float(w.sum())
        if total < _MIN_WEIGHT:
            return self._inverse_volatility(returns, pod_states)
        w = w / total

        return {str(col_names[i]): float(w[i]) for i in range(n)}

    def _adaptive_kelly(
        self,
        returns: pd.DataFrame,
        pod_states: dict[str, LifecycleState],
        pod_live_days: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Adaptive Kelly: Risk Parity + Kelly blend."""
        n = len(returns.columns)
        col_names = list(returns.columns)
        cov = _compute_cov_matrix(returns, len(returns))

        # Risk Parity base
        rp_weights = self._risk_parity_weights(cov, col_names, returns, pod_states)
        rp_arr = np.array([rp_weights[str(c)] for c in col_names])

        # Kelly optimal
        mu: npt.NDArray[np.float64] = returns.mean().to_numpy() * _ANNUALIZATION_FACTOR  # type: ignore[assignment]

        # 모든 기대수익 <= 0 -> 순수 RP
        if np.all(mu <= 0):
            return rp_weights

        kelly_weights = self._kelly_optimal(mu, cov)
        if kelly_weights is None:
            return rp_weights

        # Fractional Kelly (long-only)
        f_frac: npt.NDArray[floating[Any]] = self._config.kelly_fraction * np.clip(
            kelly_weights, 0, None
        )
        f_sum = float(f_frac.sum())
        if f_sum < _MIN_WEIGHT:
            return rp_weights
        f_norm = f_frac / f_sum

        # Confidence ramp
        avg_live_days = self._avg_live_days(col_names, pod_states, pod_live_days)
        confidence = _compute_confidence(avg_live_days, self._config.kelly_confidence_ramp)
        alpha = confidence

        # Blend
        blended: npt.NDArray[np.float64] = (1.0 - alpha) * rp_arr + alpha * f_norm
        total = float(blended.sum())
        if total < _MIN_WEIGHT:
            return rp_weights
        blended = blended / total

        return {str(col_names[i]): float(blended[i]) for i in range(n)}

    def _kelly_optimal(
        self,
        mu: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64],
    ) -> npt.NDArray[floating[Any]] | None:
        """Full Kelly: f* = cov^{-1} mu."""
        try:
            result: npt.NDArray[floating[Any]] = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            logger.warning("Kelly solve failed (singular matrix)")
            return None
        else:
            return result

    def _avg_live_days(
        self,
        pod_ids: list[Any],
        pod_states: dict[str, LifecycleState],
        pod_live_days: dict[str, int] | None = None,
    ) -> int:
        """평균 live_days 계산.

        pod_live_days가 주어지면 실제 값을 사용하고,
        없으면 state 기반 heuristic으로 추정합니다.
        """
        if pod_live_days is not None:
            total = sum(pod_live_days.get(str(pid), 0) for pid in pod_ids)
            return total // max(len(pod_ids), 1)

        days_by_state = {
            LifecycleState.INCUBATION: 30,
            LifecycleState.PRODUCTION: 180,
            LifecycleState.WARNING: 120,
            LifecycleState.PROBATION: 60,
            LifecycleState.RETIRED: 0,
        }
        total = sum(
            days_by_state.get(pod_states.get(str(pid), LifecycleState.INCUBATION), 0)
            for pid in pod_ids
        )
        return total // max(len(pod_ids), 1)

    # ── Guard Layer ───────────────────────────────────────────────

    def _apply_lifecycle_clamps(
        self,
        weights: dict[str, float],
        pod_states: dict[str, LifecycleState],
    ) -> dict[str, float]:
        """Lifecycle 상태별 가중치 제한.

        RETIRED -> 0.0
        INCUBATION -> min(weight, initial_fraction)
        PRODUCTION -> clip(weight, min_fraction, max_fraction)
        WARNING -> weight * 0.5, then clip(min, max)
        PROBATION -> min_fraction 고정
        """
        result: dict[str, float] = {}
        for pid, w in weights.items():
            state = pod_states.get(pid, LifecycleState.INCUBATION)
            cfg = self._pod_configs.get(pid)
            if cfg is None:
                result[pid] = 0.0
                continue

            clamped = self._clamp_single(w, state, cfg)
            result[pid] = clamped

        # Clamp 후 합계 > 1.0이면 비례 축소
        total = sum(result.values())
        if total > 1.0:
            for pid in result:
                result[pid] /= total

        return result

    def _clamp_single(
        self,
        weight: float,
        state: LifecycleState,
        cfg: PodConfig,
    ) -> float:
        """단일 pod 가중치 clamp."""
        if state == LifecycleState.RETIRED:
            return 0.0
        if state == LifecycleState.INCUBATION:
            return min(weight, cfg.initial_fraction)
        if state == LifecycleState.PRODUCTION:
            return max(cfg.min_fraction, min(weight, cfg.max_fraction))
        if state == LifecycleState.WARNING:
            scaled = weight * _WARNING_SCALE
            return max(cfg.min_fraction, min(scaled, cfg.max_fraction))
        if state == LifecycleState.PROBATION:
            return cfg.min_fraction
        return weight
