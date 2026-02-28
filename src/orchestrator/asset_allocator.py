"""Intra-Pod Asset Allocator — Pod 내 에셋 간 비중 배분 엔진.

4가지 배분 알고리즘(EW, InvVol, Risk Parity, Signal Weighted)과
min/max clamp, 주기적 리밸런싱을 제공합니다.

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - #11 Pydantic Modeling: frozen=True, Field validators
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.orchestrator.models import AssetAllocationMethod

# ── Constants ─────────────────────────────────────────────────────

_MIN_VOL = 1e-8
_COV_REGULARIZATION = 1e-6
_CLAMP_MAX_ITERATIONS = 5
_CLAMP_TOLERANCE = 1e-10
_OPTIMIZER_FTOL = 1e-12
_OPTIMIZER_MAXITER = 1000
_MIN_SAMPLES = 2


# ── Risk Math Utilities ──────────────────────────────────────────


def _spinu_objective(
    w: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    budgets: npt.NDArray[np.float64],
) -> float:
    """Spinu(2013) Risk Parity convex objective."""
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


# ── Config ────────────────────────────────────────────────────────


class AssetAllocationConfig(BaseModel):
    """Pod 내 에셋 배분 설정.

    Attributes:
        method: 배분 알고리즘
        vol_lookback: 변동성 계산 윈도우 (bars)
        rebalance_bars: 비중 재계산 주기 (bars)
        min_weight: 에셋당 최소 비중
        max_weight: 에셋당 최대 비중
        mom_lookback: Dual Momentum lookback (bars)
        top_pct: Dual Momentum long 할 상위 비율
        abs_mom_threshold: 절대 모멘텀 게이트 임계값
        exposure_floor: 최소 exposure (현금 대피 시 하한)
    """

    model_config = ConfigDict(frozen=True)

    method: AssetAllocationMethod = Field(
        default=AssetAllocationMethod.EQUAL_WEIGHT,
        description="배분 알고리즘",
    )
    vol_lookback: int = Field(
        default=60,
        ge=10,
        description="변동성 계산 윈도우 (bars)",
    )
    rebalance_bars: int = Field(
        default=5,
        ge=1,
        description="비중 재계산 주기 (bars)",
    )
    min_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="에셋당 최소 비중",
    )
    max_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="에셋당 최대 비중",
    )

    # Dual Momentum 전용 필드 (다른 method에서는 무시됨)
    mom_lookback: int = Field(
        default=42,
        ge=5,
        description="Momentum 계산 lookback (bars)",
    )
    top_pct: float = Field(
        default=0.50,
        gt=0.0,
        le=1.0,
        description="Long 할 상위 비율",
    )
    abs_mom_threshold: float = Field(
        default=0.0,
        description="절대 모멘텀 게이트 임계값",
    )
    exposure_floor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최소 exposure (현금 대피 시 하한)",
    )

    @model_validator(mode="after")
    def validate_weight_bounds(self) -> Self:
        """min_weight <= max_weight 검증."""
        if self.min_weight > self.max_weight:
            msg = f"min_weight ({self.min_weight}) cannot exceed max_weight ({self.max_weight})"
            raise ValueError(msg)
        return self


# ── IntraPodAllocator ─────────────────────────────────────────────


class IntraPodAllocator:
    """Pod 내 에셋 간 비중 계산기.

    bar 단위로 호출되며, rebalance_bars 주기마다 비중을 재계산합니다.

    Args:
        config: 배분 설정
        symbols: 에셋 심볼 목록
    """

    def __init__(
        self,
        config: AssetAllocationConfig,
        symbols: tuple[str, ...],
    ) -> None:
        self._config = config
        self._symbols = symbols
        n = len(symbols)
        self._weights: dict[str, float] = dict.fromkeys(symbols, 1.0 / n)
        self._bar_count: int = 0
        self._exposure: float = 1.0

    @property
    def weights(self) -> dict[str, float]:
        """현재 에셋 비중."""
        return dict(self._weights)

    @property
    def exposure(self) -> float:
        """현재 전체 exposure (0.0~1.0). 기존 method는 항상 1.0."""
        return self._exposure

    @property
    def bar_count(self) -> int:
        """누적 bar 수."""
        return self._bar_count

    # ── Public API ────────────────────────────────────────────────

    def on_bar(
        self,
        returns: dict[str, list[float]],
        strengths: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Bar 도착 시 호출. rebalance_bars 주기마다 비중 재계산.

        Args:
            returns: 심볼별 과거 수익률 리스트
            strengths: 심볼별 최신 signal strength (signal_weighted용)

        Returns:
            심볼별 비중 (합계 1.0)
        """
        self._bar_count += 1

        if self._bar_count % self._config.rebalance_bars != 0:
            return dict(self._weights)

        method = self._config.method

        if method == AssetAllocationMethod.DUAL_MOMENTUM:
            self._weights, self._exposure = self._dual_momentum(returns)
            return dict(self._weights)

        if method == AssetAllocationMethod.EQUAL_WEIGHT:
            raw = self._equal_weight()
        elif method == AssetAllocationMethod.INVERSE_VOLATILITY:
            raw = self._inverse_vol(returns)
        elif method == AssetAllocationMethod.RISK_PARITY:
            raw = self._risk_parity(returns)
        elif method == AssetAllocationMethod.SIGNAL_WEIGHTED:
            raw = self._signal_weighted(strengths)
        else:  # pragma: no cover
            raw = self._equal_weight()

        self._weights = _clamp_and_normalize(
            raw,
            self._config.min_weight,
            self._config.max_weight,
        )
        return dict(self._weights)

    # ── Dynamic Symbol Management ────────────────────────────────

    def add_symbol(self, symbol: str) -> None:
        """신규 심볼 추가 → EW 재분배.

        Args:
            symbol: 추가할 심볼
        """
        if symbol in self._weights:
            return
        self._symbols = (*self._symbols, symbol)
        n = len(self._symbols)
        self._weights = dict.fromkeys(self._symbols, 1.0 / n)

    # ── Algorithms ────────────────────────────────────────────────

    def _equal_weight(self) -> dict[str, float]:
        """균등 배분."""
        n = len(self._symbols)
        w = 1.0 / n
        return dict.fromkeys(self._symbols, w)

    def _inverse_vol(
        self,
        returns: dict[str, list[float]],
    ) -> dict[str, float]:
        """변동성 역비례 배분. 데이터 부족 시 EW fallback."""
        lookback = self._config.vol_lookback
        vols: dict[str, float] = {}

        for s in self._symbols:
            r = returns.get(s, [])
            tail = r[-lookback:] if len(r) >= _MIN_SAMPLES else r
            if len(tail) < _MIN_SAMPLES:
                logger.debug("IntraPod IV: insufficient data for {}, fallback to EW", s)
                return self._equal_weight()
            vols[s] = max(float(np.std(tail, ddof=1)), _MIN_VOL)

        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total = sum(inv_vols.values())
        return {s: iv / total for s, iv in inv_vols.items()}

    def _risk_parity(
        self,
        returns: dict[str, list[float]],
    ) -> dict[str, float]:
        """Spinu(2013) Risk Parity. 실패 시 inverse_vol fallback."""
        lookback = self._config.vol_lookback
        symbols = list(self._symbols)
        n = len(symbols)

        # 수익률 행렬 구성
        arrays: list[npt.NDArray[np.float64]] = []
        for s in symbols:
            r = returns.get(s, [])
            tail = r[-lookback:]
            if len(tail) < _MIN_SAMPLES:
                return self._inverse_vol(returns)
            arrays.append(np.array(tail, dtype=np.float64))

        # 최소 길이로 맞춤
        min_len = min(len(a) for a in arrays)
        if min_len < _MIN_SAMPLES:
            return self._inverse_vol(returns)

        mat = np.column_stack([a[-min_len:] for a in arrays])
        cov: npt.NDArray[np.float64] = np.asarray(
            np.cov(mat, rowvar=False, ddof=1), dtype=np.float64
        )

        # 1D인 경우 2D로 변환
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)

        if not _is_cov_valid(cov):
            cov = _regularize_cov(cov)
            if not _is_cov_valid(cov):
                logger.warning("IntraPod RP: cov invalid, fallback to IV")
                return self._inverse_vol(returns)

        budgets = np.ones(n) / n
        w0 = np.ones(n) / n
        bounds = [(_MIN_VOL, None) for _ in range(n)]

        from scipy.optimize import minimize

        opt = minimize(
            _spinu_objective,
            w0,
            args=(cov, budgets),
            jac=_spinu_gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": _OPTIMIZER_FTOL, "maxiter": _OPTIMIZER_MAXITER},
        )

        if not opt.success:
            logger.warning("IntraPod RP: optimizer failed, fallback to IV")
            return self._inverse_vol(returns)

        w: npt.NDArray[np.float64] = np.maximum(opt.x, 0.0)
        total = float(w.sum())
        if total < _MIN_VOL:
            return self._inverse_vol(returns)
        w = w / total

        return {symbols[i]: float(w[i]) for i in range(n)}

    def _signal_weighted(
        self,
        strengths: dict[str, float] | None,
    ) -> dict[str, float]:
        """신호 강도 기반 배분. 전체 0이면 EW fallback."""
        if not strengths:
            return self._equal_weight()

        abs_strengths = {s: abs(strengths.get(s, 0.0)) for s in self._symbols}
        total = sum(abs_strengths.values())

        if total < _MIN_VOL:
            return self._equal_weight()

        return {s: v / total for s, v in abs_strengths.items()}

    def _dual_momentum(
        self,
        returns: dict[str, list[float]],
    ) -> tuple[dict[str, float], float]:
        """Dual Momentum 배분: 절대 게이트 + 상대 ranking.

        1. 심볼별 cumulative return 계산 (mom_lookback)
        2. 데이터 부족 → (EW, 1.0) fallback
        3. Absolute Gate: median(mom) < threshold → exposure 축소
        4. Cross-sectional Ranking: top_pct 선택 → EW 배분

        Returns:
            (weights, exposure) 튜플
        """
        lookback = self._config.mom_lookback
        n = len(self._symbols)
        ew_weights = dict.fromkeys(self._symbols, 1.0 / n)

        # 1. 심볼별 momentum 계산
        momentums: dict[str, float] = {}
        for s in self._symbols:
            r = returns.get(s, [])
            if len(r) < lookback:
                logger.debug("DualMom: insufficient data for {}, fallback to EW", s)
                return ew_weights, 1.0
            tail = r[-lookback:]
            # Cumulative return: product of (1+r) - 1
            cum_ret = float(np.prod(np.array(tail) + 1.0)) - 1.0
            momentums[s] = cum_ret

        # 2. Absolute Gate: median momentum
        mom_values = list(momentums.values())
        median_mom = float(np.median(mom_values))

        threshold = self._config.abs_mom_threshold
        floor = self._config.exposure_floor

        if median_mom < threshold:
            # Exposure 축소: 음수일수록 더 많이 축소
            scale = max(abs(median_mom), _MIN_VOL)
            exposure = max(floor, 1.0 - abs(median_mom) / scale)
            # median == 0 → scale == _MIN_VOL → exposure ≈ 0 → floor
            # 실질적으로 median < 0 → exposure = floor
            if median_mom < 0:
                exposure = floor
        else:
            exposure = 1.0

        # 3. Cross-sectional Ranking
        sorted_symbols = sorted(momentums, key=lambda s: momentums[s], reverse=True)
        n_long = max(1, round(n * self._config.top_pct))

        selected = set(sorted_symbols[:n_long])
        weights: dict[str, float] = {}
        for s in self._symbols:
            weights[s] = (1.0 / n_long) if s in selected else 0.0

        return weights, exposure

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Allocator 상태 직렬화."""
        return {
            "weights": dict(self._weights),
            "bar_count": self._bar_count,
            "exposure": self._exposure,
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Allocator 상태 복원."""
        weights_val = data.get("weights")
        if isinstance(weights_val, dict):
            self._weights = {str(k): float(v) for k, v in weights_val.items()}
        bar_count_val = data.get("bar_count")
        if isinstance(bar_count_val, int | float):
            self._bar_count = int(bar_count_val)
        exposure_val = data.get("exposure")
        if isinstance(exposure_val, int | float):
            self._exposure = float(exposure_val)


# ── Module-level utility ──────────────────────────────────────────


def _clamp_and_normalize(
    raw: dict[str, float],
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """min/max clamp 후 합계 1.0 정규화 (water-filling).

    고정된 값(min/max에 도달)을 제외하고 나머지를 비례 배분합니다.
    """
    n = len(raw)
    if n == 0:
        return {}

    total_raw = sum(raw.values())
    if total_raw < _CLAMP_TOLERANCE:
        return dict.fromkeys(raw, 1.0 / n)

    # 초기 정규화
    weights = {s: w / total_raw for s, w in raw.items()}
    fixed: set[str] = set()

    for _ in range(_CLAMP_MAX_ITERATIONS):
        changed = False
        fixed_sum = 0.0
        free_keys: list[str] = []

        for s, w in weights.items():
            if s in fixed:
                fixed_sum += w
            elif w <= min_weight:
                weights[s] = min_weight
                fixed.add(s)
                fixed_sum += min_weight
                changed = True
            elif w >= max_weight:
                weights[s] = max_weight
                fixed.add(s)
                fixed_sum += max_weight
                changed = True
            else:
                free_keys.append(s)

        if not changed:
            break

        # 남은 예산을 free keys에 비례 배분
        remaining = 1.0 - fixed_sum
        if not free_keys or remaining < _CLAMP_TOLERANCE:
            break

        free_total = sum(weights[s] for s in free_keys)
        if free_total < _CLAMP_TOLERANCE:
            per_free = remaining / len(free_keys)
            for s in free_keys:
                weights[s] = per_free
        else:
            scale = remaining / free_total
            for s in free_keys:
                weights[s] *= scale

    # 최종 정규화 보장
    final_total = sum(weights.values())
    if abs(final_total - 1.0) > _CLAMP_TOLERANCE:
        weights = {s: w / final_total for s, w in weights.items()}

    return weights
