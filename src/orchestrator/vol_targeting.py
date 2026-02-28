"""Volatility Targeting Overlay — 포트폴리오 변동성 목표 조절.

포트폴리오 실현 변동성을 목표치에 맞추도록 전체 Pod 가중치를 스케일링합니다.

Pure function 패턴 (risk_aggregator.py와 동일).

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Zero-Tolerance Lint Policy
"""

from __future__ import annotations

import numpy as np

# ── Constants ─────────────────────────────────────────────────────

_MIN_VARIANCE = 1e-12
_ANNUALIZATION_FACTOR = 365.0
_MIN_RETURNS_FOR_VOL = 5  # 실현 변동성 계산 최소 데이터 수


# ── Pure Functions ────────────────────────────────────────────────


def compute_realized_vol(
    pod_returns: dict[str, list[float]],
    weights: dict[str, float],
    lookback: int,
) -> float:
    """포트폴리오 연간 실현 변동성을 계산합니다.

    Args:
        pod_returns: {pod_id: [daily_returns]} 매핑
        weights: {pod_id: fraction} 매핑
        lookback: 실현 변동성 계산 lookback (일)

    Returns:
        연간화된 포트폴리오 실현 변동성. 데이터 부족 시 0.0.
    """
    common_pods = [pid for pid in weights if pid in pod_returns and weights[pid] > _MIN_VARIANCE]
    if not common_pods:
        return 0.0

    # lookback만큼 자르기
    min_len = min(len(pod_returns[pid]) for pid in common_pods)
    actual_lookback = min(lookback, min_len)
    if actual_lookback < _MIN_RETURNS_FOR_VOL:
        return 0.0

    # 가중 포트폴리오 수익률 계산
    portfolio_returns = np.zeros(actual_lookback)
    for pid in common_pods:
        w = weights[pid]
        returns_arr = np.array(pod_returns[pid][-actual_lookback:])
        portfolio_returns += w * returns_arr

    # 연간화 변동성
    daily_vol = float(np.std(portfolio_returns, ddof=1))
    return daily_vol * np.sqrt(_ANNUALIZATION_FACTOR)


def compute_vol_scalar(
    realized_vol: float,
    target_vol: float,
    floor: float,
    cap: float,
) -> float:
    """변동성 스케일링 계수를 계산합니다.

    scalar = target_vol / realized_vol, clamped to [floor, cap].

    Args:
        realized_vol: 실현 연간 변동성
        target_vol: 목표 연간 변동성
        floor: 스케일러 하한 (최소 투자 비율)
        cap: 스케일러 상한 (최대 투자 비율)

    Returns:
        변동성 스케일링 계수. realized_vol이 0에 가까우면 cap 반환.
    """
    if realized_vol < _MIN_VARIANCE:
        return cap

    raw_scalar = target_vol / realized_vol
    return max(floor, min(raw_scalar, cap))


def apply_vol_targeting(
    weights: dict[str, float],
    pod_returns: dict[str, list[float]],
    target_vol: float,
    lookback: int,
    floor: float = 0.10,
    cap: float = 1.5,
) -> tuple[dict[str, float], float]:
    """Volatility targeting overlay를 적용합니다.

    Args:
        weights: {pod_id: fraction} 현재 가중치
        pod_returns: {pod_id: [daily_returns]} 매핑
        target_vol: 목표 연간 변동성
        lookback: 실현 변동성 계산 lookback (일)
        floor: 스케일러 하한
        cap: 스케일러 상한

    Returns:
        (scaled_weights, vol_scalar) 튜플.
    """
    realized = compute_realized_vol(pod_returns, weights, lookback)
    scalar = compute_vol_scalar(realized, target_vol, floor, cap)

    scaled_weights = {pid: w * scalar for pid, w in weights.items()}
    return scaled_weights, scalar
