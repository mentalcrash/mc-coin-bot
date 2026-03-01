"""Drawdown-Based De-Risking Overlay — Pod-level drawdown 방어.

Pod별 현재 drawdown이 임계값을 초과하면 해당 Pod의 weight를 축소합니다.

기존 `_apply_risk_defense()`와의 차이:
    - _apply_risk_defense(): portfolio CRITICAL alert → **모든** pod 50% (비상)
    - dd_derisk: **개별** pod DD 기반 → 해당 pod만 축소 (상시 운용, 독립)

Pure function 패턴 (vol_targeting.py와 동일).

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Zero-Tolerance Lint Policy
"""

from __future__ import annotations


def apply_dd_derisk(
    weights: dict[str, float],
    pod_drawdowns: dict[str, float],
    half_threshold: float,
    zero_threshold: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Pod별 drawdown 기반 weight 축소.

    Args:
        weights: {pod_id: fraction} 현재 가중치
        pod_drawdowns: {pod_id: current_drawdown} (양수 표현, 예: 0.15 = -15%)
        half_threshold: DD >= 이 값 → weight * 0.5
        zero_threshold: DD >= 이 값 → weight = 0.0

    Returns:
        (adjusted_weights, actions) 튜플.
        actions: {pod_id: "normal" | "halved" | "zeroed"}
    """
    adjusted: dict[str, float] = {}
    actions: dict[str, str] = {}

    for pod_id, w in weights.items():
        dd = pod_drawdowns.get(pod_id)

        if dd is None:
            # drawdown 정보 없는 pod → 변경 없음
            adjusted[pod_id] = w
            actions[pod_id] = "normal"
        elif dd >= zero_threshold:
            adjusted[pod_id] = 0.0
            actions[pod_id] = "zeroed"
        elif dd >= half_threshold:
            adjusted[pod_id] = w * 0.5
            actions[pod_id] = "halved"
        else:
            adjusted[pod_id] = w
            actions[pod_id] = "normal"

    return adjusted, actions
