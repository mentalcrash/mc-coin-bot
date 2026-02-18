"""Position Netting — 포지션 넷팅 순수 함수.

Pod별 가중치를 심볼 레벨로 넷팅하고, Fill을 비례 귀속하며,
레버리지 한도 초과 시 가중치를 축소합니다.

모든 함수는 stateless pure function입니다 (allocator.py 패턴 준수).

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Module-level pure functions (no class needed)
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Constants ─────────────────────────────────────────────────────

_MIN_WEIGHT = 1e-8


# ── Data Classes ─────────────────────────────────────────────────


@dataclass(frozen=True)
class NettingStats:
    """Pod 간 포지션 상쇄 통계.

    Attributes:
        gross_sum: 전체 gross exposure (sum of abs weights across all pods)
        net_sum: 넷팅 후 net exposure (sum of abs netted weights)
        offset_ratio: 상쇄 비율 (0=상쇄 없음, 1=완전 상쇄)
    """

    gross_sum: float
    net_sum: float
    offset_ratio: float


# ── Pure Functions ────────────────────────────────────────────────


def compute_net_weights(
    pod_global_weights: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Pod별 글로벌 가중치를 심볼 레벨로 넷팅합니다.

    Args:
        pod_global_weights: {pod_id: {symbol: weight}} 매핑

    Returns:
        {symbol: net_weight} 넷팅된 심볼별 가중치
    """
    net: dict[str, float] = {}
    for symbol_weights in pod_global_weights.values():
        for symbol, weight in symbol_weights.items():
            net[symbol] = net.get(symbol, 0.0) + weight
    return net


def compute_netting_stats(
    pod_global_weights: dict[str, dict[str, float]],
) -> NettingStats:
    """Pod별 글로벌 가중치에서 넷팅 상쇄 통계를 계산합니다.

    Args:
        pod_global_weights: {pod_id: {symbol: weight}} 매핑

    Returns:
        NettingStats (gross_sum, net_sum, offset_ratio)
    """
    # Gross: sum of abs(weight) across all pods
    gross = 0.0
    for symbol_weights in pod_global_weights.values():
        for weight in symbol_weights.values():
            gross += abs(weight)

    # Net: sum of abs(netted weight per symbol)
    net_weights = compute_net_weights(pod_global_weights)
    net = sum(abs(w) for w in net_weights.values())

    if gross < _MIN_WEIGHT:
        return NettingStats(gross_sum=0.0, net_sum=0.0, offset_ratio=0.0)

    offset_ratio = 1.0 - net / gross
    return NettingStats(gross_sum=gross, net_sum=net, offset_ratio=offset_ratio)


def compute_deltas(
    net_targets: dict[str, float],
    current_weights: dict[str, float],
) -> dict[str, float]:
    """목표 가중치와 현재 가중치의 차이를 계산합니다.

    Args:
        net_targets: {symbol: target_weight}
        current_weights: {symbol: current_weight}

    Returns:
        {symbol: delta} 매핑 (양수 = 매수, 음수 = 매도)
    """
    all_symbols = set(net_targets) | set(current_weights)
    return {
        symbol: net_targets.get(symbol, 0.0) - current_weights.get(symbol, 0.0)
        for symbol in all_symbols
    }


def attribute_fill(
    symbol: str,
    qty: float,
    price: float,
    fee: float,
    pod_targets: dict[str, float],
    *,
    is_buy: bool,
) -> dict[str, tuple[float, float, float]]:
    """Fill을 Pod별 타겟 비율에 따라 비례 귀속합니다.

    BUY fill은 target > 0(long) pods에만, SELL fill은 target < 0(short) pods에만 귀속합니다.

    Args:
        symbol: 체결 심볼
        qty: 체결 수량
        price: 체결 가격
        fee: 수수료
        pod_targets: {pod_id: global_target_weight} — 해당 심볼에 대한 Pod별 타겟
        is_buy: True면 BUY fill, False면 SELL fill

    Returns:
        {pod_id: (attributed_qty, price, attributed_fee)} 매핑.
        매칭 pods가 없거나 합이 0이면 빈 dict 반환.
    """
    if not pod_targets:
        return {}

    # 방향 필터: BUY → long pods, SELL → short pods
    matching = {
        pid: t
        for pid, t in pod_targets.items()
        if (is_buy and t > _MIN_WEIGHT) or (not is_buy and t < -_MIN_WEIGHT)
    }

    if not matching:
        return {}

    total_abs = sum(abs(t) for t in matching.values())
    if total_abs < _MIN_WEIGHT:
        return {}

    result: dict[str, tuple[float, float, float]] = {}
    for pod_id, target in matching.items():
        share = abs(target) / total_abs
        result[pod_id] = (qty * share, price, fee * share)

    return result


def compute_gross_leverage(net_weights: dict[str, float]) -> float:
    """넷 가중치에서 총 레버리지(gross exposure)를 계산합니다.

    Gross leverage = sum(abs(weight_i))

    Args:
        net_weights: {symbol: net_weight}

    Returns:
        총 레버리지 (float)
    """
    return sum(abs(w) for w in net_weights.values())


def scale_weights_to_leverage(
    weights: dict[str, float],
    max_leverage: float,
) -> dict[str, float]:
    """가중치를 최대 레버리지에 맞게 비례 축소합니다.

    현재 gross leverage가 max_leverage 이하이면 원본 반환.
    초과하면 비례 축소하여 정확히 max_leverage가 되도록 합니다.

    Args:
        weights: {symbol: weight}
        max_leverage: 최대 허용 gross leverage

    Returns:
        축소된 {symbol: weight} 매핑
    """
    gross = compute_gross_leverage(weights)
    if gross <= max_leverage or gross < _MIN_WEIGHT:
        return dict(weights)

    scale = max_leverage / gross
    return {symbol: w * scale for symbol, w in weights.items()}
