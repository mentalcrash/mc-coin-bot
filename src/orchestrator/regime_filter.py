"""Regime-Adaptive Cash Buffer — ADX 기반 횡보장 현금 비중 확대.

3개 ACTIVE 전략이 모두 trend-following이므로,
추세 부재(낮은 ADX) 시 방어적으로 현금 비중을 높여 위험을 줄입니다.

Pure function 패턴 (risk_aggregator.py와 동일).

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Zero-Tolerance Lint Policy
"""

from __future__ import annotations

import numpy as np

# ── Constants ─────────────────────────────────────────────────────

_MIN_ADX_PERIOD = 5
_MIN_DATA_FOR_ADX = 3  # ADX 계산에 필요한 최소 DX 값 수


# ── Pure Functions ────────────────────────────────────────────────


def compute_adx(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int,
) -> float:
    """최신 ADX 값을 계산합니다 (0~100).

    Wilder의 ADX 공식:
    1. True Range, +DM, -DM 계산
    2. Smoothed TR, +DM, -DM (Wilder EMA)
    3. +DI, -DI → DX
    4. ADX = Smoothed DX

    Args:
        high: 고가 시계열
        low: 저가 시계열
        close: 종가 시계열
        period: ADX 기간 (일반적으로 14)

    Returns:
        최신 ADX 값 (0~100). 데이터 부족 시 0.0.
    """
    n = len(close)
    min_required = 2 * period + 1
    if n < min_required or len(high) < min_required or len(low) < min_required:
        return 0.0

    h = np.array(high[-min_required - period :], dtype=np.float64)
    l = np.array(low[-min_required - period :], dtype=np.float64)  # noqa: E741
    c = np.array(close[-min_required - period :], dtype=np.float64)
    data_len = len(h)

    # True Range, +DM, -DM
    tr = np.zeros(data_len - 1)
    plus_dm = np.zeros(data_len - 1)
    minus_dm = np.zeros(data_len - 1)

    for i in range(1, data_len):
        idx = i - 1
        tr[idx] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        plus_dm[idx] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[idx] = down_move if (down_move > up_move and down_move > 0) else 0.0

    if len(tr) < period:
        return 0.0

    # Wilder smoothing (initial: SMA, then EMA)
    smoothed_tr = float(np.sum(tr[:period]))
    smoothed_plus_dm = float(np.sum(plus_dm[:period]))
    smoothed_minus_dm = float(np.sum(minus_dm[:period]))

    dx_values: list[float] = []

    for i in range(period, len(tr)):
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr[i]
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]

        if smoothed_tr > 0:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
        dx_values.append(dx)

    if len(dx_values) < _MIN_DATA_FOR_ADX:
        return 0.0

    # ADX = Wilder smoothed DX
    adx = float(np.mean(dx_values[:period]))  # initial ADX = SMA of first `period` DX
    for i in range(period, len(dx_values)):
        adx = (adx * (period - 1) + dx_values[i]) / period

    return adx


def compute_regime_cash_buffer(
    price_histories: dict[str, list[float]],
    high_histories: dict[str, list[float]],
    low_histories: dict[str, list[float]],
    adx_period: int = 14,
    trend_threshold: float = 25.0,
    range_threshold: float = 20.0,
    max_cash_buffer: float = 0.40,
) -> tuple[float, float]:
    """다중 에셋 평균 ADX 기반 cash buffer를 계산합니다.

    Args:
        price_histories: {symbol: [close_prices]}
        high_histories: {symbol: [high_prices]}
        low_histories: {symbol: [low_prices]}
        adx_period: ADX 기간
        trend_threshold: ADX ≥ 이 값 → buffer = 0 (풀 투자)
        range_threshold: ADX ≤ 이 값 → buffer = max_cash_buffer
        max_cash_buffer: 최대 현금 비중

    Returns:
        (cash_buffer, avg_adx) 튜플.
        데이터 부족 시 (0.0, 0.0) — 보수적으로 풀 투자.
    """
    adx_values: list[float] = []

    common_symbols = [s for s in price_histories if s in high_histories and s in low_histories]

    for symbol in common_symbols:
        adx = compute_adx(
            high=high_histories[symbol],
            low=low_histories[symbol],
            close=price_histories[symbol],
            period=adx_period,
        )
        if adx > 0.0:
            adx_values.append(adx)

    if not adx_values:
        return 0.0, 0.0

    avg_adx = float(np.mean(adx_values))

    # 선형 보간
    if avg_adx >= trend_threshold:
        cash_buffer = 0.0
    elif avg_adx <= range_threshold:
        cash_buffer = max_cash_buffer
    else:
        # 선형 보간: range→trend 구간에서 max→0
        ratio = (avg_adx - range_threshold) / (trend_threshold - range_threshold)
        cash_buffer = max_cash_buffer * (1.0 - ratio)

    return cash_buffer, avg_adx


def apply_cash_buffer(
    weights: dict[str, float],
    cash_buffer: float,
) -> dict[str, float]:
    """Cash buffer를 적용하여 가중치를 스케일링합니다.

    Args:
        weights: {pod_id: fraction} 현재 가중치
        cash_buffer: 현금 비중 (0~1)

    Returns:
        스케일된 가중치 (전체 합 = 원래 합 * (1 - cash_buffer)).
    """
    investment_ratio = 1.0 - cash_buffer
    return {pid: w * investment_ratio for pid, w in weights.items()}
