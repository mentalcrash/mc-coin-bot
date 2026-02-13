"""Regime Score 계산 — 순수 함수.

파생상품 지표(funding rate, OI, LS ratio, taker ratio)로
시장 regime을 -1.0 ~ +1.0 범위의 단일 점수로 환산합니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - 금융 계산 교차 검증 가능한 정규화 기준 명시
"""

from __future__ import annotations

# 정규화 기준
_FUNDING_RATE_SCALE = 0.001  # ±0.1% → ±1.0
_OI_CHANGE_SCALE = 0.20  # ±20% → ±1.0
_LS_RATIO_CENTER = 1.0  # 중심값
_LS_RATIO_HALF_RANGE = 0.5  # 0.5~1.5 → -1~+1
_TAKER_RATIO_CENTER = 1.0
_TAKER_RATIO_HALF_RANGE = 0.2  # 0.8~1.2 → -1~+1

# Regime 라벨 임계값
_EXTREME_GREED_THRESHOLD = 0.5
_BULLISH_THRESHOLD = 0.2
_BEARISH_THRESHOLD = -0.2
_EXTREME_FEAR_THRESHOLD = -0.5


def _clamp(value: float, lo: float, hi: float) -> float:
    """값을 [lo, hi] 범위로 클램핑."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _normalize_funding(funding_rate: float) -> float:
    """Funding rate 정규화: ±0.1% → ±1.0."""
    return _clamp(funding_rate / _FUNDING_RATE_SCALE, -1.0, 1.0)


def _normalize_oi_change(oi_change_pct: float) -> float:
    """OI 변화율 정규화: ±20% → ±1.0."""
    return _clamp(oi_change_pct / _OI_CHANGE_SCALE, -1.0, 1.0)


def _normalize_ls_ratio(ls_ratio: float) -> float:
    """LS ratio 정규화: contrarian (높은 LS → 과열 → 음수).

    1.0 중심, 0.5~1.5 → +1~-1 (역방향).
    """
    deviation = ls_ratio - _LS_RATIO_CENTER
    normalized = deviation / _LS_RATIO_HALF_RANGE
    return _clamp(-normalized, -1.0, 1.0)


def _normalize_taker_ratio(taker_ratio: float) -> float:
    """Taker ratio 정규화: 1.0 중심, 0.8~1.2 → -1~+1."""
    deviation = taker_ratio - _TAKER_RATIO_CENTER
    normalized = deviation / _TAKER_RATIO_HALF_RANGE
    return _clamp(normalized, -1.0, 1.0)


def compute_regime_score(
    funding_rate: float,
    oi_change_pct: float,
    ls_ratio: float,
    taker_ratio: float,
) -> float:
    """4개 파생상품 지표로 regime score 계산.

    Args:
        funding_rate: 최근 funding rate (e.g. 0.0001 = 0.01%)
        oi_change_pct: OI 변화율 (e.g. 0.08 = +8%)
        ls_ratio: Long/Short 계정 비율 (e.g. 1.47)
        taker_ratio: Taker Buy/Sell 비율 (e.g. 1.12)

    Returns:
        -1.0 ~ +1.0 범위의 regime score
    """
    components = [
        _normalize_funding(funding_rate),
        _normalize_oi_change(oi_change_pct),
        _normalize_ls_ratio(ls_ratio),
        _normalize_taker_ratio(taker_ratio),
    ]
    return sum(components) / len(components)


def classify_regime(score: float) -> str:
    """Regime score를 라벨로 분류.

    Args:
        score: regime score (-1.0 ~ +1.0)

    Returns:
        Regime 라벨 문자열
    """
    if score > _EXTREME_GREED_THRESHOLD:
        return "Extreme Greed"
    if score > _BULLISH_THRESHOLD:
        return "Bullish"
    if score > _BEARISH_THRESHOLD:
        return "Neutral"
    if score > _EXTREME_FEAR_THRESHOLD:
        return "Bearish"
    return "Extreme Fear"
