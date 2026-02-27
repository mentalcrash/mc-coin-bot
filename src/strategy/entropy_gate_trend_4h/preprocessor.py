"""Entropy-Gate Trend 4H 전처리 모듈.

Permutation Entropy 계산 + 3-scale Donchian Channel + 변동성 스케일러.
모든 연산은 벡터화 (for 루프 금지). Entropy rolling은 .apply() 예외 허용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.market.indicators import (
    donchian_channel,
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.entropy_gate_trend_4h.config import EntropyGateTrend4hConfig

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# SampEn은 대안 구현으로 제공 (PermEn 대비 느리지만 정보론적 해석이 직관적)
__all__ = ["_permutation_entropy", "_sample_entropy_single", "preprocess"]


def _permutation_entropy(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]], m: int = 3, delay: int = 1
) -> float:
    """단일 window에 대한 Permutation Entropy 계산.

    Permutation Entropy는 time series의 ordinal pattern 분포를 통해
    복잡도를 측정한다. SampEn보다 O(N*m!) 시간복잡도로 훨씬 빠르고 robust.

    Args:
        x: 1D time series (length >= m * delay).
        m: Embedding dimension (permutation 길이).
        delay: Time delay between elements.

    Returns:
        Permutation entropy (bits). 높을수록 random, 낮을수록 predictable.
        데이터 부족 시 NaN 반환.
    """
    n = len(x)
    n_patterns = n - (m - 1) * delay
    if n_patterns <= 0:
        return float("nan")

    # 각 시점에서 m개 원소의 순위 패턴(ordinal pattern) 추출
    # shape: (n_patterns, m)
    indices = np.arange(m) * delay
    patterns = np.array([np.argsort(x[i + indices]) for i in range(n_patterns)])

    # 고유 패턴 카운트
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / n_patterns

    # Shannon entropy (bits)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def _sample_entropy_single(
    x: np.ndarray[Any, np.dtype[np.floating[Any]]], m: int, r: float
) -> float:
    """단일 window에 대한 Sample Entropy 계산.

    SampEn = -ln(A/B) where:
        A = number of template matches for length m+1
        B = number of template matches for length m

    Args:
        x: 1D time series.
        m: Embedding dimension.
        r: Tolerance (typically r_mult * std(x)).

    Returns:
        Sample entropy. 높을수록 random, 낮을수록 predictable.
        데이터 부족 시 NaN 반환.
    """
    n = len(x)
    if n < m + 2 or r <= 0:
        return float("nan")

    def _count_matches(dim: int) -> int:
        """Chebyshev distance 기반 template matching count."""
        count = 0
        templates = np.array([x[i : i + dim] for i in range(n - dim)])
        n_templates = len(templates)
        for i in range(n_templates):
            # Broadcasting: |template_i - template_j| < r for all dimensions
            diffs = np.abs(templates[i + 1 :] - templates[i])
            matches = np.all(diffs <= r, axis=1)
            count += int(np.sum(matches))
        return count

    b_count = _count_matches(m)
    a_count = _count_matches(m + 1)

    if b_count == 0:
        return float("nan")
    if a_count == 0:
        # ln(0)은 정의 불가 → 매우 높은 entropy 반환
        return float("inf")

    return -np.log(a_count / b_count)


def preprocess(df: pd.DataFrame, config: EntropyGateTrend4hConfig) -> pd.DataFrame:
    """Entropy-Gate Trend 4H feature 계산.

    Calculated Columns:
        - perm_entropy: rolling permutation entropy
        - is_predictable: entropy < threshold (1=predictable, 0=random)
        - dc_upper_{s}, dc_lower_{s}: 3-scale Donchian Channel
        - returns: log return
        - realized_vol: 연환산 실현 변동성
        - vol_scalar: 변동성 스케일러
        - drawdown: HEDGE_ONLY용 drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Permutation Entropy (rolling) ---
    # PermEn은 O(N*m!) per window. m=3, window=48이면 48*6=288 ops/bar — 충분히 빠름
    m = config.entropy_m
    delay = config.entropy_delay

    def _pe_apply(data: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
        return _permutation_entropy(data, m=m, delay=delay)

    df["perm_entropy"] = returns.rolling(
        window=config.entropy_window,
        min_periods=config.entropy_window,
    ).apply(_pe_apply, raw=True)

    # --- Predictability Gate ---
    df["is_predictable"] = (df["perm_entropy"] < config.entropy_threshold).astype(int)

    # --- 3-Scale Donchian Channels ---
    scales = (config.dc_scale_short, config.dc_scale_mid, config.dc_scale_long)
    for s in scales:
        upper, _mid, lower = donchian_channel(high, low, s)
        df[f"dc_upper_{s}"] = upper
        df[f"dc_lower_{s}"] = lower

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
