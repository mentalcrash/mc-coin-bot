"""Composite / cross-indicator functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def count_consecutive(mask: np.ndarray[Any, np.dtype[np.bool_]]) -> npt.NDArray[np.intp]:
    """Boolean mask에서 연속 True 횟수를 벡터화 계산.

    True가 연속되면 1,2,3,... 증가. False에서 0으로 리셋.

    Args:
        mask: boolean numpy 배열.

    Returns:
        연속 True 카운트 배열 (False 위치는 0).

    Example:
        >>> count_consecutive(np.array([False, True, True, True, False, True]))
        array([0, 1, 2, 3, 0, 1])
    """
    if len(mask) == 0:
        return np.array([], dtype=np.intp)
    if not mask.any():
        return np.zeros(len(mask), dtype=np.intp)

    # F→T 전환점에서 새 그룹 시작
    transitions = np.diff(mask.astype(int), prepend=0) > 0
    group_ids = np.cumsum(transitions)
    # False 위치는 그룹 0으로 마스킹
    group_ids = np.where(mask, group_ids, 0)

    # 각 True 그룹 내 누적 카운트
    s = pd.Series(group_ids)
    cumcount = s.groupby(s).cumcount() + 1
    return np.where(mask, cumcount.to_numpy(), 0).astype(np.intp)


def drawdown(close: pd.Series) -> pd.Series:
    """롤링 최고점 대비 드로다운.

    Args:
        close: 종가 시리즈.

    Returns:
        드로다운 시리즈 (항상 <= 0).
    """
    rolling_max = close.expanding().max()
    dd: pd.Series = (close - rolling_max) / rolling_max  # type: ignore[assignment]
    return pd.Series(dd, index=close.index, name="drawdown")


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling z-score.

    Args:
        series: 입력 시리즈.
        window: Rolling 윈도우 크기.
        min_periods: 최소 관측치 수 (None이면 *window*).

    Returns:
        Z-score 시리즈.
    """
    if min_periods is None:
        min_periods = window
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    result: pd.Series = (series - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return result


def bb_position(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """Bollinger Band position (0-1 normalized).

    close가 lower band에 있으면 0, upper band에 있으면 1.

    Args:
        close: 종가 시리즈.
        period: 이동평균 기간.
        std_dev: 표준편차 배수.

    Returns:
        BB position 시리즈 (0~1).
    """
    sma_val = close.rolling(period).mean()
    std_val = close.rolling(period).std()
    upper: pd.Series = sma_val + std_dev * std_val  # type: ignore[assignment]
    lower: pd.Series = sma_val - std_dev * std_val  # type: ignore[assignment]
    denom = pd.Series(upper - lower, index=close.index).replace(0, np.nan)
    bb_pos: pd.Series = (close - lower) / denom  # type: ignore[assignment]
    return bb_pos


def sma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """SMA crossover signal (fast_sma/slow_sma - 1).

    Args:
        close: 종가 시리즈.
        fast: 빠른 SMA 기간.
        slow: 느린 SMA 기간.

    Returns:
        SMA cross 시리즈.
    """
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()
    cross: pd.Series = fast_sma / slow_sma.replace(0, np.nan) - 1  # type: ignore[assignment]
    return cross


def ema_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """EMA crossover signal (fast_ema/slow_ema - 1).

    Args:
        close: 종가 시리즈.
        fast: 빠른 EMA 기간.
        slow: 느린 EMA 기간.

    Returns:
        EMA cross 시리즈.
    """
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    cross: pd.Series = fast_ema / slow_ema.replace(0, np.nan) - 1  # type: ignore[assignment]
    return cross


def hurst_exponent(
    close: pd.Series,
    window: int,
) -> pd.Series:
    """허스트 지수 (Hurst Exponent) — Rescaled Range (R/S) 방식.

    H > 0.5: 추세 지속, H < 0.5: 평균회귀, H ≈ 0.5: 랜덤워크.

    Args:
        close: 종가 시리즈.
        window: Rolling 윈도우 크기 (최소 20 권장).

    Returns:
        허스트 지수 시리즈 (0~1).
    """

    _min_hurst_samples = 20

    def _rs_hurst(arr: npt.NDArray[np.float64]) -> float:
        """R/S statistic으로 Hurst exponent 추정."""
        n = len(arr)
        if n < _min_hurst_samples:
            return np.nan
        returns = np.diff(np.log(arr))
        mean_r = returns.mean()
        deviations = returns - mean_r
        cumulative = np.cumsum(deviations)
        r = cumulative.max() - cumulative.min()
        s = returns.std(ddof=1)
        if s == 0:
            return np.nan
        rs = r / s
        return float(np.log(rs) / np.log(n))

    result = close.rolling(window=window, min_periods=window).apply(
        _rs_hurst, raw=True
    )
    return pd.Series(result, index=close.index, name="hurst_exponent")


def fractal_dimension(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """프랙탈 차원 (Higuchi method approximation).

    D ≈ 1: 추세, D ≈ 1.5: 랜덤, D ≈ 2: 복잡/혼돈.

    Args:
        close: 종가 시리즈.
        period: 윈도우 크기.

    Returns:
        프랙탈 차원 시리즈 (1~2).
    """
    n = 2 * period
    log_ret = np.log(close / close.shift(1))

    # 경로 길이 (절대 수익률 합)
    path_length = log_ret.abs().rolling(window=n, min_periods=n).sum()
    # 직선 거리 (n기간 log return 절대값)
    line_distance = np.log(close / close.shift(n)).abs()

    line_safe = line_distance.replace(0, np.nan)
    ratio = path_length / line_safe
    # D = 1 + ln(L) / ln(2n), 간소화
    result = 1.0 + np.log(ratio.clip(lower=1e-10)) / np.log(n)
    return pd.Series(result.clip(1.0, 2.0), index=close.index, name="fractal_dimension")


def price_acceleration(
    close: pd.Series,
    fast: int,
    slow: int,
) -> pd.Series:
    """가격 가속도 (2차 미분 근사).

    빠른 ROC와 느린 ROC의 차이로 모멘텀 가속/감속을 측정합니다.

    Args:
        close: 종가 시리즈.
        fast: 빠른 ROC 기간.
        slow: 느린 ROC 기간.

    Returns:
        가속도 시리즈 (양수: 가속, 음수: 감속).
    """
    fast_roc: pd.Series = close.pct_change(fast)  # type: ignore[assignment]
    slow_roc: pd.Series = close.pct_change(slow)  # type: ignore[assignment]
    return pd.Series(fast_roc - slow_roc, index=close.index, name="price_acceleration")


def rsi_divergence(
    close: pd.Series,
    rsi_series: pd.Series,
    window: int,
) -> pd.Series:
    """RSI 다이버전스 감지.

    가격과 RSI의 rolling 상관을 계산합니다.
    음의 상관 = 다이버전스 (가격 방향 ≠ RSI 방향).

    Args:
        close: 종가 시리즈.
        rsi_series: RSI 시리즈.
        window: Rolling 상관 윈도우.

    Returns:
        상관계수 시리즈 (-1 ~ +1, 음수일수록 다이버전스).
    """
    result: pd.Series = close.rolling(window).corr(rsi_series)  # type: ignore[assignment]
    return result


def trend_strength(
    adx_series: pd.Series,
    threshold: float = 25.0,
) -> pd.Series:
    """ADX 기반 추세 강도 분류.

    ADX > threshold: 추세 존재 (1.0), 아니면 비추세 (0.0).

    Args:
        adx_series: ADX 시리즈.
        threshold: 추세 판단 기준값.

    Returns:
        추세 강도 시리즈 (0.0 또는 1.0).
    """
    return pd.Series(
        np.where(adx_series > threshold, 1.0, 0.0),
        index=adx_series.index,
        name="trend_strength",
    )


def mean_reversion_score(
    close: pd.Series,
    window: int,
    std_mult: float = 2.0,
) -> pd.Series:
    """평균회귀 점수 (z-score 기반).

    종가의 rolling z-score를 반전하여 평균으로부터 떨어진 정도를 점수화합니다.
    양수: 과매도 (매수 기회), 음수: 과매수 (매도 기회).

    Args:
        close: 종가 시리즈.
        window: Rolling 윈도우.
        std_mult: z-score 스케일링 계수 (클립 범위).

    Returns:
        평균회귀 점수 시리즈.
    """
    mean = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    std_safe = std.replace(0, np.nan)
    zscore: pd.Series = (close - mean) / std_safe  # type: ignore[assignment]
    # 반전: 과매도(낮은 가격)일 때 양수
    mr_score = -zscore.clip(-std_mult, std_mult)
    return pd.Series(mr_score, index=close.index, name="mean_reversion_score")


def squeeze_detect(
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    kc_upper: pd.Series,
    kc_lower: pd.Series,
) -> pd.Series:
    """Squeeze 상태 감지 (BB가 KC 안에 있으면 squeeze ON).

    Args:
        bb_upper: Bollinger Band 상단.
        bb_lower: Bollinger Band 하단.
        kc_upper: Keltner Channel 상단.
        kc_lower: Keltner Channel 하단.

    Returns:
        bool Series (True = squeeze ON).
    """
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    return pd.Series(squeeze_on, index=bb_upper.index, name="squeeze_on", dtype=bool)
