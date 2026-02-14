"""Volatility indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: float = 365.0,
    min_periods: int | None = None,
) -> pd.Series:
    """실현 변동성 (연환산).

    Args:
        returns: 수익률 시리즈.
        window: Rolling 윈도우 크기.
        annualization_factor: 연환산 계수 (일봉 365).
        min_periods: 최소 관측치 수 (None이면 *window*).

    Returns:
        연환산 변동성 시리즈.
    """
    if min_periods is None:
        min_periods = window
    rolling_std = returns.rolling(window=window, min_periods=min_periods).std()
    return rolling_std * np.sqrt(annualization_factor)


def volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러: vol_target / realized_vol.

    Args:
        realized_vol: 실현 변동성 시리즈.
        vol_target: 연간 목표 변동성.
        min_volatility: 최소 변동성 클램프 (0 나누기 방지).

    Returns:
        변동성 스케일러 시리즈.
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    return vol_target / clamped_vol


def parkinson_volatility(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson volatility (range-based estimator).

    Formula: sqrt(1 / (4 * ln(2)) * ln(high / low)^2)

    Args:
        high: 고가 시리즈.
        low: 저가 시리즈.

    Returns:
        Parkinson volatility 시리즈.
    """
    log_hl_ratio = np.log(high / low)
    return pd.Series(
        np.sqrt(1.0 / (4.0 * np.log(2.0)) * log_hl_ratio**2),
        index=high.index,
        name="parkinson_vol",
    )


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Garman-Klass single-period variance.

    GK variance는 OHLC 4가지 가격을 모두 활용하여
    close-to-close 변동성보다 효율적인 추정치를 제공합니다.

    Formula: 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

    Args:
        open_: 시가 시리즈.
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.

    Returns:
        GK variance 시리즈 (per-bar).
    """
    ln2 = np.log(2)
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * ln2 - 1) * log_co**2
    return pd.Series(gk_var, index=close.index, name="gk_var")


def volatility_of_volatility(
    realized_vol: pd.Series,
    window: int,
) -> pd.Series:
    """VoV — 변동성의 변동성 (Volatility of Volatility).

    실현 변동성의 rolling 표준편차를 계산합니다.
    높은 VoV는 regime 전환 신호로 활용됩니다.

    Args:
        realized_vol: 실현 변동성 시리즈.
        window: Rolling 윈도우 크기.

    Returns:
        VoV 시리즈.
    """
    result: pd.Series = realized_vol.rolling(  # type: ignore[assignment]
        window=window, min_periods=window
    ).std()
    return result


def yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Yang-Zhang volatility estimator.

    Overnight returns, Rogers-Satchell, open-to-close 3개를 결합한
    가장 효율적인 range-based 변동성 추정치입니다.

    Args:
        open_: 시가 시리즈.
        high: 고가 시리즈.
        low: 저가 시리즈.
        close: 종가 시리즈.
        window: Rolling 윈도우 크기.

    Returns:
        Yang-Zhang volatility 시리즈.
    """
    # Overnight returns (close_prev → open)
    log_oc = np.log(open_ / close.shift(1))
    # Open-to-close
    log_co = np.log(close / open_)
    # Rogers-Satchell
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)
    rs = log_ho * log_hc + log_lo * log_lc

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    overnight_var = log_oc.rolling(window=window, min_periods=window).var()
    oc_var = log_co.rolling(window=window, min_periods=window).var()
    rs_var = rs.rolling(window=window, min_periods=window).mean()

    yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
    yz_vol = np.sqrt(yz_var.clip(lower=0))
    return pd.Series(yz_vol, index=close.index, name="yang_zhang_vol")


def vol_percentile_rank(
    realized_vol: pd.Series,
    window: int,
) -> pd.Series:
    """변동성 백분위 순위 (0~1).

    현재 실현 변동성이 과거 window 기간 대비 어느 위치인지 반환합니다.

    Args:
        realized_vol: 실현 변동성 시리즈.
        window: Percentile rank 윈도우.

    Returns:
        백분위 순위 시리즈 (0~1, 1에 가까울수록 고변동성).
    """
    result: pd.Series = realized_vol.rolling(  # type: ignore[assignment]
        window=window, min_periods=min(window, 60)
    ).rank(pct=True)
    return result


def vol_regime(
    returns: pd.Series,
    vol_lookback: int,
    vol_rank_lookback: int,
    annualization_factor: float = 365.0,
) -> pd.Series:
    """변동성 regime 판별 (percentile rank 0~1).

    Rolling 변동성의 percentile rank를 계산하여 현재 변동성이
    과거 대비 어느 수준인지 0~1 범위로 반환합니다.

    Args:
        returns: 수익률 시리즈.
        vol_lookback: 변동성 계산 윈도우.
        vol_rank_lookback: Percentile rank 계산 윈도우.
        annualization_factor: 연환산 계수.

    Returns:
        vol_pct 시리즈 (0~1, 1에 가까울수록 고변동성).
    """
    vol = returns.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(
        annualization_factor
    )
    vol_pct = vol.rolling(
        vol_rank_lookback, min_periods=min(vol_rank_lookback, 60)
    ).rank(pct=True)
    return pd.Series(vol_pct, index=returns.index, name="vol_regime")
