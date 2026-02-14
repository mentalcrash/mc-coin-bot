"""VWAP Disposition Momentum Preprocessor (Indicator Calculation).

Rolling VWAP, CGO, Volume ratio 등 행동재무학 기반 지표를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.vwap_disposition.config import VWAPDispositionConfig

logger = logging.getLogger(__name__)


def _calculate_rolling_vwap(
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling VWAP (Volume-Weighted Average Price) 계산.

    Rolling VWAP = sum(close * volume, window) / sum(volume, window)

    시장 참여자의 평균 취득가(cost basis) proxy로 사용합니다.

    Args:
        close: 종가 시리즈
        volume: 거래량 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Rolling VWAP 시리즈
    """
    pv = close * volume
    vwap = (
        pv.rolling(window=window, min_periods=window).sum()
        / volume.rolling(window=window, min_periods=window).sum()
    )
    return pd.Series(vwap, index=close.index, name="vwap")


def _calculate_cgo(
    close: pd.Series,
    vwap: pd.Series,
) -> pd.Series:
    """Capital Gains Overhang (CGO) 계산.

    CGO = (close - vwap) / vwap

    양수: 미실현 이익 (차익 실현 압력)
    음수: 미실현 손실 (항복 매도 또는 손실 회피)

    Args:
        close: 종가 시리즈
        vwap: Rolling VWAP 시리즈

    Returns:
        CGO 시리즈
    """
    vwap_safe: pd.Series = vwap.replace(0, np.nan)  # type: ignore[assignment]
    cgo = (close - vwap_safe) / vwap_safe
    return pd.Series(cgo, index=close.index, name="cgo")


def _calculate_volume_ratio(
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Volume ratio 계산 (현재 거래량 / 이동 평균 거래량).

    Args:
        volume: 거래량 시리즈
        window: 이동 평균 윈도우

    Returns:
        Volume ratio 시리즈 (1.0 = 평균, >1.0 = spike, <1.0 = decline)
    """
    vol_ma = volume.rolling(window=window, min_periods=window).mean()
    vol_ma_safe: pd.Series = vol_ma.replace(0, np.nan)  # type: ignore[assignment]
    ratio = volume / vol_ma_safe
    return pd.Series(ratio, index=volume.index, name="volume_ratio")


def preprocess(
    df: pd.DataFrame,
    config: VWAPDispositionConfig,
) -> pd.DataFrame:
    """VWAP Disposition Momentum 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - vwap: Rolling VWAP (cost basis proxy)
        - cgo: Capital Gains Overhang
        - volume_ratio: 현재 거래량 / 평균 거래량
        - mom_direction: 모멘텀 방향 (sign of rolling sum)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: VWAP Disposition 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Rolling VWAP
    result["vwap"] = _calculate_rolling_vwap(
        close_series,
        volume_series,
        window=config.vwap_window,
    )
    vwap_series: pd.Series = result["vwap"]  # type: ignore[assignment]

    # 3. Capital Gains Overhang
    result["cgo"] = _calculate_cgo(close_series, vwap_series)

    # 4. Volume ratio
    result["volume_ratio"] = _calculate_volume_ratio(
        volume_series,
        window=config.vol_ratio_window,
    )

    # 5. 모멘텀 방향
    mom_sum = returns_series.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    result["mom_direction"] = np.sign(mom_sum)

    # 6. 실현 변동성
    realized_vol = realized_volatility(
        returns_series,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR 계산
    result["atr"] = atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 9. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        cgo_mean = valid_data["cgo"].mean()
        vr_mean = valid_data["volume_ratio"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "VWAP-Disposition Indicators | Avg CGO: %.4f, Avg Vol Ratio: %.4f, Avg Vol Scalar: %.4f",
            cgo_mean,
            vr_mean,
            vs_mean,
        )

    return result
