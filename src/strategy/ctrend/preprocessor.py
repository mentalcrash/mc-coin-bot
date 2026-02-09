"""CTREND Preprocessor (28 Technical Feature Computation).

이 모듈은 CTREND 전략에 필요한 28개 기술적 지표를 계산합니다.
모든 feature는 벡터화된 연산으로 계산되며 "feat_" 접두사로 컬럼에 추가됩니다.

Features (28):
    - MACD (3): macd, macd_signal, macd_hist
    - RSI (2): rsi_14, rsi_7
    - CCI (2): cci_20, cci_14
    - Williams %R (1): williams_r_14
    - Stochastic (2): stoch_k, stoch_d
    - OBV (1): obv_norm (normalized)
    - Chaikin MF (1): cmf_20
    - Bollinger (2): bb_pos_20, bb_pos_10
    - SMA Cross (3): sma_cross_5_20, sma_cross_10_50, sma_cross_20_100
    - EMA Cross (2): ema_cross_5_20, ema_cross_10_50
    - ROC (3): roc_5, roc_10, roc_21
    - ATR Ratio (2): atr_ratio_14, atr_ratio_28
    - Volume MACD (1): volume_macd
    - Momentum (2): mom_5, mom_21
    - Volatility Ratio (1): vol_ratio

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: Compatible output format
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.tsmom.preprocessor import (
    calculate_atr,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.ctrend.config import CTRENDConfig

logger = logging.getLogger(__name__)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence) 계산.

    Args:
        close: 종가 시리즈
        fast: 빠른 EMA 기간
        slow: 느린 EMA 기간
        signal: 시그널 EMA 기간

    Returns:
        (MACD line, Signal line, Histogram) 튜플
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line: pd.Series = ema_fast - ema_slow  # type: ignore[assignment]
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram: pd.Series = macd_line - signal_line  # type: ignore[assignment]
    return macd_line, signal_line, histogram


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) 계산 (0-100).

    Args:
        close: 종가 시리즈
        period: 계산 기간

    Returns:
        RSI 시리즈 (0-100 범위)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi: pd.Series = 100 - (100 / (1 + rs))  # type: ignore[assignment]
    return rsi


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """CCI (Commodity Channel Index) 계산.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: 계산 기간

    Returns:
        CCI 시리즈
    """
    tp: pd.Series = (high + low + close) / 3  # type: ignore[assignment]
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(),  # type: ignore[reportUnknownLambdaType]
        raw=True,
    )
    cci: pd.Series = (tp - sma) / (0.015 * mad.replace(0, np.nan))  # type: ignore[assignment]
    return cci


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R 계산 (-100 to 0).

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: 계산 기간

    Returns:
        Williams %R 시리즈 (-100 ~ 0)
    """
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    denom = (highest - lowest).replace(0, np.nan)
    wr: pd.Series = -100 * (highest - close) / denom  # type: ignore[assignment]
    return wr


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K, %D 계산.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        k_period: %K 기간
        d_period: %D 기간 (K의 이동평균)

    Returns:
        (%K, %D) 튜플
    """
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k: pd.Series = 100 * (close - lowest) / denom  # type: ignore[assignment]
    d: pd.Series = k.rolling(d_period).mean()  # type: ignore[assignment]
    return k, d


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV (On Balance Volume) 계산.

    Args:
        close: 종가 시리즈
        volume: 거래량 시리즈

    Returns:
        OBV 시리즈
    """
    direction = np.sign(close.diff())
    obv: pd.Series = (volume * direction).cumsum()  # type: ignore[assignment]
    return obv


def compute_chaikin_mf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow 계산.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        volume: 거래량 시리즈
        period: 계산 기간

    Returns:
        CMF 시리즈
    """
    hl_range = (high - low).replace(0, np.nan)
    mfm: pd.Series = ((close - low) - (high - close)) / hl_range  # type: ignore[assignment]
    mfv: pd.Series = mfm * volume  # type: ignore[assignment]
    vol_sum: pd.Series = volume.rolling(period).sum()  # type: ignore[assignment]
    vol_sum = vol_sum.replace(0, np.nan)
    cmf: pd.Series = mfv.rolling(period).sum() / vol_sum  # type: ignore[assignment]
    return cmf


def compute_bb_position(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> pd.Series:
    """Bollinger Band position 계산 (0-1 normalized).

    Args:
        close: 종가 시리즈
        period: 이동평균 기간
        std: 표준편차 배수

    Returns:
        BB position 시리즈 (0~1, close가 lower band에 있으면 0, upper band에 있으면 1)
    """
    sma = close.rolling(period).mean()
    std_val = close.rolling(period).std()
    upper: pd.Series = sma + std * std_val  # type: ignore[assignment]
    lower: pd.Series = sma - std * std_val  # type: ignore[assignment]
    denom = pd.Series(upper - lower, index=close.index).replace(0, np.nan)
    bb_pos: pd.Series = (close - lower) / denom  # type: ignore[assignment]
    return bb_pos


def compute_sma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """SMA crossover signal 계산 (fast/slow - 1).

    Args:
        close: 종가 시리즈
        fast: 빠른 SMA 기간
        slow: 느린 SMA 기간

    Returns:
        SMA cross 시리즈
    """
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()
    cross: pd.Series = fast_sma / slow_sma.replace(0, np.nan) - 1  # type: ignore[assignment]
    return cross


def compute_ema_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """EMA crossover signal 계산 (fast/slow - 1).

    Args:
        close: 종가 시리즈
        fast: 빠른 EMA 기간
        slow: 느린 EMA 기간

    Returns:
        EMA cross 시리즈
    """
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    cross: pd.Series = fast_ema / slow_ema.replace(0, np.nan) - 1  # type: ignore[assignment]
    return cross


def compute_roc(close: pd.Series, period: int) -> pd.Series:
    """ROC (Rate of Change) 계산.

    Args:
        close: 종가 시리즈
        period: 계산 기간

    Returns:
        ROC 시리즈
    """
    return close.pct_change(period)


def compute_volume_macd(
    volume: pd.Series,
    fast: int = 12,
    slow: int = 26,
) -> pd.Series:
    """Volume MACD 계산 (EMA difference).

    Args:
        volume: 거래량 시리즈
        fast: 빠른 EMA 기간
        slow: 느린 EMA 기간

    Returns:
        Volume MACD 시리즈
    """
    v_macd: pd.Series = (
        volume.ewm(span=fast, adjust=False).mean() - volume.ewm(span=slow, adjust=False).mean()
    )  # type: ignore[assignment]
    return v_macd


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """28개 기술적 feature를 모두 계산.

    Args:
        df: OHLCV DataFrame (close, high, low, volume 필수)

    Returns:
        28개 feature 컬럼을 가진 DataFrame (feat_ 접두사)

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    features: dict[str, pd.Series] = {}

    # 1-3. MACD (3 features)
    macd_line, macd_signal, macd_hist = compute_macd(close)
    features["feat_macd"] = macd_line
    features["feat_macd_signal"] = macd_signal
    features["feat_macd_hist"] = macd_hist

    # 4-5. RSI (2 features: 14-period and 7-period)
    features["feat_rsi_14"] = compute_rsi(close, period=14)
    features["feat_rsi_7"] = compute_rsi(close, period=7)

    # 6-7. CCI (2 features: 20-period and 14-period)
    features["feat_cci_20"] = compute_cci(high, low, close, period=20)
    features["feat_cci_14"] = compute_cci(high, low, close, period=14)

    # 8. Williams %R
    features["feat_williams_r_14"] = compute_williams_r(high, low, close, period=14)

    # 9-10. Stochastic (2 features)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    features["feat_stoch_k"] = stoch_k
    features["feat_stoch_d"] = stoch_d

    # 11. OBV (normalized by rolling std to make stationary)
    obv_raw = compute_obv(close, volume)
    obv_std = obv_raw.rolling(20).std().replace(0, np.nan)
    features["feat_obv_norm"] = obv_raw / obv_std

    # 12. Chaikin Money Flow
    features["feat_cmf_20"] = compute_chaikin_mf(high, low, close, volume, period=20)

    # 13-14. Bollinger Band Position (2 features: 20-period and 10-period)
    features["feat_bb_pos_20"] = compute_bb_position(close, period=20)
    features["feat_bb_pos_10"] = compute_bb_position(close, period=10)

    # 15-17. SMA Cross (3 features)
    features["feat_sma_cross_5_20"] = compute_sma_cross(close, fast=5, slow=20)
    features["feat_sma_cross_10_50"] = compute_sma_cross(close, fast=10, slow=50)
    features["feat_sma_cross_20_100"] = compute_sma_cross(close, fast=20, slow=100)

    # 18-19. EMA Cross (2 features)
    features["feat_ema_cross_5_20"] = compute_ema_cross(close, fast=5, slow=20)
    features["feat_ema_cross_10_50"] = compute_ema_cross(close, fast=10, slow=50)

    # 20-22. Rate of Change (3 features)
    features["feat_roc_5"] = compute_roc(close, period=5)
    features["feat_roc_10"] = compute_roc(close, period=10)
    features["feat_roc_21"] = compute_roc(close, period=21)

    # 23-24. ATR Ratio (2 features: ATR / close for normalization)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_28 = calculate_atr(high, low, close, period=28)
    close_safe = close.replace(0, np.nan)
    features["feat_atr_ratio_14"] = atr_14 / close_safe
    features["feat_atr_ratio_28"] = atr_28 / close_safe

    # 25. Volume MACD
    features["feat_volume_macd"] = compute_volume_macd(volume)

    # 26-27. Momentum (close change over N periods, different from ROC)
    features["feat_mom_5"] = close.diff(5) / close_safe
    features["feat_mom_21"] = close.diff(21) / close_safe

    # 28. Volatility Ratio (short vol / long vol)
    short_vol = close.pct_change().rolling(10).std()
    long_vol = close.pct_change().rolling(30).std().replace(0, np.nan)
    features["feat_vol_ratio"] = short_vol / long_vol

    return pd.DataFrame(features, index=df.index)


def preprocess(
    df: pd.DataFrame,
    config: CTRENDConfig,
) -> pd.DataFrame:
    """CTREND 전처리 (28 features + returns + vol_scalar + forward_return).

    OHLCV DataFrame에 28개 기술적 feature와 변동성 스케일러,
    forward return(training target)을 계산하여 추가합니다.

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, high, low, volume
        config: CTREND 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 또는 빈 DataFrame
    """
    # 입력 검증
    if df.empty:
        msg = "Input DataFrame is empty"
        raise ValueError(msg)

    required_cols = {"close", "high", "low", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 1. 28개 feature 계산
    feature_df = compute_all_features(result)
    for col in feature_df.columns:
        result[col] = feature_df[col]

    # 컬럼 추출
    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # 2. Returns 계산
    result["returns"] = calculate_returns(close_series, use_log=True)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 3. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. Forward return (training target)
    # close.pct_change(prediction_horizon).shift(-prediction_horizon)
    # NaN at end — this is intentional (training only uses historical data)
    result["forward_return"] = close_series.pct_change(config.prediction_horizon).shift(
        -config.prediction_horizon
    )

    # 디버그: feature 통계
    feat_cols = [c for c in result.columns if c.startswith("feat_")]
    valid_data = result[feat_cols].dropna()
    if len(valid_data) > 0:
        logger.info(
            "CTREND Features | %d features computed, %d valid rows (%.1f%%)",
            len(feat_cols),
            len(valid_data),
            len(valid_data) / len(result) * 100,
        )

    return result
