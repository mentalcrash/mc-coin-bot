"""Enhanced VW-TSMOM Preprocessor (Indicator Calculation).

이 모듈은 Enhanced VW-TSMOM 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
기존 TSMOM의 log1p(volume) 대신 볼륨 비율 정규화(volume_ratio)를 사용합니다.

Key Difference from TSMOM:
    - TSMOM: log1p(volume) 가중 -> 거래량 이상치 압축
    - Enhanced: volume / rolling_mean(volume) -> 상대적 볼륨 비율 가중

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.enhanced_tsmom.config import EnhancedTSMOMConfig

logger = logging.getLogger(__name__)


def calculate_returns(
    close: pd.Series,
    use_log: bool = True,
) -> pd.Series:
    """수익률 계산 (로그 또는 단순).

    Args:
        close: 종가 시리즈
        use_log: True면 로그 수익률, False면 단순 수익률

    Returns:
        수익률 시리즈 (첫 값은 NaN)

    Example:
        >>> returns = calculate_returns(df["close"], use_log=True)
    """
    if len(close) == 0:
        msg = "Empty Series provided"
        raise ValueError(msg)

    if use_log:
        price_ratio = close / close.shift(1)
        return pd.Series(np.log(price_ratio), index=close.index, name="returns")
    return close.pct_change()


def calculate_realized_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: float = 365.0,
    min_periods: int | None = None,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Rolling standard deviation을 사용하여 실현 변동성을 계산합니다.

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수 (일봉: 365)
        min_periods: 최소 관측치 수 (None이면 window 사용)

    Returns:
        연환산 변동성 시리즈
    """
    if min_periods is None:
        min_periods = window

    rolling_std = returns.rolling(window=window, min_periods=min_periods).std()
    return rolling_std * np.sqrt(annualization_factor)


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산.

    목표 변동성 대비 실현 변동성의 비율을 계산합니다.

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)

    Returns:
        변동성 스케일러 시리즈
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    return vol_target / clamped_vol


def calculate_enhanced_vw_momentum(
    returns: pd.Series,
    volume: pd.Series,
    lookback: int,
    volume_lookback: int = 20,
    volume_clip_max: float = 5.0,
) -> pd.Series:
    """볼륨 비율 정규화 기반 모멘텀 계산 (Enhanced VW-TSMOM 핵심).

    기존 TSMOM의 log1p(volume) 방식 대신, 이동평균 대비 상대적 거래량 비율을
    가중치로 사용합니다. 이를 통해 시장 구조 변화(볼륨 레벨 변화)에 더
    강건한 모멘텀 시그널을 생성합니다.

    Formula:
        vol_ratio = volume / volume.rolling(volume_lookback).mean()
        vol_ratio = vol_ratio.clip(upper=clip_max)
        weighted_return = return * vol_ratio
        momentum = weighted_return.rolling(lookback).sum()

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        lookback: 모멘텀 합산 기간
        volume_lookback: 거래량 이동평균 윈도우
        volume_clip_max: 거래량 비율 클리핑 상한

    Returns:
        Enhanced volume-weighted 모멘텀 시리즈
    """
    vol_mean: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=volume_lookback, min_periods=volume_lookback
    ).mean()
    vol_ratio = (volume / vol_mean.replace(0, np.nan)).clip(upper=volume_clip_max)
    weighted_returns = returns * vol_ratio
    momentum = weighted_returns.rolling(window=lookback, min_periods=lookback).sum()
    return pd.Series(momentum, index=returns.index, name="evw_momentum")


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR (Average True Range) 계산.

    Wilder's smoothing(EWM)을 사용한 True Range의 지수이동평균입니다.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: 계산 기간 (기본 14)

    Returns:
        ATR 시리즈
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.Series(atr, index=close.index, name="atr")


def calculate_drawdown(close: pd.Series) -> pd.Series:
    """롤링 최고점 대비 드로다운 계산.

    Args:
        close: 종가 시리즈

    Returns:
        드로다운 시리즈 (항상 0 이하, 예: -0.15 = -15%)
    """
    rolling_max = close.expanding().max()
    drawdown: pd.Series = (close - rolling_max) / rolling_max  # type: ignore[assignment]
    return pd.Series(drawdown, index=close.index, name="drawdown")


def preprocess(
    df: pd.DataFrame,
    config: EnhancedTSMOMConfig,
) -> pd.DataFrame:
    """Enhanced VW-TSMOM 전처리 (순수 지표 계산).

    OHLCV DataFrame에 Enhanced VW-TSMOM 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - evw_momentum: 볼륨 비율 정규화 모멘텀
        - vol_scalar: 변동성 스케일러
        - drawdown: 최고점 대비 드로다운
        - atr: ATR (Average True Range)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: Enhanced TSMOM 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"open", "high", "low", "close", "volume"}
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

    # 컬럼 추출 (명시적 Series 타입)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. Enhanced Volume-Weighted 모멘텀 계산 (핵심 차이점)
    result["evw_momentum"] = calculate_enhanced_vw_momentum(
        returns_series,
        volume_series,
        lookback=config.lookback,
        volume_lookback=config.volume_lookback,
        volume_clip_max=config.volume_clip_max,
    )

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 6. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(high_series, low_series, close_series, period=config.atr_period)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        mom_min = valid_data["evw_momentum"].min()
        mom_max = valid_data["evw_momentum"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "Enhanced VW-TSMOM Indicators | Momentum: [%.4f, %.4f], Vol Scalar: [%.2f, %.2f]",
            mom_min,
            mom_max,
            vs_min,
            vs_max,
        )

    return result
