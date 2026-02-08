"""TTM Squeeze Preprocessor (Indicator Calculation).

Bollinger Bands, Keltner Channels, Squeeze 감지, Momentum,
Exit SMA, Volatility Scalar 등을 벡터화 연산으로 계산합니다.

Calculated Columns:
    - bb_upper, bb_lower: Bollinger Bands
    - kc_upper, kc_lower: Keltner Channels
    - squeeze_on: Squeeze 상태 (bool, BB inside KC)
    - momentum: close - donchian midline
    - exit_sma: 청산용 SMA
    - realized_vol: 실현 변동성 (연환산)
    - vol_scalar: 변동성 스케일러 (shift(1) 적용)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - Shift(1) Rule: vol_scalar에 적용하여 미래 참조 방지
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.strategy.ttm_squeeze.config import TtmSqueezeConfig

logger = logging.getLogger(__name__)


def calculate_bollinger_bands(
    close: pd.Series,
    period: int,
    std_dev: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands 계산.

    Args:
        close: 종가 시리즈
        period: 이동평균 기간
        std_dev: 표준편차 배수

    Returns:
        (bb_upper, bb_middle, bb_lower) 튜플
    """
    middle: pd.Series = close.rolling(window=period).mean()  # type: ignore[assignment]
    std: pd.Series = close.rolling(window=period).std()  # type: ignore[assignment]
    upper: pd.Series = middle + std_dev * std  # type: ignore[assignment]
    lower: pd.Series = middle - std_dev * std  # type: ignore[assignment]
    return upper, middle, lower


def calculate_keltner_channels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int,
    mult: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels 계산.

    EMA 기반 중심선에 ATR 배수를 더한 채널입니다.

    Args:
        close: 종가 시리즈
        high: 고가 시리즈
        low: 저가 시리즈
        period: EMA/ATR 계산 기간
        mult: ATR 배수

    Returns:
        (kc_upper, kc_middle, kc_lower) 튜플
    """
    # KC 중심선: EMA
    kc_middle: pd.Series = close.ewm(span=period, adjust=False).mean()  # type: ignore[assignment]

    # True Range 계산
    prev_close: pd.Series = close.shift(1)  # type: ignore[assignment]
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR: EMA of True Range
    atr: pd.Series = true_range.ewm(span=period, adjust=False).mean()  # type: ignore[assignment]

    kc_upper: pd.Series = kc_middle + mult * atr  # type: ignore[assignment]
    kc_lower: pd.Series = kc_middle - mult * atr  # type: ignore[assignment]

    return kc_upper, kc_middle, kc_lower


def calculate_squeeze(
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    kc_upper: pd.Series,
    kc_lower: pd.Series,
) -> pd.Series:
    """Squeeze 상태 감지.

    BB가 KC 안에 있으면 squeeze ON (저변동성 수축 상태).

    Args:
        bb_upper: BB 상단
        bb_lower: BB 하단
        kc_upper: KC 상단
        kc_lower: KC 하단

    Returns:
        bool Series (True = squeeze ON)
    """
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    return pd.Series(squeeze_on, index=bb_upper.index, name="squeeze_on", dtype=bool)


def calculate_momentum(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int,
) -> pd.Series:
    """Momentum 계산 (close - donchian midline).

    Donchian midline = (highest_high + lowest_low) / 2 over period.
    Momentum = close - midline.

    Args:
        close: 종가 시리즈
        high: 고가 시리즈
        low: 저가 시리즈
        period: lookback 기간

    Returns:
        momentum 시리즈
    """
    highest: pd.Series = high.rolling(window=period).max()  # type: ignore[assignment]
    lowest: pd.Series = low.rolling(window=period).min()  # type: ignore[assignment]
    midline: pd.Series = (highest + lowest) / 2  # type: ignore[assignment]
    momentum: pd.Series = close - midline  # type: ignore[assignment]
    return pd.Series(momentum, index=close.index, name="momentum")


def calculate_exit_sma(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """청산용 SMA 계산.

    Args:
        close: 종가 시리즈
        period: SMA 기간

    Returns:
        SMA 시리즈
    """
    sma: pd.Series = close.rolling(window=period).mean()  # type: ignore[assignment]
    return pd.Series(sma, index=close.index, name="exit_sma")


def calculate_realized_volatility(
    close: pd.Series,
    window: int,
    annualization_factor: float,
) -> pd.Series:
    """실현 변동성 계산 (로그 수익률 기반, 연환산).

    Args:
        close: 종가 시리즈
        window: rolling 윈도우
        annualization_factor: 연환산 계수

    Returns:
        연환산 실현 변동성 시리즈
    """
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    realized_vol: pd.Series = rolling_std * np.sqrt(annualization_factor)  # type: ignore[assignment]
    return pd.Series(realized_vol, index=close.index, name="realized_vol")


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float,
) -> pd.Series:
    """변동성 스케일러 계산 (shift(1) 적용).

    목표 변동성 대비 실현 변동성의 비율을 계산합니다.
    Shift(1)을 적용하여 미래 참조 편향을 방지합니다.

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 시리즈 (shift(1) 적용됨)
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    scalar = vol_target / clamped_vol
    shifted: pd.Series = scalar.shift(1)  # type: ignore[assignment]
    return pd.Series(shifted, index=realized_vol.index, name="vol_scalar")


def preprocess(
    df: pd.DataFrame,
    config: TtmSqueezeConfig,
) -> pd.DataFrame:
    """TTM Squeeze 전처리 (지표 계산).

    OHLCV DataFrame에 TTM Squeeze 전략에 필요한 기술적 지표를 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - bb_upper, bb_lower: Bollinger Bands
        - kc_upper, kc_lower: Keltner Channels
        - squeeze_on: bool (BB inside KC)
        - momentum: close - donchian midline
        - exit_sma: 청산용 SMA
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러 (shift(1) 적용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close
        config: TTM Squeeze 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    # 입력 검증
    required_cols = {"open", "high", "low", "close"}
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
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. Bollinger Bands 계산
    bb_upper, _bb_middle, bb_lower = calculate_bollinger_bands(
        close_series,
        period=config.bb_period,
        std_dev=config.bb_std,
    )
    result["bb_upper"] = bb_upper
    result["bb_lower"] = bb_lower

    # 2. Keltner Channels 계산
    kc_upper, _kc_middle, kc_lower = calculate_keltner_channels(
        close_series,
        high_series,
        low_series,
        period=config.kc_period,
        mult=config.kc_mult,
    )
    result["kc_upper"] = kc_upper
    result["kc_lower"] = kc_lower

    # 3. Squeeze 감지
    result["squeeze_on"] = calculate_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

    # 4. Momentum 계산
    result["momentum"] = calculate_momentum(
        close_series,
        high_series,
        low_series,
        period=config.mom_period,
    )

    # 5. Exit SMA 계산
    result["exit_sma"] = calculate_exit_sma(
        close_series,
        period=config.exit_sma_period,
    )

    # 6. 실현 변동성 계산
    result["realized_vol"] = calculate_realized_volatility(
        close_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 7. 변동성 스케일러 계산 (shift(1) 포함)
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        squeeze_pct = valid_data["squeeze_on"].mean() * 100
        mom_min = valid_data["momentum"].min()
        mom_max = valid_data["momentum"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "TTM Squeeze Indicators | Squeeze: %.1f%%, Momentum: [%.2f, %.2f], Vol Scalar: [%.2f, %.2f]",
            squeeze_pct,
            mom_min,
            mom_max,
            vs_min,
            vs_max,
        )

    return result
