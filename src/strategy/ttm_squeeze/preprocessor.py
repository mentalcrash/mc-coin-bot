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

import pandas as pd

from src.market.indicators import (
    bollinger_bands,
    log_returns,
    realized_volatility,
    squeeze_detect,
)

if TYPE_CHECKING:
    from src.strategy.ttm_squeeze.config import TtmSqueezeConfig

logger = logging.getLogger(__name__)


def calculate_keltner_channels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int,
    mult: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels 계산.

    EMA 기반 중심선에 ATR 배수를 더한 채널입니다.
    NOTE: indicators.keltner_channels는 ema_period/atr_period를 별도로 받지만
    이 전략은 period를 EMA/ATR 모두에 공유하므로 로컬 구현을 유지합니다.

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
    atr_val: pd.Series = true_range.ewm(span=period, adjust=False).mean()  # type: ignore[assignment]

    kc_upper: pd.Series = kc_middle + mult * atr_val  # type: ignore[assignment]
    kc_lower: pd.Series = kc_middle - mult * atr_val  # type: ignore[assignment]

    return kc_upper, kc_middle, kc_lower


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
    bb_upper, _bb_middle, bb_lower = bollinger_bands(
        close_series,
        period=config.bb_period,
        std_dev=config.bb_std,
    )
    result["bb_upper"] = bb_upper
    result["bb_lower"] = bb_lower

    # 2. Keltner Channels 계산 (로컬 구현 유지: period 공유 패턴)
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
    result["squeeze_on"] = squeeze_detect(bb_upper, bb_lower, kc_upper, kc_lower)

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

    # 6. 실현 변동성 계산 (log returns → realized_volatility)
    returns_series = log_returns(close_series)
    rv = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = rv

    # 7. 변동성 스케일러 계산 (shift(1) 적용: 미래 참조 방지)
    clamped_vol = rv.clip(lower=config.min_volatility)
    scalar = config.vol_target / clamped_vol
    shifted: pd.Series = scalar.shift(1)  # type: ignore[assignment]
    result["vol_scalar"] = pd.Series(shifted, index=rv.index, name="vol_scalar")

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
