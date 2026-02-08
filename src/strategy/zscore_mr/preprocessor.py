"""Z-Score Mean Reversion Preprocessor (Indicator Calculation).

동적 lookback z-score, 변동성 레짐, ATR 등 기술적 지표를 벡터화 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.zscore_mr.config import ZScoreMRConfig

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
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수

    Returns:
        연환산 변동성 시리즈
    """
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    return rolling_std * np.sqrt(annualization_factor)


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산 (vol_target / realized_vol).

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프

    Returns:
        변동성 스케일러 시리즈
    """
    clamped_vol = realized_vol.clip(lower=min_volatility)
    return vol_target / clamped_vol


def calculate_vol_regime(
    returns: pd.Series,
    vol_regime_lookback: int,
    vol_rank_lookback: int,
) -> pd.Series:
    """변동성 레짐 계산 (percentile rank).

    수익률의 rolling std를 계산한 후, 그 값의 percentile rank를 반환합니다.
    rank가 높을수록 현재 변동성이 과거 대비 높다는 의미입니다.

    Args:
        returns: 수익률 시리즈
        vol_regime_lookback: 변동성 추정 윈도우
        vol_rank_lookback: percentile rank 윈도우

    Returns:
        변동성 percentile rank 시리즈 (0~1 범위)
    """
    vol = returns.rolling(vol_regime_lookback, min_periods=vol_regime_lookback).std()
    vol_pct = vol.rolling(
        vol_rank_lookback, min_periods=min(vol_rank_lookback, 60)
    ).rank(pct=True)
    return pd.Series(vol_pct, index=returns.index, name="vol_regime")


def calculate_zscore(
    close: pd.Series,
    lookback: int,
) -> pd.Series:
    """Z-score 계산.

    z = (close - rolling_mean) / rolling_std

    Args:
        close: 종가 시리즈
        lookback: Rolling lookback 기간

    Returns:
        Z-score 시리즈
    """
    mean = close.rolling(lookback, min_periods=lookback).mean()
    std = close.rolling(lookback, min_periods=lookback).std()
    z: pd.Series = (close - mean) / std.replace(0, np.nan)  # type: ignore[assignment]
    return pd.Series(z, index=close.index, name=f"zscore_{lookback}")


def calculate_adaptive_zscore(
    close: pd.Series,
    returns: pd.Series,
    short_lookback: int,
    long_lookback: int,
    vol_regime_lookback: int,
    vol_rank_lookback: int,
    high_vol_percentile: float,
) -> tuple[pd.Series, pd.Series]:
    """적응적 z-score 계산 (변동성 레짐 기반 lookback 전환).

    고변동성 레짐에서는 short_lookback, 저변동성 레짐에서는 long_lookback을 사용합니다.

    Args:
        close: 종가 시리즈
        returns: 수익률 시리즈
        short_lookback: 단기 lookback
        long_lookback: 장기 lookback
        vol_regime_lookback: 변동성 추정 윈도우
        vol_rank_lookback: percentile rank 윈도우
        high_vol_percentile: 고변동성 판단 임계값

    Returns:
        (adaptive_zscore, vol_regime) 튜플
    """
    vol_pct = calculate_vol_regime(returns, vol_regime_lookback, vol_rank_lookback)
    z_short = calculate_zscore(close, short_lookback)
    z_long = calculate_zscore(close, long_lookback)

    high_vol = vol_pct > high_vol_percentile
    z_adaptive = pd.Series(
        np.where(high_vol, z_short, z_long),
        index=close.index,
        name="zscore",
    )

    return z_adaptive, vol_pct


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """ATR (Average True Range) 계산.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: ATR 기간

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
        드로다운 시리즈 (항상 0 이하)
    """
    rolling_max = close.expanding().max()
    drawdown: pd.Series = (close - rolling_max) / rolling_max  # type: ignore[assignment]
    return pd.Series(drawdown, index=close.index, name="drawdown")


def preprocess(
    df: pd.DataFrame,
    config: ZScoreMRConfig,
) -> pd.DataFrame:
    """Z-Score MR 전처리 (지표 계산).

    OHLCV DataFrame에 평균회귀 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - returns: 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - zscore: 적응적 z-score (변동성 레짐 기반)
        - vol_regime: 변동성 percentile rank
        - atr: Average True Range
        - drawdown: 최고점 대비 하락률

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Z-Score MR 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Decimal 타입 -> float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률
    result["returns"] = calculate_returns(close, use_log=config.use_log_returns)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 4. 적응적 z-score + 변동성 레짐
    z_adaptive, vol_regime = calculate_adaptive_zscore(
        close,
        returns_series,
        short_lookback=config.short_lookback,
        long_lookback=config.long_lookback,
        vol_regime_lookback=config.vol_regime_lookback,
        vol_rank_lookback=config.vol_rank_lookback,
        high_vol_percentile=config.high_vol_percentile,
    )
    result["zscore"] = z_adaptive
    result["vol_regime"] = vol_regime

    # 5. ATR
    result["atr"] = calculate_atr(high, low, close, config.atr_period)

    # 6. 드로다운 (HEDGE_ONLY 모드용)
    result["drawdown"] = calculate_drawdown(close)

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        z_mean = valid_data["zscore"].mean()
        z_std = valid_data["zscore"].std()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "Z-Score MR Indicators | Z Mean: %.2f, Z Std: %.2f, Vol Scalar: [%.2f, %.2f]",
            z_mean,
            z_std,
            vs_min,
            vs_max,
        )

    return result
