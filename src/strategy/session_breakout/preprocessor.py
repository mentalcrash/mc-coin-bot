"""Session Breakout Preprocessor (Indicator Calculation).

Asian session (00-08 UTC)의 high/low range 및 percentile을 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import pandas as pd

from src.strategy.session_breakout.config import SessionBreakoutConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: SessionBreakoutConfig,
) -> pd.DataFrame:
    """Session Breakout 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - is_asian: Asian session 여부 (boolean)
        - is_trade_window: 거래 허용 구간 여부 (boolean)
        - is_exit_hour: 강제 청산 시각 여부 (boolean)
        - asian_high: 당일 Asian session 최고가 (ffill)
        - asian_low: 당일 Asian session 최저가 (ffill)
        - asian_range: asian_high - asian_low
        - range_pctl: asian_range의 rolling percentile
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수, 1H freq)
        config: Session Breakout 설정

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
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Session 플래그 (hour 기반)
    hour = result.index.hour  # type: ignore[union-attr]
    result["is_asian"] = (hour >= config.asian_start_hour) & (hour < config.asian_end_hour)
    result["is_trade_window"] = (hour >= config.asian_end_hour) & (hour < config.trade_end_hour)
    result["is_exit_hour"] = hour == config.exit_hour

    # 3. Asian High/Low (당일 기준 groupby)
    date_group = result.index.date  # type: ignore[union-attr]
    is_asian: pd.Series = result["is_asian"]  # type: ignore[assignment]

    asian_high_raw = high_series.where(is_asian).groupby(date_group).transform("max")
    asian_low_raw = low_series.where(is_asian).groupby(date_group).transform("min")

    # ffill: Asian session 종료 후에도 해당 일의 값 유지
    result["asian_high"] = asian_high_raw.ffill()
    result["asian_low"] = asian_low_raw.ffill()

    # 4. Asian Range
    asian_high_series: pd.Series = result["asian_high"]  # type: ignore[assignment]
    asian_low_series: pd.Series = result["asian_low"]  # type: ignore[assignment]
    asian_range = (asian_high_series - asian_low_series).clip(lower=0)
    result["asian_range"] = asian_range

    # 5. Range Percentile (rolling rank)
    result["range_pctl"] = (
        asian_range.rolling(
            window=config.range_pctl_window,
            min_periods=config.range_pctl_window,
        ).rank(pct=True)
        * 100
    )

    # 6. 실현 변동성
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=max(24, config.atr_period),
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 7. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. ATR 계산
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 9. 드로다운 계산
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna(subset=["range_pctl"])
    if len(valid_data) > 0:
        rp_mean = valid_data["range_pctl"].mean()
        ar_mean = valid_data["asian_range"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        msg = "Session-Breakout Indicators | Avg Range Pctl: %.1f, Avg Asian Range: %.4f, Avg Vol Scalar: %.4f"
        logger.info(msg, rp_mean, ar_mean, vs_mean)

    return result
