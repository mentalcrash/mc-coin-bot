"""VPIN Flow Toxicity Preprocessor (Indicator Calculation).

BVC(Bulk Volume Classification)와 VPIN을 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)
from src.strategy.vpin_flow.config import VPINFlowConfig

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    config: VPINFlowConfig,
) -> pd.DataFrame:
    """VPIN Flow Toxicity 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - v_buy: 추정 매수 거래량
        - v_sell: 추정 매도 거래량
        - vpin: Volume-Synchronized PIN
        - flow_direction: 플로우 방향 (sign of buy-sell imbalance)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: VPIN Flow 설정

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
    open_series: pd.Series = result["open"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. BVC (Bulk Volume Classification)
    # buy_pct = norm.cdf((close - open) / (high - low + eps))
    range_diff = high_series - low_series + 1e-10
    z_score = (close_series - open_series) / range_diff
    buy_pct = pd.Series(norm.cdf(z_score), index=df.index)

    v_buy = volume_series * buy_pct
    v_sell = volume_series * (1 - buy_pct)
    result["v_buy"] = v_buy
    result["v_sell"] = v_sell

    # 3. Order imbalance & VPIN
    order_imbalance = (v_buy - v_sell).abs()
    rolling_imbalance = order_imbalance.rolling(
        window=config.n_buckets,
        min_periods=config.n_buckets,
    ).sum()
    rolling_volume = volume_series.rolling(
        window=config.n_buckets,
        min_periods=config.n_buckets,
    ).sum()
    rolling_volume_safe: pd.Series = rolling_volume.replace(0, np.nan)  # type: ignore[assignment]
    result["vpin"] = rolling_imbalance / rolling_volume_safe

    # 4. Flow direction
    buy_sum = v_buy.rolling(
        window=config.flow_direction_period,
        min_periods=config.flow_direction_period,
    ).sum()
    sell_sum = v_sell.rolling(
        window=config.flow_direction_period,
        min_periods=config.flow_direction_period,
    ).sum()
    result["flow_direction"] = np.sign(buy_sum - sell_sum)

    # 5. 실현 변동성
    realized_vol = realized_volatility(
        returns_series,
        window=config.flow_direction_period,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 6. 변동성 스케일러
    result["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. ATR 계산
    result["atr"] = atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 8. 드로다운 계산
    result["drawdown"] = drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vpin_mean = valid_data["vpin"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "VPIN-Flow Indicators | Avg VPIN: %.4f, Avg Vol Scalar: %.4f",
            vpin_mean,
            vs_mean,
        )

    return result
