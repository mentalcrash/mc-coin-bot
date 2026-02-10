"""Flow Imbalance Preprocessor (Indicator Calculation).

BVC, OFI, VPIN proxy 지표를 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.flow_imbalance.config import FlowImbalanceConfig
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
    config: FlowImbalanceConfig,
) -> pd.DataFrame:
    """Flow Imbalance 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - buy_ratio: BVC 매수 비율 (close - low) / (high - low)
        - buy_vol: 매수 볼륨 (volume * buy_ratio)
        - sell_vol: 매도 볼륨 (volume * (1 - buy_ratio))
        - ofi: Order Flow Imbalance (rolling normalized)
        - vpin_proxy: VPIN proxy (buy_ratio rolling std)
        - realized_vol: 실현 변동성
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수, 1H freq)
        config: Flow Imbalance 설정

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
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. BVC: Buy ratio = (close - low) / (high - low)
    hl_range = (high_series - low_series).clip(lower=1e-10)
    buy_ratio = ((close_series - low_series) / hl_range).clip(0, 1)
    result["buy_ratio"] = buy_ratio

    # 3. Buy/Sell volume
    result["buy_vol"] = volume_series * buy_ratio
    result["sell_vol"] = volume_series * (1 - buy_ratio)

    buy_vol_series: pd.Series = result["buy_vol"]  # type: ignore[assignment]
    sell_vol_series: pd.Series = result["sell_vol"]  # type: ignore[assignment]

    # 4. OFI: Order Flow Imbalance (normalized)
    net_flow = (
        (buy_vol_series - sell_vol_series)
        .rolling(
            window=config.ofi_window,
            min_periods=config.ofi_window,
        )
        .sum()
    )
    total_vol = volume_series.rolling(
        window=config.ofi_window,
        min_periods=config.ofi_window,
    ).sum()
    total_vol_safe = pd.Series(total_vol).replace(0, np.nan)
    result["ofi"] = net_flow / total_vol_safe

    # 5. VPIN proxy: buy_ratio rolling std
    result["vpin_proxy"] = buy_ratio.rolling(
        window=config.vpin_window,
        min_periods=config.vpin_window,
    ).std()

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
    valid_data = result.dropna(subset=["ofi", "vpin_proxy"])
    if len(valid_data) > 0:
        ofi_mean = valid_data["ofi"].mean()
        vpin_mean = valid_data["vpin_proxy"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "Flow-Imbalance Indicators | Avg OFI: %.4f, Avg VPIN: %.4f, Avg Vol Scalar: %.4f",
            ofi_mean,
            vpin_mean,
            vs_mean,
        )

    return result
