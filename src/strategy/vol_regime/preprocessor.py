"""Vol-Regime Adaptive Preprocessor (Indicator Calculation).

이 모듈은 Vol-Regime Adaptive 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
변동성 regime 판별 및 regime별 모멘텀 시그널 계산을 수행합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr as calculate_atr,
    drawdown as calculate_drawdown,
    log_returns,
    realized_volatility as calculate_realized_volatility,
    simple_returns,
    vol_regime as calculate_vol_regime,
    volatility_scalar as calculate_volatility_scalar,
    volume_weighted_returns,
)
from src.strategy.vol_regime.config import VolRegimeConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward-compat re-exports (used by ~25 downstream strategies)
# ---------------------------------------------------------------------------

__all__ = [
    "calculate_atr",
    "calculate_drawdown",
    "calculate_realized_volatility",
    "calculate_regime_strength",
    "calculate_returns",
    "calculate_vol_regime",
    "calculate_volatility_scalar",
    "calculate_vw_momentum",
    "preprocess",
]


def calculate_returns(
    close: pd.Series,
    use_log: bool = True,
) -> pd.Series:
    """수익률 계산 (로그 또는 단순). Backward-compat wrapper."""
    return log_returns(close) if use_log else simple_returns(close)


def calculate_vw_momentum(
    returns: pd.Series,
    volume: pd.Series,
    lookback: int,
) -> pd.Series:
    """거래량 가중 모멘텀 계산 (TSMOM 패턴 재사용).

    각 기간의 수익률에 로그 거래량을 가중하여 평균합니다.
    로그 스케일링으로 거래량 이상치의 과도한 영향력을 압축합니다.

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        lookback: 모멘텀 계산 기간

    Returns:
        거래량 가중 모멘텀 시리즈
    """
    return volume_weighted_returns(returns, volume, window=lookback)


def calculate_regime_strength(
    returns: pd.Series,
    volume: pd.Series,
    close: pd.Series,
    vol_pct: pd.Series,
    config: VolRegimeConfig,
) -> pd.Series:
    """Regime별 모멘텀 강도 계산 및 선택.

    3개의 regime(high/normal/low)별 모멘텀과 vol scalar를 계산하고,
    현재 vol_pct에 따라 적절한 regime의 강도를 선택합니다.

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        close: 종가 시리즈
        vol_pct: 변동성 percentile rank 시리즈
        config: Vol-Regime 설정

    Returns:
        선택된 regime의 strength 시리즈
    """
    # 공통 변동성 계산 (vol scalar 산출용)
    high_vol = returns.rolling(
        config.vol_lookback, min_periods=config.vol_lookback
    ).std() * np.sqrt(config.annualization_factor)
    clamped_vol = high_vol.clip(lower=config.min_volatility)

    # High vol regime: 보수적 (긴 lookback, 낮은 vol target)
    high_mom = calculate_vw_momentum(returns, volume, config.high_vol_lookback)
    high_scalar = config.high_vol_target / clamped_vol
    high_strength = np.sign(high_mom) * high_scalar

    # Normal regime: 중간
    normal_mom = calculate_vw_momentum(returns, volume, config.normal_lookback)
    normal_scalar = config.normal_vol_target / clamped_vol
    normal_strength = np.sign(normal_mom) * normal_scalar

    # Low vol regime: 공격적 (짧은 lookback, 높은 vol target)
    low_mom = calculate_vw_momentum(returns, volume, config.low_vol_lookback)
    low_scalar = config.low_vol_target / clamped_vol
    low_strength = np.sign(low_mom) * low_scalar

    # Regime에 따라 선택
    strength = pd.Series(
        np.where(
            vol_pct > config.high_vol_threshold,
            high_strength,
            np.where(vol_pct < config.low_vol_threshold, low_strength, normal_strength),
        ),
        index=close.index,
        name="regime_strength",
    )
    return strength


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    config: VolRegimeConfig,
) -> pd.DataFrame:
    """Vol-Regime Adaptive 전처리 (지표 계산).

    OHLCV DataFrame에 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vol_regime: 변동성 percentile rank (0~1)
        - regime_strength: regime별 모멘텀 * vol scalar
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: open, high, low, close, volume
        config: Vol-Regime 설정

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

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
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
        window=config.vol_lookback,
        annualization_factor=config.annualization_factor,
    )

    # 3. 변동성 regime 계산 (percentile rank)
    result["vol_regime"] = calculate_vol_regime(
        returns_series,
        vol_lookback=config.vol_lookback,
        vol_rank_lookback=config.vol_rank_lookback,
        annualization_factor=config.annualization_factor,
    )
    vol_regime_series: pd.Series = result["vol_regime"]  # type: ignore[assignment]

    # 4. Regime별 strength 계산
    result["regime_strength"] = calculate_regime_strength(
        returns_series,
        volume_series,
        close_series,
        vol_regime_series,
        config,
    )

    # 5. ATR 계산 (Trailing Stop용)
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 6. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rs_min = valid_data["regime_strength"].min()
        rs_max = valid_data["regime_strength"].max()
        vr_mean = valid_data["vol_regime"].mean()
        logger.info(
            "Vol-Regime Indicators | Regime Strength: [%.4f, %.4f], Avg Vol Regime: %.4f",
            rs_min,
            rs_max,
            vr_mean,
        )

    return result
