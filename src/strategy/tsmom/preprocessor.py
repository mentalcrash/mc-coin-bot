"""VW-TSMOM Preprocessor (Indicator Calculation).

이 모듈은 VW-TSMOM 전략에 필요한 지표를 벡터화된 연산으로 계산합니다.
백테스팅과 라이브 트레이딩 모두에서 동일한 코드를 사용합니다.

Pure TSMOM + Vol Target 구현:
    1. 거래량 가중 모멘텀 (vw_momentum)
    2. 실현 변동성 (realized_vol)
    3. 변동성 스케일러 (vol_scalar = vol_target / realized_vol)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import pandas as pd

from src.market.indicators import (
    adx as calculate_adx,
    atr as calculate_atr,
    drawdown as calculate_drawdown,
    log_returns,
    realized_volatility as calculate_realized_volatility,
    simple_returns,
    volatility_scalar as calculate_volatility_scalar,
    volume_weighted_returns as calculate_volume_weighted_returns,
)
from src.strategy.tsmom.config import TSMOMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compat re-exports (used by ~30 downstream strategies)
# ---------------------------------------------------------------------------


def calculate_returns(
    close: pd.Series,
    use_log: bool = True,
) -> pd.Series:
    """수익률 계산 (로그 또는 단순). Backward-compat wrapper."""
    return log_returns(close) if use_log else simple_returns(close)


# ---------------------------------------------------------------------------
# TSMOM-specific: Volume-Weighted Momentum
# ---------------------------------------------------------------------------


def calculate_vw_momentum(
    returns: pd.Series,
    volume: pd.Series,
    lookback: int,
    smoothing: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """거래량 가중 모멘텀 계산.

    VW-TSMOM의 핵심 지표입니다. 거래량 가중 수익률의 누적 합계로
    모멘텀을 측정합니다.

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        lookback: 모멘텀 계산 기간
        smoothing: EMA 스무딩 윈도우 (선택적)
        min_periods: 최소 관측치 수

    Returns:
        모멘텀 시리즈
    """
    vw_returns: pd.Series = calculate_volume_weighted_returns(
        returns, volume, lookback, min_periods
    )
    if smoothing is not None and smoothing > 1:
        vw_returns = vw_returns.ewm(span=smoothing, adjust=False).mean()  # type: ignore[assignment]
    return vw_returns


# ---------------------------------------------------------------------------
# preprocess / preprocess_live
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    config: TSMOMConfig,
) -> pd.DataFrame:
    """VW-TSMOM 전처리 (순수 지표 계산).

    OHLCV DataFrame에 VW-TSMOM 전략에 필요한 기술적 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Note:
        이 모듈은 순수 지표 계산만 담당합니다.
        시그널 생성(scaled_momentum 등)은 signal.py에서 처리됩니다.
        레버리지 클램핑은 PortfolioManagerConfig에서 처리됩니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vw_momentum: 거래량 가중 모멘텀
        - vol_scalar: 변동성 스케일러

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, volume
        config: TSMOM 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    required_cols = {"close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

    # 1. 수익률 계산
    result["returns"] = calculate_returns(close_series, use_log=config.use_log_returns)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. 실현 변동성 계산 (연환산)
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 3. 거래량 가중 모멘텀 계산
    result["vw_momentum"] = calculate_vw_momentum(
        returns_series,
        volume_series,
        lookback=config.lookback,
        smoothing=config.momentum_smoothing,
    )

    # 4. 변동성 스케일러 계산
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. 드로다운 계산 (헤지 숏 모드용)
    result["drawdown"] = calculate_drawdown(close_series)

    # 6. ATR 계산 (Trailing Stop용 — 항상 계산)
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]
    result["atr"] = calculate_atr(high_series, low_series, close_series)

    # 7. ADX 계산 (횡보장 필터용)
    if config.use_sideways_filter:
        result["adx"] = calculate_adx(
            high_series,
            low_series,
            close_series,
            period=config.adx_period,
        )

    # 디버그: 지표 통계 (NaN 제외)
    valid_data = result.dropna()
    if len(valid_data) > 0:
        mom_min = valid_data["vw_momentum"].min()
        mom_max = valid_data["vw_momentum"].max()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "📊 VW-TSMOM Indicators | Momentum: [%.4f, %.4f], Vol Scalar: [%.2f, %.2f]",
            mom_min,
            mom_max,
            vs_min,
            vs_max,
        )
        price_change = (result["close"].iloc[-1] / result["close"].iloc[0] - 1) * 100
        avg_momentum = valid_data["vw_momentum"].mean()
        aligned = (price_change > 0 and avg_momentum > 0) or (price_change < 0 and avg_momentum < 0)
        status = "✅ Aligned" if aligned else "⚠️ Diverged"
        logger.info(
            "🎯 Direction Check | Price Change: %+.2f%%, Avg Momentum: %+.4f (%s)",
            price_change,
            avg_momentum,
            status,
        )

    return result


def preprocess_live(
    buffer: pd.DataFrame,
    config: TSMOMConfig,
    max_rows: int = 200,
) -> pd.DataFrame:
    """라이브 트레이딩용 전처리 (버퍼 기반).

    Args:
        buffer: 최근 캔들 버퍼 (최신이 마지막)
        config: TSMOM 설정
        max_rows: 최대 버퍼 크기

    Returns:
        전처리된 버퍼 (마지막 행이 최신 시그널)
    """
    if len(buffer) > max_rows:
        buffer = buffer.tail(max_rows)
    return preprocess(buffer, config)
