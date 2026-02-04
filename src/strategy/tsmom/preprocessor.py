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

import numpy as np
import pandas as pd

from src.strategy.tsmom.config import TSMOMConfig

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
        # 로그 수익률: ln(P_t / P_{t-1})
        price_ratio = close / close.shift(1)
        return pd.Series(np.log(price_ratio), index=close.index, name="returns")
    # 단순 수익률: (P_t - P_{t-1}) / P_{t-1}
    return close.pct_change()


def calculate_realized_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: float = 365.0,
    min_periods: int | None = None,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Rolling standard deviation을 사용하여 실현 변동성을 계산합니다.
    결과는 연환산되어 반환됩니다.

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수 (일봉: 365)
        min_periods: 최소 관측치 수 (None이면 window 사용)

    Returns:
        연환산 변동성 시리즈

    Example:
        >>> vol = calculate_realized_volatility(returns, window=30)
    """
    if min_periods is None:
        min_periods = window

    # Rolling 표준편차 계산
    rolling_std = returns.rolling(window=window, min_periods=min_periods).std()

    # 연환산: vol_annual = vol_period * sqrt(periods_per_year)
    return rolling_std * np.sqrt(annualization_factor)


def calculate_volume_weighted_returns(
    returns: pd.Series,
    volume: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """거래량 가중 수익률 계산 (로그 스케일링 적용).

    각 기간의 수익률에 로그 거래량을 가중하여 평균합니다.
    로그 스케일링으로 거래량 이상치(패닉 셀링 등)의 과도한 영향력을 압축합니다.

    Log-Volume Scaling:
        - 거래량 100배 → 가중치 ln(100) ≈ 4.6배 (100배가 아님)
        - 패닉 셀링 한 방에 전체 추세가 뒤집히는 것을 방지

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        window: Rolling 윈도우 크기
        min_periods: 최소 관측치 수

    Returns:
        거래량 가중 수익률 시리즈 (로그 스케일링 적용)

    Example:
        >>> vw_returns = calculate_volume_weighted_returns(
        ...     df["returns"], df["volume"], window=30
        ... )
    """
    if min_periods is None:
        min_periods = window

    # 로그 스케일링: ln(volume + 1)로 이상치 영향력 압축
    # +1은 volume=0일 때 ln(0) = -inf 방지
    log_volume = np.log1p(volume)  # log1p(x) = ln(1 + x), 수치 안정성 우수

    # 가중 수익률: sum(return * ln_volume) / sum(ln_volume)
    weighted_returns: pd.Series = (  # type: ignore[assignment]
        (returns * log_volume).rolling(window=window, min_periods=min_periods).sum()
    )
    total_log_volume: pd.Series = log_volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()

    # 0으로 나누기 방지
    total_log_volume_safe = total_log_volume.replace(0, np.nan)
    return weighted_returns / total_log_volume_safe


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

    Example:
        >>> momentum = calculate_vw_momentum(
        ...     df["returns"], df["volume"], lookback=30
        ... )
    """
    # 거래량 가중 수익률 계산
    vw_returns: pd.Series = calculate_volume_weighted_returns(
        returns, volume, lookback, min_periods
    )

    # 선택적 스무딩 (EMA)
    if smoothing is not None and smoothing > 1:
        vw_returns = vw_returns.ewm(span=smoothing, adjust=False).mean()  # type: ignore[assignment]

    return vw_returns


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산.

    목표 변동성 대비 실현 변동성의 비율을 계산합니다.
    변동성이 높을 때 포지션을 줄이고, 낮을 때 늘립니다.

    Args:
        realized_vol: 실현 변동성 시리즈
        vol_target: 연간 목표 변동성 (예: 0.40)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)

    Returns:
        변동성 스케일러 시리즈

    Example:
        >>> scalar = calculate_volatility_scalar(vol, vol_target=0.40)
    """
    # 최소 변동성으로 클램프 (0으로 나누기 방지)
    clamped_vol = realized_vol.clip(lower=min_volatility)

    # 스케일러 계산: target / realized
    return vol_target / clamped_vol


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

    Example:
        >>> config = TSMOMConfig(lookback=30, vol_target=0.40)
        >>> processed_df = preprocess(ohlcv_df, config)
        >>> processed_df["vw_momentum"]  # 모멘텀 시리즈
    """
    # 입력 검증
    required_cols = {"close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # 원본 보존 (복사본 생성)
    result = df.copy()

    # OHLCV 컬럼을 float64로 변환 (Decimal 타입 처리)
    # Parquet에서 Decimal로 저장된 경우 np.log() 등이 작동하지 않음
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출 (명시적 Series 타입)
    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    volume_series: pd.Series = result["volume"]  # type: ignore[assignment]

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
        # 방향성 검증: 가격 vs 모멘텀
        price_change = (result["close"].iloc[-1] / result["close"].iloc[0] - 1) * 100
        avg_momentum = valid_data["vw_momentum"].mean()
        aligned = (price_change > 0 and avg_momentum > 0) or (
            price_change < 0 and avg_momentum < 0
        )
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

    라이브 트레이딩에서는 전체 데이터가 아닌 최근 버퍼만 유지하며
    계산합니다. 메모리 효율적이며 실시간 처리에 적합합니다.

    Args:
        buffer: 최근 캔들 버퍼 (최신이 마지막)
        config: TSMOM 설정
        max_rows: 최대 버퍼 크기

    Returns:
        전처리된 버퍼 (마지막 행이 최신 시그널)

    Example:
        >>> # 라이브 트레이딩 루프에서
        >>> buffer = buffer.append(new_candle).tail(200)
        >>> processed = preprocess_live(buffer, config)
        >>> latest_signal = processed["raw_signal"].iloc[-1]
    """
    # 버퍼 크기 제한
    if len(buffer) > max_rows:
        buffer = buffer.tail(max_rows)

    # 일반 전처리 수행
    return preprocess(buffer, config)
