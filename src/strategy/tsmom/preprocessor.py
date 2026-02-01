"""VW-TSMOM Preprocessor (Indicator Calculation).

이 모듈은 VW-TSMOM 전략에 필요한 모든 지표를 벡터화된 연산으로 계산합니다.
백테스팅과 라이브 트레이딩 모두에서 동일한 코드를 사용합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

from typing import Any

import numpy as np
import pandas as pd

from src.strategy.tsmom.config import TSMOMConfig


def calculate_returns(
    close: pd.Series | Any,
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
    # Series 타입 검증
    if not isinstance(close, pd.Series):
        msg = f"Expected pd.Series, got {type(close)}"
        raise TypeError(msg)

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
    annualization_factor: float = 8760.0,
    min_periods: int | None = None,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Rolling standard deviation을 사용하여 실현 변동성을 계산합니다.
    결과는 연환산되어 반환됩니다.

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수 (시간봉: 8760)
        min_periods: 최소 관측치 수 (None이면 window 사용)

    Returns:
        연환산 변동성 시리즈

    Example:
        >>> vol = calculate_realized_volatility(returns, window=24)
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
    """거래량 가중 수익률 계산.

    각 기간의 수익률에 거래량을 가중하여 평균합니다.
    거래량이 큰 기간의 가격 변화에 더 높은 가중치를 부여합니다.

    Args:
        returns: 수익률 시리즈
        volume: 거래량 시리즈
        window: Rolling 윈도우 크기
        min_periods: 최소 관측치 수

    Returns:
        거래량 가중 수익률 시리즈

    Example:
        >>> vw_returns = calculate_volume_weighted_returns(
        ...     df["returns"], df["volume"], window=24
        ... )
    """
    if min_periods is None:
        min_periods = window

    # 가중 수익률: sum(return * volume) / sum(volume)
    weighted_returns: pd.Series = (  # type: ignore[assignment]
        (returns * volume).rolling(window=window, min_periods=min_periods).sum()
    )
    total_volume: pd.Series = volume.rolling(  # type: ignore[assignment]
        window=window, min_periods=min_periods
    ).sum()

    # 0으로 나누기 방지
    total_volume_safe = total_volume.replace(0, np.nan)
    return weighted_returns / total_volume_safe


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
        ...     df["returns"], df["volume"], lookback=24
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
        vol_target: 연간 목표 변동성 (예: 0.15)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)

    Returns:
        변동성 스케일러 시리즈

    Example:
        >>> scalar = calculate_volatility_scalar(vol, vol_target=0.15)
    """
    # 최소 변동성으로 클램프 (0으로 나누기 방지)
    clamped_vol = realized_vol.clip(lower=min_volatility)

    # 스케일러 계산: target / realized
    return vol_target / clamped_vol


def preprocess(
    df: pd.DataFrame,
    config: TSMOMConfig,
) -> pd.DataFrame:
    """VW-TSMOM 전처리 (모든 지표 계산).

    OHLCV DataFrame에 VW-TSMOM 전략에 필요한 모든 지표를 계산하여 추가합니다.
    모든 계산은 벡터화되어 있으며 for 루프를 사용하지 않습니다.

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - realized_vol: 실현 변동성 (연환산)
        - vw_momentum: 거래량 가중 모멘텀
        - vol_scalar: 변동성 스케일러
        - raw_signal: 원시 시그널 (방향 x 스케일러)
        - position_size: 포지션 크기 (레버리지 제한 적용)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
            필수 컬럼: close, volume
        config: TSMOM 설정

    Returns:
        지표가 추가된 새로운 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> config = TSMOMConfig(lookback=24, vol_target=0.15)
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

    # 5. 원시 시그널 계산 (방향 x 스케일러)
    # np.sign()으로 방향 결정, vol_scalar로 크기 조절
    momentum_direction = np.sign(result["vw_momentum"])
    result["raw_signal"] = momentum_direction * result["vol_scalar"]

    # 6. 포지션 크기 (레버리지 제한 적용)
    result["position_size"] = result["raw_signal"].clip(
        lower=-config.max_leverage,
        upper=config.max_leverage,
    )

    # 7. 시그널 임계값 필터 (선택적)
    if config.signal_threshold > 0:
        mask = result["position_size"].abs() < config.signal_threshold
        result.loc[mask, "position_size"] = 0.0

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
        >>> latest_signal = processed["position_size"].iloc[-1]
    """
    # 버퍼 크기 제한
    if len(buffer) > max_rows:
        buffer = buffer.tail(max_rows)

    # 일반 전처리 수행
    return preprocess(buffer, config)
