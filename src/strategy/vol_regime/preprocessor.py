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

from src.strategy.vol_regime.config import VolRegimeConfig

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
    min_periods: int | None = None,
) -> pd.Series:
    """실현 변동성 계산 (연환산).

    Args:
        returns: 수익률 시리즈
        window: Rolling 윈도우 크기
        annualization_factor: 연환산 계수 (일봉: 365)
        min_periods: 최소 관측치 수 (None이면 window 사용)

    Returns:
        연환산 변동성 시리즈
    """
    if min_periods is None:
        min_periods = window

    rolling_std = returns.rolling(window=window, min_periods=min_periods).std()
    return rolling_std * np.sqrt(annualization_factor)


def calculate_volatility_scalar(
    realized_vol: pd.Series,
    vol_target: float,
    min_volatility: float = 0.05,
) -> pd.Series:
    """변동성 스케일러 계산.

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
    vol_lookback: int,
    vol_rank_lookback: int,
    annualization_factor: float,
) -> pd.Series:
    """변동성 regime 판별 (percentile rank).

    Rolling 변동성의 percentile rank를 계산하여 현재 변동성이
    과거 대비 어느 수준인지 0~1 범위로 반환합니다.

    Args:
        returns: 수익률 시리즈
        vol_lookback: 변동성 계산 윈도우
        vol_rank_lookback: Percentile rank 계산 윈도우
        annualization_factor: 연환산 계수

    Returns:
        vol_pct 시리즈 (0~1, 1에 가까울수록 고변동성)
    """
    vol = returns.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(
        annualization_factor
    )
    vol_pct = vol.rolling(vol_rank_lookback, min_periods=min(vol_rank_lookback, 60)).rank(pct=True)
    return pd.Series(vol_pct, index=returns.index, name="vol_regime")


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
    log_volume = np.log1p(volume)
    weighted: pd.Series = (  # type: ignore[assignment]
        (returns * log_volume).rolling(lookback, min_periods=lookback).sum()
    )
    total_vol: pd.Series = log_volume.rolling(  # type: ignore[assignment]
        lookback, min_periods=lookback
    ).sum()
    return weighted / total_vol.replace(0, np.nan)


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


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR (Average True Range) 계산.

    Wilder's smoothing(EWM)을 사용한 True Range의 지수이동평균입니다.

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: 계산 기간

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
        드로다운 시리즈 (항상 0 이하, 예: -0.15 = -15%)
    """
    rolling_max = close.expanding().max()
    drawdown: pd.Series = (close - rolling_max) / rolling_max  # type: ignore[assignment]
    return pd.Series(drawdown, index=close.index, name="drawdown")


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
