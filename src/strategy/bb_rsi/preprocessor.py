"""BB+RSI Mean Reversion Preprocessor (Indicator Calculation).

볼린저밴드, RSI, ATR, ADX, 변동성 지표를 벡터화 연산으로 계산합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
    - #26 VectorBT Standards: Compatible output format
"""

import logging

import numpy as np
import pandas as pd

from src.strategy.bb_rsi.config import BBRSIConfig

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


def calculate_bollinger_bands(
    close: pd.Series,
    period: int,
    std_dev: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저밴드 계산 (upper, middle, lower).

    Args:
        close: 종가 시리즈
        period: SMA 기간
        std_dev: 표준편차 배수

    Returns:
        (bb_upper, bb_middle, bb_lower) 튜플
    """
    bb_middle = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()

    bb_upper: pd.Series = bb_middle + std_dev * rolling_std  # type: ignore[assignment]
    bb_lower: pd.Series = bb_middle - std_dev * rolling_std  # type: ignore[assignment]

    return (
        pd.Series(bb_upper, index=close.index, name="bb_upper"),
        pd.Series(bb_middle, index=close.index, name="bb_middle"),
        pd.Series(bb_lower, index=close.index, name="bb_lower"),
    )


def calculate_rsi(
    close: pd.Series,
    period: int,
) -> pd.Series:
    """RSI 계산 (Wilder's smoothing).

    Args:
        close: 종가 시리즈
        period: RSI 기간

    Returns:
        RSI 시리즈 (0-100 범위)
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (ewm with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # RS = avg_gain / avg_loss (0 나누기 방지)
    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi: pd.Series = 100 - (100 / (1 + rs))  # type: ignore[assignment]
    return pd.Series(rsi, index=close.index, name="rsi")


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


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX (Average Directional Index) 계산.

    ADX >= 25: 강한 추세 (평균회귀에 불리)
    ADX < 25: 횡보장 (평균회귀에 유리)

    Args:
        high: 고가 시리즈
        low: 저가 시리즈
        close: 종가 시리즈
        period: ADX 기간

    Returns:
        ADX 시리즈 (0-100 범위)
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM, -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index,
    )

    # Wilder's smoothing
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di: pd.Series = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )  # type: ignore[assignment]
    minus_di: pd.Series = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    )  # type: ignore[assignment]

    # DX → ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))

    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.Series(adx, index=close.index, name="adx")


def calculate_bb_position(
    close: pd.Series,
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    bb_middle: pd.Series,
) -> pd.Series:
    """볼린저밴드 내 정규화 위치 계산.

    (close - middle) / (upper - lower) → 대략 -0.5 ~ +0.5 범위

    Args:
        close: 종가 시리즈
        bb_upper: 볼린저밴드 상단
        bb_lower: 볼린저밴드 하단
        bb_middle: 볼린저밴드 중간

    Returns:
        BB 내 정규화 위치 시리즈
    """
    bandwidth = bb_upper - bb_lower
    # 0 나누기 방지
    bandwidth_safe = bandwidth.replace(0, np.nan)
    bb_position: pd.Series = (close - bb_middle) / bandwidth_safe  # type: ignore[assignment]
    return pd.Series(bb_position, index=close.index, name="bb_position")


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
    config: BBRSIConfig,
) -> pd.DataFrame:
    """BB+RSI 전처리 (지표 계산).

    OHLCV DataFrame에 평균회귀 전략에 필요한 기술적 지표를 추가합니다.

    Calculated Columns:
        - bb_upper, bb_middle, bb_lower: 볼린저밴드
        - rsi: RSI (0-100)
        - bb_position: BB 내 정규화 위치
        - atr: Average True Range
        - returns: 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - drawdown: 최고점 대비 하락률
        - adx: ADX (use_adx_filter=True일 때)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: BB+RSI 설정

    Returns:
        지표가 추가된 새로운 DataFrame
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Decimal 타입 → float64 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 컬럼 추출
    close: pd.Series = result["close"]  # type: ignore[assignment]
    high: pd.Series = result["high"]  # type: ignore[assignment]
    low: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. 볼린저밴드
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        close, config.bb_period, config.bb_std
    )
    result["bb_upper"] = bb_upper
    result["bb_middle"] = bb_middle
    result["bb_lower"] = bb_lower

    # 2. RSI
    result["rsi"] = calculate_rsi(close, config.rsi_period)

    # 3. BB 내 정규화 위치
    result["bb_position"] = calculate_bb_position(close, bb_upper, bb_lower, bb_middle)

    # 4. ATR
    result["atr"] = calculate_atr(high, low, close, config.atr_period)

    # 5. 수익률
    result["returns"] = calculate_returns(close, use_log=config.use_log_returns)

    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 6. 실현 변동성
    result["realized_vol"] = calculate_realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )

    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]

    # 7. 변동성 스케일러
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 8. 드로다운 (HEDGE_ONLY 모드용)
    result["drawdown"] = calculate_drawdown(close)

    # 9. ADX (레짐 필터)
    if config.use_adx_filter:
        result["adx"] = calculate_adx(high, low, close, period=config.adx_period)

    # 지표 통계 로깅
    valid_data = result.dropna()
    if len(valid_data) > 0:
        rsi_mean = valid_data["rsi"].mean()
        bb_pos_mean = valid_data["bb_position"].mean()
        vs_min = valid_data["vol_scalar"].min()
        vs_max = valid_data["vol_scalar"].max()
        logger.info(
            "📊 BB-RSI Indicators | RSI Mean: %.1f, BB Pos Mean: %.3f, Vol Scalar: [%.2f, %.2f]",
            rsi_mean,
            bb_pos_mean,
            vs_min,
            vs_max,
        )

    return result
