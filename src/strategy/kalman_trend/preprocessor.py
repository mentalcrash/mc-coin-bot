"""Adaptive Kalman Trend Preprocessor (Indicator Calculation).

칼만 필터로 가격의 state(smoothed price)와 velocity를 추정합니다.
Adaptive Q parameter로 변동성 레짐에 자동 적응합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation

Note:
    Kalman filter는 본질적으로 순차적(sequential) 알고리즘입니다.
    각 스텝의 state와 covariance가 이전 스텝에 의존하므로,
    벡터화가 불가능하여 for 루프를 사용합니다.
    이것은 Zero Loop Policy의 유일한 예외입니다.
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.strategy.kalman_trend.config import KalmanTrendConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def _run_kalman_filter(
    prices: npt.NDArray[np.float64],
    q_adaptive: npt.NDArray[np.float64],
    r: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run Kalman filter over price series.

    칼만 필터는 본질적으로 순차적(inherently sequential) 알고리즘입니다.
    각 스텝의 state estimate와 error covariance가 이전 스텝에 의존하므로,
    벡터화(vectorization)가 불가능합니다. 따라서 for 루프를 사용합니다.
    이것은 Zero Loop Policy의 유일한 예외입니다.

    State Model:
        x = [price, velocity]
        F = [[1, 1], [0, 1]]  (dt=1 for bar-by-bar)
        H = [[1, 0]]

    Args:
        prices: 종가 시리즈 (numpy array)
        q_adaptive: 각 bar의 adaptive 프로세스 노이즈 (numpy array)
        r: 관측 노이즈 스칼라

    Returns:
        (smoothed_prices, velocities) 튜플
    """
    n = len(prices)
    smoothed = np.zeros(n)
    velocities = np.zeros(n)

    # Initialize state
    smoothed[0] = prices[0]
    velocities[0] = 0.0

    # Initial covariance (large uncertainty)
    p = np.eye(2) * 100.0

    # System matrices
    f = np.array([[1.0, 1.0], [0.0, 1.0]])  # Transition
    h = np.array([[1.0, 0.0]])  # Observation
    r_mat = np.array([[r]])  # Observation noise

    state = np.array([prices[0], 0.0])

    # Kalman filter loop (inherently sequential — each step depends on
    # the previous step's state and covariance, making vectorization impossible)
    for i in range(1, n):
        # Adaptive process noise covariance
        q = q_adaptive[i]
        q_mat = q * np.array([[1.0 / 3.0, 1.0 / 2.0], [1.0 / 2.0, 1.0]])

        # Predict
        x_pred = f @ state
        p_pred = f @ p @ f.T + q_mat

        # Innovation
        y = prices[i] - h @ x_pred
        s = h @ p_pred @ h.T + r_mat

        # Kalman gain
        k = p_pred @ h.T @ np.linalg.inv(s)

        # Update
        state = x_pred + (k @ y).flatten()
        p = (np.eye(2) - k @ h) @ p_pred

        smoothed[i] = state[0]
        velocities[i] = state[1]

    return smoothed, velocities


def preprocess(
    df: pd.DataFrame,
    config: KalmanTrendConfig,
) -> pd.DataFrame:
    """Adaptive Kalman Trend 전처리 (지표 계산).

    Calculated Columns:
        - returns: 수익률 (로그 또는 단순)
        - kalman_state: 칼만 필터 smoothed price
        - kalman_velocity: 칼만 필터 velocity (trend speed)
        - q_adaptive: 적응형 프로세스 노이즈
        - realized_vol: 실현 변동성 (단기, vol_lookback)
        - vol_scalar: 변동성 스케일러
        - atr: Average True Range
        - drawdown: 롤링 최고점 대비 드로다운

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: Kalman Trend 설정

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

    # 2. 단기/장기 실현 변동성 (Adaptive Q 계산용)
    short_term_vol = calculate_realized_volatility(
        returns_series,
        window=config.vol_lookback,
        annualization_factor=config.annualization_factor,
    )
    long_term_vol = calculate_realized_volatility(
        returns_series,
        window=config.long_term_vol_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = short_term_vol

    # 3. Adaptive Q 계산: base_q * (short_term_vol / long_term_vol)
    # long_term_vol이 0이면 NaN → clip으로 안전하게 처리
    long_term_vol_safe = long_term_vol.clip(lower=1e-10)
    q_ratio = short_term_vol / long_term_vol_safe
    q_adaptive_series = config.base_q * q_ratio.clip(lower=0.001, upper=10.0)

    # NaN을 base_q로 채움 (warmup 기간)
    q_adaptive_filled: pd.Series = q_adaptive_series.fillna(config.base_q)  # type: ignore[assignment]
    result["q_adaptive"] = q_adaptive_filled

    # 4. Kalman filter 실행
    prices = close_series.to_numpy(dtype=np.float64)
    q_adaptive_arr = q_adaptive_filled.to_numpy(dtype=np.float64)

    smoothed, velocities = _run_kalman_filter(
        prices,
        q_adaptive_arr,
        config.observation_noise,
    )
    result["kalman_state"] = smoothed
    result["kalman_velocity"] = velocities

    # 5. 변동성 스케일러 (포지션 사이징용)
    result["vol_scalar"] = calculate_volatility_scalar(
        short_term_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 6. ATR 계산
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 7. 드로다운 계산
    result["drawdown"] = calculate_drawdown(close_series)

    # 디버그: 지표 통계
    valid_data = result.dropna()
    if len(valid_data) > 0:
        vel_mean = valid_data["kalman_velocity"].mean()
        vel_std = valid_data["kalman_velocity"].std()
        q_mean = valid_data["q_adaptive"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        logger.info(
            "Kalman-Trend Indicators | Avg Velocity: %.4f, Std Velocity: %.4f, Avg Q: %.6f, Avg Vol Scalar: %.4f",
            vel_mean,
            vel_std,
            q_mean,
            vs_mean,
        )

    return result
