"""GBTrend 시그널 생성 (Rolling GradientBoosting + 12 momentum features).

Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.strategy.gbtrend.config import GBTrendConfig, ShortMode
from src.strategy.types import StrategySignals

logger = logging.getLogger(__name__)

_MIN_TRAIN_SAMPLES = 30


def generate_signals(
    df: pd.DataFrame,
    config: GBTrendConfig | None = None,
    *,
    predict_last_only: bool = False,
) -> StrategySignals:
    """GBTrend 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame.
        config: 전략 설정.
        predict_last_only: True면 마지막 시그널만 계산.

    Returns:
        StrategySignals.

    Raises:
        ValueError: 필수 컬럼 누락 시.
    """
    if config is None:
        config = GBTrendConfig()

    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feat_cols:
        msg = "No feature columns (feat_*) found. Run preprocess() first."
        raise ValueError(msg)

    required_cols = {"forward_return", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    n = len(df)
    training_window = config.training_window
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    forward_return_series: pd.Series = df["forward_return"]  # type: ignore[assignment]

    feature_matrix = df[feat_cols].to_numpy()
    forward_returns = forward_return_series.to_numpy()

    predictions = np.full(n, np.nan)

    prediction_horizon = config.prediction_horizon
    loop_start = max(training_window, n - 2) if predict_last_only else training_window
    for t in range(loop_start, n):
        start_idx = t - training_window

        # Look-ahead bias 방지: forward_return[i]는 close[i + prediction_horizon]이 필요.
        # 시점 t에서 확정된 인덱스만 사용: i <= t - prediction_horizon.
        resolved_end = t - prediction_horizon + 1
        if resolved_end <= start_idx:
            continue

        x_train = feature_matrix[start_idx:resolved_end]
        y_train = forward_returns[start_idx:resolved_end]

        valid_mask = ~(np.isnan(y_train) | np.any(np.isnan(x_train), axis=1))
        if valid_mask.sum() < _MIN_TRAIN_SAMPLES:
            continue

        x_valid = x_train[valid_mask]
        y_valid = y_train[valid_mask]

        model = GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            random_state=42,
        )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(x_valid, y_valid)
        except Exception:
            logger.debug("GBR fit failed at index %d", t)
            continue

        x_current = feature_matrix[t : t + 1]
        if np.any(np.isnan(x_current)):
            continue

        predictions[t] = model.predict(x_current)[0]

    # Prediction → Signal
    prediction_series = pd.Series(predictions, index=df.index)
    pred_direction = np.sign(prediction_series)
    scaled_signal: pd.Series = pred_direction * vol_scalar_series  # type: ignore[assignment]

    # Shift(1)
    signal_shifted: pd.Series = scaled_signal.shift(1)  # type: ignore[assignment]

    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
    )
    strength = pd.Series(signal_shifted.fillna(0.0), index=df.index)

    # ShortMode 처리
    direction, strength = _apply_short_mode(direction, strength, config)

    # Entry/Exit
    prev_dir = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_dir)
    exits = (direction == 0) & (prev_dir != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )


def _apply_short_mode(
    direction: pd.Series,
    strength: pd.Series,
    config: GBTrendConfig,
) -> tuple[pd.Series, pd.Series]:
    """ShortMode에 따라 direction/strength 필터링."""
    if config.short_mode in {ShortMode.DISABLED, ShortMode.HEDGE_ONLY}:
        short_mask = direction == -1
        direction = direction.where(~short_mask, 0)
        strength = strength.where(~short_mask, 0.0)
    return direction, strength
