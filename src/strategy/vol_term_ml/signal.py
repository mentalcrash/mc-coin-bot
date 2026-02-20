"""Vol-Term ML 시그널 생성 (Rolling Ridge Regression).

Vol term structure features를 Ridge로 결합하여 방향 예측.
Shift(1) Rule 적용.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.strategy.types import StrategySignals
from src.strategy.vol_term_ml.config import ShortMode, VolTermMLConfig

logger = logging.getLogger(__name__)

_MIN_TRAIN_SAMPLES = 30


def generate_signals(
    df: pd.DataFrame,
    config: VolTermMLConfig | None = None,
    *,
    predict_last_only: bool = False,
) -> StrategySignals:
    """Vol-Term ML 시그널 생성.

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
        config = VolTermMLConfig()

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

    loop_start = max(training_window, n - 2) if predict_last_only else training_window
    for t in range(loop_start, n):
        start_idx = t - training_window
        x_train = feature_matrix[start_idx:t]
        y_train = forward_returns[start_idx:t]

        valid_mask = ~(np.isnan(y_train) | np.any(np.isnan(x_train), axis=1))
        if valid_mask.sum() < _MIN_TRAIN_SAMPLES:
            continue

        x_valid = x_train[valid_mask]
        y_valid = y_train[valid_mask]

        model = Ridge(alpha=config.ridge_alpha)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(x_valid, y_valid)
        except Exception:
            logger.debug("Ridge fit failed at index %d", t)
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
    config: VolTermMLConfig,
) -> tuple[pd.Series, pd.Series]:
    """ShortMode에 따라 direction/strength 필터링."""
    if config.short_mode in {ShortMode.DISABLED, ShortMode.HEDGE_ONLY}:
        short_mask = direction == -1
        direction = direction.where(~short_mask, 0)
        strength = strength.where(~short_mask, 0.0)
    return direction, strength
