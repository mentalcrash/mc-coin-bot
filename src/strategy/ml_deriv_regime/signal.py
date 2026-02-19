"""ML Derivatives Regime 시그널 생성.

Rolling Elastic Net으로 derivatives features에서 forward return 예측.
Shift(1) Rule: 모든 feature는 shift(1) 적용 후 시그널 계산.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from src.strategy.types import StrategySignals

if TYPE_CHECKING:
    from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig

logger = logging.getLogger(__name__)

# Elastic Net 학습에 필요한 최소 유효 샘플 수
_MIN_TRAIN_SAMPLES = 30


def generate_signals(
    df: pd.DataFrame,
    config: MlDerivRegimeConfig,
    *,
    predict_last_only: bool = False,
) -> StrategySignals:
    """ML Derivatives Regime 시그널 생성 (Rolling Elastic Net).

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정
        predict_last_only: True이면 마지막 시그널만 계산 (incremental 모드)

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    from src.strategy.ml_deriv_regime.config import ShortMode

    # Feature 컬럼 추출
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feat_cols:
        msg = "No feature columns (feat_*) found. Run preprocess() first."
        raise ValueError(msg)

    required_cols = {"forward_return", "vol_scalar"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}. Run preprocess() first."
        raise ValueError(msg)

    n = len(df)
    training_window = config.training_window
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    forward_return_series: pd.Series = df["forward_return"]  # type: ignore[assignment]

    # Feature matrix
    feature_matrix = df[feat_cols].to_numpy()
    forward_returns = forward_return_series.to_numpy()

    # Prediction 결과 배열
    predictions = np.full(n, np.nan)

    # Rolling ML training (loop은 ML 예외)
    # prediction_horizon 만큼의 forward return이 확정된 데이터만 학습에 사용
    horizon = config.prediction_horizon
    loop_start = max(training_window, n - 2) if predict_last_only else training_window
    for t in range(loop_start, n):
        start_idx = t - training_window
        # forward_return[i]는 close[i+horizon] 필요 → i <= t-horizon 까지만 확정
        train_end = t - horizon
        if train_end <= start_idx:
            continue
        x_train = feature_matrix[start_idx:train_end]
        y_train = forward_returns[start_idx:train_end]

        valid_mask = ~(np.isnan(y_train) | np.any(np.isnan(x_train), axis=1))

        if valid_mask.sum() < _MIN_TRAIN_SAMPLES:
            continue

        x_valid = x_train[valid_mask]
        y_valid = y_train[valid_mask]

        model = ElasticNet(
            l1_ratio=config.alpha,
            alpha=0.01,
            max_iter=1000,
            random_state=42,
        )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(x_valid, y_valid)
        except Exception:
            logger.debug("ElasticNet fit failed at index %d", t)
            continue

        x_current = feature_matrix[t : t + 1]
        if np.any(np.isnan(x_current)):
            continue

        predictions[t] = model.predict(x_current)[0]

    # Prediction -> Signal
    prediction_series = pd.Series(predictions, index=df.index)
    pred_direction = np.sign(prediction_series)
    scaled_signal: pd.Series = pred_direction * vol_scalar_series  # type: ignore[assignment]

    # Shift(1) 적용
    signal_shifted: pd.Series = scaled_signal.shift(1)  # type: ignore[assignment]

    # Direction
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # Strength
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # Short mode 처리
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == -1
        direction = direction.where(~short_mask, 0)
        strength = strength.where(~short_mask, 0.0)
    elif config.short_mode == ShortMode.HEDGE_ONLY:
        if "drawdown" in df.columns:
            dd = df["drawdown"].shift(1)
            hedge_active = dd < config.hedge_threshold
            short_mask = direction == -1
            suppress_short = short_mask & ~hedge_active
            direction = direction.where(~suppress_short, 0)
            strength = strength.where(~suppress_short, 0.0)
            active_short = short_mask & hedge_active
            strength = strength.where(~active_short, strength * config.hedge_strength_ratio)
        else:
            short_mask = direction == -1
            direction = direction.where(~short_mask, 0)
            strength = strength.where(~short_mask, 0.0)

    # Entry/Exit
    prev_direction = direction.shift(1).fillna(0).astype(int)
    entries = (direction != 0) & (direction != prev_direction)
    exits = (direction == 0) & (prev_direction != 0)

    return StrategySignals(
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        direction=direction,
        strength=strength,
    )
