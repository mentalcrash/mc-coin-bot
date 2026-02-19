"""Volatility Structure ML 시그널 생성.

Rolling Elastic Net으로 13종 vol features → forward return 예측.
Shift(1) Rule: ML 예측 + vol_scalar에 shift(1) 적용.
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
from src.strategy.vol_struct_ml.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.vol_struct_ml.config import VolStructMLConfig

logger = logging.getLogger(__name__)

_MIN_TRAIN_SAMPLES = 30


def generate_signals(
    df: pd.DataFrame,
    config: VolStructMLConfig,
    *,
    predict_last_only: bool = False,
) -> StrategySignals:
    """Volatility Structure ML 시그널 생성.

    Args:
        df: preprocess() 출력 DataFrame
        config: 전략 설정
        predict_last_only: True면 마지막 시그널만 계산 (EDA incremental)

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    # Feature columns
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feat_cols:
        msg = "No feature columns (feat_*) found. Run preprocess() first."
        raise ValueError(msg)

    n = len(df)
    training_window = config.training_window
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]
    forward_return_series: pd.Series = df["forward_return"]  # type: ignore[assignment]

    feature_matrix = df[feat_cols].to_numpy()
    forward_returns = forward_return_series.to_numpy()

    predictions = np.full(n, np.nan)

    h = config.prediction_horizon
    loop_start = max(training_window, n - 2) if predict_last_only else training_window
    for t in range(loop_start, n):
        start_idx = t - training_window
        # forward_return[i] = (close[i+h] - close[i]) / close[i]
        # At time t, close[t] is the latest known price (current bar close).
        # y_train must exclude rows where forward_return references close > t.
        # Row (t-h) forward_return uses close[t], which is marginally acceptable
        # (current bar). Row (t-h+1) uses close[t+1] = future → must exclude.
        # Safe cutoff: train on [start_idx, t-h) only.
        train_end = max(start_idx, t - h)
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

    # --- Shift(1): prediction + vol_scalar ---
    prediction_series = pd.Series(predictions, index=df.index)
    pred_direction = np.sign(prediction_series)
    scaled_signal: pd.Series = (pred_direction * vol_scalar_series).shift(1)  # type: ignore[assignment]

    # --- Direction ---
    direction = _apply_short_mode(scaled_signal, df, config)

    # --- Strength ---
    strength = direction.astype(float) * scaled_signal.abs().fillna(0)

    if config.short_mode == ShortMode.HEDGE_ONLY:
        strength = pd.Series(
            np.where(direction == -1, strength * config.hedge_strength_ratio, strength),
            index=df.index,
        )

    strength = strength.fillna(0.0)

    # --- Entries / Exits ---
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
    scaled_signal: pd.Series,
    df: pd.DataFrame,
    config: VolStructMLConfig,
) -> pd.Series:
    """ShortMode에 따른 direction 계산."""
    raw_dir = pd.Series(np.sign(scaled_signal), index=df.index).fillna(0).astype(int)

    if config.short_mode == ShortMode.DISABLED:
        return pd.Series(
            np.where(raw_dir > 0, 1, 0),
            index=df.index,
            dtype=int,
        )

    if config.short_mode == ShortMode.HEDGE_ONLY:
        dd = df["drawdown"].shift(1)
        hedge_active = dd < config.hedge_threshold
        return pd.Series(
            np.where(
                raw_dir > 0,
                1,
                np.where((raw_dir < 0) & hedge_active, -1, 0),
            ),
            index=df.index,
            dtype=int,
        )

    # FULL
    return pd.Series(raw_dir, index=df.index, dtype=int)
