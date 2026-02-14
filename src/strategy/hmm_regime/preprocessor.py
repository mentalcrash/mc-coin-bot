"""HMM Regime Preprocessor.

GaussianHMM expanding window training for look-ahead bias prevention.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.market.indicators import (
    atr,
    drawdown,
    log_returns,
    realized_volatility,
    simple_returns,
    volatility_scalar,
)

if TYPE_CHECKING:
    from src.strategy.hmm_regime.config import HMMRegimeConfig

logger = logging.getLogger(__name__)

# Constant for 2-state HMM mapping
_TWO_STATES = 2


def _map_state_to_regime(
    means: np.ndarray,  # type: ignore[type-arg]
    state: int,
    n_states: int,
) -> int:
    """Map HMM state to regime label based on mean return ordering.

    Sorts states by mean return:
    - Highest mean -> Bull (1)
    - Lowest mean -> Bear (-1)
    - Middle -> Sideways (0)

    For 2 states: highest=Bull, lowest=Bear.

    Args:
        means: Array of mean returns per state (shape: n_states,)
        state: Current HMM predicted state index
        n_states: Total number of HMM states

    Returns:
        Regime label: 1 (Bull), -1 (Bear), 0 (Sideways)
    """
    sorted_states = np.argsort(means)

    if n_states == _TWO_STATES:
        mapping = {int(sorted_states[0]): -1, int(sorted_states[1]): 1}
    else:
        mapping = {}
        mapping[int(sorted_states[0])] = -1  # Bear (lowest mean)
        mapping[int(sorted_states[-1])] = 1  # Bull (highest mean)
        for idx in range(1, n_states - 1):
            mapping[int(sorted_states[idx])] = 0  # Sideways (middle)

    return mapping.get(state, 0)


def _compute_hmm_regimes(  # pyright: ignore[reportUnknownParameterType]
    returns: pd.Series,
    rolling_vol: pd.Series,
    config: HMMRegimeConfig,
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Expanding window HMM training -- only uses past data.

    NOTE: This function uses a for loop for expanding window HMM training.
    This is an acceptable exception to the Zero Loop Policy as ML model
    training requires iterative computation.

    Args:
        returns: Return series
        rolling_vol: Rolling volatility series
        config: HMM Regime configuration

    Returns:
        Tuple of (regimes array, regime probabilities array)
    """
    # Lazy import to handle optional dependency
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        msg = "hmmlearn is required for HMM Regime strategy. Install with: uv add hmmlearn"
        raise ImportError(msg)  # noqa: B904

    n = len(returns)
    regimes = np.full(n, -1)  # -1 = unknown
    regime_probs = np.full(n, 0.0)

    returns_arr = returns.to_numpy().astype(np.float64)
    vol_arr = rolling_vol.to_numpy().astype(np.float64)

    last_model = None

    for i in range(config.min_train_window, n):
        # Retrain at specified intervals
        if (i - config.min_train_window) % config.retrain_interval == 0:
            # Build feature matrix from 0..i-1 (past data only)
            features = np.column_stack([returns_arr[:i], vol_arr[:i]])
            valid_mask = ~np.isnan(features).any(axis=1)
            features_clean = features[valid_mask]

            if len(features_clean) < config.n_states * 10:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = GaussianHMM(
                        n_components=config.n_states,
                        n_iter=config.n_iter,
                        covariance_type="full",
                        random_state=42,
                    )
                    model.fit(features_clean)
                    last_model = model
                except Exception:
                    logger.debug("HMM training failed at bar %d", i)
                    continue

        if last_model is not None:
            feat_i = np.array([[returns_arr[i], vol_arr[i]]])
            if not np.isnan(feat_i).any():
                try:
                    state = int(last_model.predict(feat_i)[0])
                    posteriors = last_model.predict_proba(feat_i)
                    prob = float(posteriors[0, state])
                    means = last_model.means_[:, 0]
                    regimes[i] = _map_state_to_regime(means, state, config.n_states)
                    regime_probs[i] = prob
                except Exception:
                    logger.debug("HMM prediction failed at bar %d", i)

    return regimes, regime_probs


def preprocess(
    df: pd.DataFrame,
    config: HMMRegimeConfig,
) -> pd.DataFrame:
    """HMM Regime preprocessor.

    OHLCV DataFrame에 HMM regime 기반 지표를 계산하여 추가합니다.

    Calculated Columns:
        - returns: Log/simple returns
        - rolling_vol: Rolling volatility
        - regime: HMM regime label (-1=Bear/Unknown, 0=Sideways, 1=Bull)
        - regime_prob: Regime posterior probability
        - realized_vol: Annualized realized volatility
        - vol_scalar: Volatility scalar (vol_target / realized_vol)
        - atr: Average True Range
        - drawdown: Rolling max drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex required)
        config: HMM Regime configuration

    Returns:
        DataFrame with calculated indicators

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    result = df.copy()

    # Convert OHLCV columns to float64
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    close_series: pd.Series = result["close"]  # type: ignore[assignment]
    high_series: pd.Series = result["high"]  # type: ignore[assignment]
    low_series: pd.Series = result["low"]  # type: ignore[assignment]

    # 1. Returns
    result["returns"] = (
        log_returns(close_series) if config.use_log_returns else simple_returns(close_series)
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Rolling volatility
    result["rolling_vol"] = returns_series.rolling(
        config.vol_window, min_periods=config.vol_window
    ).std()
    rolling_vol_series: pd.Series = result["rolling_vol"]  # type: ignore[assignment]

    # 3. HMM regime classification (expanding window)
    regimes, regime_probs = _compute_hmm_regimes(returns_series, rolling_vol_series, config)
    result["regime"] = regimes
    result["regime_prob"] = regime_probs

    # 4. Realized vol + vol_scalar
    result["realized_vol"] = realized_volatility(
        returns_series,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    realized_vol_series: pd.Series = result["realized_vol"]  # type: ignore[assignment]
    result["vol_scalar"] = volatility_scalar(
        realized_vol_series,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 5. ATR + Drawdown
    result["atr"] = atr(high_series, low_series, close_series, config.atr_period)
    result["drawdown"] = drawdown(close_series)

    # Stats logging
    valid_regimes = regimes[regimes >= 0]
    if len(valid_regimes) > 0:
        bull_pct = (valid_regimes == 1).sum() / len(valid_regimes) * 100
        bear_pct = (valid_regimes == -1).sum() / len(valid_regimes) * 100
        sideways_pct = (valid_regimes == 0).sum() / len(valid_regimes) * 100
        logger.info(
            "HMM Regimes | Bull: %.1f%%, Bear: %.1f%%, Sideways: %.1f%%",
            bull_pct,
            bear_pct,
            sideways_pct,
        )

    return result
