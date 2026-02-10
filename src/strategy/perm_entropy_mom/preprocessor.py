"""Permutation Entropy Momentum Preprocessor (Indicator Calculation).

Computes Permutation Entropy (PE) and momentum indicators.
PE measures the complexity/orderliness of price patterns using ordinal rankings.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #12 Data Engineering: Log returns for internal calculation
"""

import logging
from math import factorial, log
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_atr,
    calculate_drawdown,
    calculate_realized_volatility,
    calculate_returns,
    calculate_volatility_scalar,
)

logger = logging.getLogger(__name__)


def _compute_pe_for_window(
    window_data: np.ndarray[Any, np.dtype[np.floating[Any]]],
    m: int,
    max_entropy: float,
) -> float:
    """Compute Permutation Entropy for a single rolling window.

    Args:
        window_data: 1D array of returns within the window.
        m: Permutation order (embedding dimension).
        max_entropy: log(m!) for normalization.

    Returns:
        Normalized PE in [0, 1]. NaN if window is too short.
    """
    n = len(window_data)
    if n < m:
        return np.nan

    # Create sliding sub-windows of size m using stride_tricks
    patterns = np.lib.stride_tricks.sliding_window_view(window_data, m)

    # Convert to ordinal patterns (rankings)
    rankings = np.argsort(np.argsort(patterns, axis=1), axis=1)

    # Encode rankings as unique pattern IDs using base-m encoding
    multipliers = m ** np.arange(m - 1, -1, -1)
    pattern_ids = (rankings * multipliers).sum(axis=1)

    # Count frequencies and compute Shannon entropy
    _, counts = np.unique(pattern_ids, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))

    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def _rolling_permutation_entropy(
    returns: pd.Series,
    window: int,
    m: int,
) -> pd.Series:
    """Compute rolling Permutation Entropy.

    Uses pandas rolling.apply with the PE computation function.
    The PE is normalized to [0, 1] by dividing by log(m!).

    Note:
        rolling.apply() is used because PE computation requires
        ordinal pattern extraction which cannot be fully vectorized
        across rolling windows. This is the standard exception pattern
        used in this codebase (see entropy_switch).

    Args:
        returns: Return series.
        window: Rolling window size.
        m: Permutation order.

    Returns:
        Rolling PE series normalized to [0, 1].
    """
    max_entropy = log(factorial(m))

    def _pe_apply(data: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
        return _compute_pe_for_window(data, m, max_entropy)

    pe_series: pd.Series = returns.rolling(
        window=window,
        min_periods=window,
    ).apply(_pe_apply, raw=True)  # type: ignore[assignment]

    return pd.Series(pe_series, index=returns.index)


def preprocess(
    df: pd.DataFrame,
    config: PermEntropyMomConfig,
) -> pd.DataFrame:
    """Permutation Entropy Momentum preprocessing (indicator calculation).

    Calculated Columns:
        - returns: Returns (log or simple)
        - pe_short: Short-term Permutation Entropy (pe_short_window)
        - pe_long: Long-term Permutation Entropy (pe_long_window)
        - mom_direction: Momentum direction (sign of rolling sum)
        - realized_vol: Realized volatility (annualized)
        - vol_scalar: Volatility scalar
        - conviction: Conviction scaler (1 - PE_normalized)
        - atr: Average True Range
        - drawdown: Rolling peak drawdown

    Args:
        df: OHLCV DataFrame (DatetimeIndex required)
        config: PermEntropyMom configuration

    Returns:
        New DataFrame with indicators added

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

    # 1. Returns calculation
    result["returns"] = calculate_returns(
        close_series,
        use_log=config.use_log_returns,
    )
    returns_series: pd.Series = result["returns"]  # type: ignore[assignment]

    # 2. Short-term Permutation Entropy (PE_30bar ~ 5d at 4H)
    result["pe_short"] = _rolling_permutation_entropy(
        returns_series,
        window=config.pe_short_window,
        m=config.pe_order,
    )

    # 3. Long-term Permutation Entropy (PE_60bar ~ 10d at 4H)
    result["pe_long"] = _rolling_permutation_entropy(
        returns_series,
        window=config.pe_long_window,
        m=config.pe_order,
    )

    # 4. Momentum direction (sign of rolling return sum)
    mom_sum = returns_series.rolling(
        window=config.mom_lookback,
        min_periods=config.mom_lookback,
    ).sum()
    result["mom_direction"] = np.sign(mom_sum)

    # 5. Realized volatility (annualized)
    realized_vol = calculate_realized_volatility(
        returns_series,
        window=config.mom_lookback,
        annualization_factor=config.annualization_factor,
    )
    result["realized_vol"] = realized_vol

    # 6. Volatility scalar (vol_target / realized_vol)
    result["vol_scalar"] = calculate_volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # 7. Conviction scaler = 1 - PE (using short-term PE as primary)
    # Low PE = high conviction (orderly), High PE = low conviction (noisy)
    pe_short_series: pd.Series = result["pe_short"]  # type: ignore[assignment]
    result["conviction"] = 1.0 - pe_short_series

    # 8. ATR calculation (for trailing stop)
    result["atr"] = calculate_atr(
        high_series,
        low_series,
        close_series,
        period=config.atr_period,
    )

    # 9. Drawdown calculation (for hedge short mode)
    result["drawdown"] = calculate_drawdown(close_series)

    # Debug: indicator statistics
    valid_data = result.dropna()
    if len(valid_data) > 0:
        pe_short_mean = valid_data["pe_short"].mean()
        pe_long_mean = valid_data["pe_long"].mean()
        conv_mean = valid_data["conviction"].mean()
        vs_mean = valid_data["vol_scalar"].mean()
        msg = (
            "PE-Mom Indicators | Avg PE_short: %.4f, PE_long: %.4f, "
            "Conviction: %.4f, Vol Scalar: %.4f"
        )
        logger.info(
            msg,
            pe_short_mean,
            pe_long_mean,
            conv_mean,
            vs_mean,
        )

    return result
