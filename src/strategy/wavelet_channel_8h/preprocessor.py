"""Wavelet-Channel 8H 전처리 모듈.

DWT denoised close에 3종 채널(Donchian/Keltner/BB) x 3스케일(22/66/132) feature 계산.
모든 연산은 벡터화 (for 루프 금지).

IMPORTANT: denoised close는 채널 계산에만 사용.
returns/vol/drawdown은 raw close 기준.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.market.indicators import (
    bollinger_bands,
    donchian_channel,
    drawdown,
    keltner_channels,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.wavelet_channel_8h.config import WaveletChannel8hConfig

logger = logging.getLogger(__name__)

# Lazy import: pywt (PyWavelets) is optional
_pywt_available = False
_pywt_mod: Any = None
try:
    import pywt as _pywt_import  # pyright: ignore[reportMissingImports]

    _pywt_mod = _pywt_import
    _pywt_available = True
except ImportError:
    pass

PYWT_AVAILABLE: bool = _pywt_available

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _wavelet_denoise(close: pd.Series, family: str, level: int) -> pd.Series:
    """DWT denoising: approximation coefficients만 유지하여 노이즈 제거.

    pywt가 없으면 EMA 기반 lowpass filter로 fallback.

    Args:
        close: raw close 가격 Series
        family: 웨이블릿 종류 (e.g. 'db4', 'haar')
        level: DWT 분해 레벨

    Returns:
        denoised close Series (동일 인덱스)
    """
    import pandas as pd

    values = close.dropna().to_numpy().astype(np.float64)

    if len(values) < 2 ** (level + 1):
        logger.warning(
            "Insufficient data for wavelet denoising (need %d, got %d). Using raw close.",
            2 ** (level + 1),
            len(values),
        )
        return close.copy()

    if _pywt_available and _pywt_mod is not None:
        # DWT decomposition → zero out detail coefficients → reconstruct
        coeffs = _pywt_mod.wavedec(values, family, level=level)
        # Keep only approximation (coeffs[0]), zero out details
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
        reconstructed = _pywt_mod.waverec(coeffs, family)

        # waverec may produce array 1 element longer due to padding
        if len(reconstructed) > len(values):
            reconstructed = reconstructed[: len(values)]
        elif len(reconstructed) < len(values):
            # Pad with last value (shouldn't happen in practice)
            reconstructed = np.pad(
                reconstructed,
                (0, len(values) - len(reconstructed)),
                mode="edge",
            )

        denoised = pd.Series(reconstructed, index=close.dropna().index)
    else:
        # Fallback: EMA lowpass filter (span ≈ 2^level)
        logger.info(
            "pywt not available. Using EMA(span=%d) lowpass filter as fallback.",
            2**level,
        )
        span = 2**level
        denoised = close.dropna().ewm(span=span, adjust=False).mean()

    # Reindex to match original (NaN positions preserved)
    result: pd.Series = denoised.reindex(close.index)  # type: ignore[assignment]
    return result


def preprocess(df: pd.DataFrame, config: WaveletChannel8hConfig) -> pd.DataFrame:
    """Wavelet-Channel 8H feature 계산.

    Calculated Columns:
        - dc_upper_{s}, dc_lower_{s}: 3-scale Donchian Channel (denoised close 기반)
        - kc_upper_{s}, kc_lower_{s}: 3-scale Keltner Channels (denoised close 기반)
        - bb_upper_{s}, bb_lower_{s}: 3-scale Bollinger Bands (denoised close 기반)
        - returns: log return (raw close)
        - realized_vol: 연환산 실현 변동성 (raw close)
        - vol_scalar: 변동성 스케일러 (raw close)
        - drawdown: HEDGE_ONLY용 drawdown (raw close)

    Args:
        df: OHLCV DataFrame (DatetimeIndex 필수)
        config: 전략 설정

    Returns:
        feature가 추가된 새 DataFrame

    Raises:
        ValueError: 필수 컬럼 누락 시
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    df = df.copy()

    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]

    # --- Wavelet Denoising (채널 계산 전용) ---
    denoised_close = _wavelet_denoise(close, config.wavelet_family, config.wavelet_level)

    scales = (config.scale_short, config.scale_mid, config.scale_long)

    # --- 3-Scale Donchian Channels (denoised high/low 대신 raw high/low 사용,
    #     but close denoised → Donchian은 high/low 기반이므로 raw 유지) ---
    for s in scales:
        upper, _mid, lower = donchian_channel(high, low, s)
        df[f"dc_upper_{s}"] = upper
        df[f"dc_lower_{s}"] = lower

    # --- 3-Scale Keltner Channels (denoised close로 EMA/ATR 계산) ---
    for s in scales:
        kc_upper, _kc_mid, kc_lower = keltner_channels(
            high,
            low,
            denoised_close,
            ema_period=s,
            atr_period=s,
            multiplier=config.kc_mult,
        )
        df[f"kc_upper_{s}"] = kc_upper
        df[f"kc_lower_{s}"] = kc_lower

    # --- 3-Scale Bollinger Bands (denoised close 기반) ---
    for s in scales:
        bb_upper, _bb_mid, bb_lower = bollinger_bands(
            denoised_close,
            period=s,
            std_dev=config.bb_std,
        )
        df[f"bb_upper_{s}"] = bb_upper
        df[f"bb_lower_{s}"] = bb_lower

    # --- Returns (raw close) ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility (raw close) ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar (raw close) ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Drawdown (raw close, HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
