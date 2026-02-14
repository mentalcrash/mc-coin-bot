"""Capital Gains Overhang Momentum 전처리 모듈.

OHLCV 데이터에서 turnover 가중 reference price와 CGO feature를 계산한다.
모든 연산은 벡터화 (for 루프 금지).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.cgo_mom.config import CgoMomConfig

from src.market.indicators import (
    drawdown,
    log_returns,
    realized_volatility,
    volatility_scalar,
)

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def preprocess(df: pd.DataFrame, config: CgoMomConfig) -> pd.DataFrame:
    """Capital Gains Overhang Momentum feature 계산.

    Calculated columns:
        - returns: 로그 수익률
        - realized_vol: 실현 변동성 (연환산)
        - vol_scalar: 변동성 스케일러
        - reference_price: turnover 가중 평균 매입단가 추정
        - cgo: Capital Gains Overhang = (price - ref_price) / ref_price
        - cgo_zscore: CGO의 rolling z-score
        - drawdown: rolling drawdown (HEDGE_ONLY용)

    Reference Price 계산:
        volume을 proxy turnover로 사용, EWM(span=turnover_window) 가중.
        ref_price = ewm(volume * close) / ewm(volume)
        Grinblatt & Han (2005) 방법론의 근사.

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
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    # --- Returns ---
    returns = log_returns(close)
    df["returns"] = returns

    # --- Realized Volatility ---
    realized_vol = realized_volatility(
        returns,
        window=config.vol_window,
        annualization_factor=config.annualization_factor,
    )
    df["realized_vol"] = realized_vol

    # --- Vol Scalar ---
    df["vol_scalar"] = volatility_scalar(
        realized_vol,
        vol_target=config.vol_target,
        min_volatility=config.min_volatility,
    )

    # --- Reference Price (Turnover-Weighted Average Cost) ---
    # Volume을 proxy turnover로 사용
    # EWM 가중으로 최근 거래에 더 높은 가중치
    vol_x_price = volume * close
    ewm_vol_price: pd.Series = vol_x_price.ewm(  # type: ignore[assignment]
        span=config.turnover_window, min_periods=config.turnover_window
    ).mean()
    ewm_vol: pd.Series = volume.ewm(  # type: ignore[assignment]
        span=config.turnover_window, min_periods=config.turnover_window
    ).mean()
    ref_price = ewm_vol_price / ewm_vol.clip(lower=1e-10)
    df["reference_price"] = ref_price

    # --- Capital Gains Overhang ---
    # CGO = (current_price - reference_price) / reference_price
    # 양수 = 미실현 이익 (disposition effect → 매도 압력 → 모멘텀 지속)
    # 음수 = 미실현 손실 (loss aversion → 매도 저항 → 반등 가능)
    cgo = (close - ref_price) / ref_price.clip(lower=1e-10)
    df["cgo"] = cgo

    # --- CGO Z-Score ---
    rolling_mean: pd.Series = cgo.rolling(  # type: ignore[assignment]
        window=config.cgo_zscore_window, min_periods=config.cgo_zscore_window
    ).mean()
    rolling_std: pd.Series = cgo.rolling(  # type: ignore[assignment]
        window=config.cgo_zscore_window, min_periods=config.cgo_zscore_window
    ).std()
    df["cgo_zscore"] = (cgo - rolling_mean) / rolling_std.clip(lower=1e-10)

    # --- Drawdown (HEDGE_ONLY용) ---
    df["drawdown"] = drawdown(close)

    return df
