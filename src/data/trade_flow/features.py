"""Trade flow feature computation — BT/Live 공유 순수 함수.

aggTrades 데이터에서 12H bar-level trade flow 피처를 계산한다.
BT(batch)와 Live(streaming accumulator) 모두 이 함수들을 공유하여 parity를 보장.

Features:
    - tflow_cvd: (buy_vol - sell_vol) / total_vol — 순 매수 압력
    - tflow_buy_ratio: buy_vol / total_vol — 매수 비율
    - tflow_intensity: trade_count / bar_hours — 거래 강도
    - tflow_large_ratio: large_trade_vol / total_vol — 대형 거래 비중
    - tflow_vpin: rolling_mean(|buy_vol - sell_vol| / total_vol, N) — 정보 비대칭 확률
"""

import numpy as np
import pandas as pd


def compute_bar_features(
    trades_df: pd.DataFrame,
    bar_hours: float = 12.0,
    large_pct: float = 95.0,
) -> dict[str, float]:
    """12H bar 내 aggTrades → trade flow 피처 계산.

    BT(batch)와 Live(streaming) 모두 이 함수를 공유하여 parity 보장.

    Args:
        trades_df: aggTrades DataFrame. 필수 컬럼:
            - quantity (float): 거래 수량
            - is_buyer_maker (bool): True=taker sell, False=taker buy
        bar_hours: bar 길이 (시간 단위). 기본 12H.
        large_pct: 대형 거래 판정 percentile. 기본 95.

    Returns:
        dict with keys: tflow_cvd, tflow_buy_ratio, tflow_intensity,
        tflow_large_ratio, tflow_abs_order_imbalance.

    Notes:
        - is_buyer_maker=False → taker buy (매수 공격)
        - is_buyer_maker=True  → taker sell (매도 공격)
        - 빈 DataFrame → 모든 피처 0.0
    """
    empty_result: dict[str, float] = {
        "tflow_cvd": 0.0,
        "tflow_buy_ratio": 0.0,
        "tflow_intensity": 0.0,
        "tflow_large_ratio": 0.0,
        "tflow_abs_order_imbalance": 0.0,
    }

    if trades_df.empty:
        return empty_result

    qty = trades_df["quantity"].to_numpy(dtype=np.float64)
    is_buyer_maker = trades_df["is_buyer_maker"].to_numpy(dtype=bool)

    # buy = taker buy (is_buyer_maker=False), sell = taker sell (is_buyer_maker=True)
    buy_mask = ~is_buyer_maker
    buy_vol = float(qty[buy_mask].sum())
    sell_vol = float(qty[~buy_mask].sum())
    total_vol = buy_vol + sell_vol

    if total_vol == 0.0:
        return empty_result

    trade_count = len(trades_df)

    # CVD: 순 매수 압력 [-1, 1]
    cvd = (buy_vol - sell_vol) / total_vol

    # Buy ratio: 매수 비율 [0, 1]
    buy_ratio = buy_vol / total_vol

    # Intensity: 거래 강도 (trades/hour)
    intensity = trade_count / bar_hours if bar_hours > 0 else 0.0

    # Large trade ratio: 대형 거래 비중
    if trade_count >= 20:  # noqa: PLR2004 — 통계적 최소 표본
        threshold = float(np.percentile(qty, large_pct))
        large_vol = float(qty[qty > threshold].sum())
        large_ratio = large_vol / total_vol
    else:
        large_ratio = 0.0

    # Absolute order imbalance: |buy - sell| / total (VPIN 계산용)
    abs_oi = abs(buy_vol - sell_vol) / total_vol

    return {
        "tflow_cvd": cvd,
        "tflow_buy_ratio": buy_ratio,
        "tflow_intensity": intensity,
        "tflow_large_ratio": large_ratio,
        "tflow_abs_order_imbalance": abs_oi,
    }


def compute_vpin(
    bar_features: pd.DataFrame,
    window: int = 10,
) -> pd.Series:
    """Rolling VPIN (bar-level 근사).

    Volume-Synchronized Probability of Informed Trading.
    각 bar의 |buy_vol - sell_vol| / total_vol의 rolling mean.

    Args:
        bar_features: 연속된 bar 피처 DataFrame.
            필수 컬럼: tflow_abs_order_imbalance
        window: Rolling window 크기 (bar 수). 기본 10.

    Returns:
        pd.Series: tflow_vpin 값. 범위 [0, 1].
    """
    if bar_features.empty or "tflow_abs_order_imbalance" not in bar_features.columns:
        return pd.Series(dtype=np.float64, name="tflow_vpin")

    vpin: pd.Series = (
        bar_features["tflow_abs_order_imbalance"].rolling(window=window, min_periods=1).mean()
    )  # type: ignore[assignment]
    vpin.name = "tflow_vpin"
    return vpin
