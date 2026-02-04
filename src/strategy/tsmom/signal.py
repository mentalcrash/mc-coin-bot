"""VW-TSMOM Signal Generator (Pure TSMOM + Vol Target).

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
VectorBT 및 QuantStats와 호환되는 표준 출력을 제공합니다.

Signal Formula:
    1. scaled_momentum = sign(vw_momentum) * vol_scalar
    2. direction = sign(scaled_momentum)
    3. strength = scaled_momentum (변동성 스케일링된 시그널 강도)

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.strategy.tsmom.config import TSMOMConfig
from src.strategy.types import Direction, StrategySignals

logger = logging.getLogger(__name__)


class SignalsWithDiagnostics(NamedTuple):
    """시그널과 진단 데이터를 함께 반환하는 결과 타입."""

    signals: StrategySignals
    diagnostics_df: pd.DataFrame


def generate_signals(
    df: pd.DataFrame,
    config: TSMOMConfig | None = None,
) -> StrategySignals:
    """VW-TSMOM 시그널 생성 (Pure TSMOM + Vol Target).

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. scaled_momentum 계산: sign(vw_momentum) * vol_scalar
        2. Shift(1) 적용: 미래 참조 편향 방지
        3. Entry/Exit 시그널 생성

    Important:
        - 입력 DataFrame에는 preprocess()로 계산된 지표가 필요합니다.
        - 필수 컬럼: vw_momentum, vol_scalar
        - entries/exits는 bool Series
        - direction은 -1, 0, 1 값을 가지는 int Series
        - strength는 순수 시그널 강도 (레버리지 제한 미적용)

    Note:
        레버리지 클램핑(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 처리됩니다. 전략은 순수한 시그널만 생성합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: vw_momentum, vol_scalar
        config: TSMOM 설정 (현재 사용되지 않음, 향후 확장용)

    Returns:
        StrategySignals NamedTuple:
            - entries: 진입 시그널 (bool Series)
            - exits: 청산 시그널 (bool Series)
            - direction: 방향 시리즈 (-1, 0, 1)
            - strength: 시그널 강도 (레버리지 무제한)

    Raises:
        ValueError: 필수 컬럼 누락 시

    Example:
        >>> from src.strategy.tsmom.preprocessor import preprocess
        >>> processed_df = preprocess(ohlcv_df, config)
        >>> signals = generate_signals(processed_df, config)
        >>> signals.entries  # pd.Series[bool]
        >>> signals.strength  # pd.Series[float] (unbounded)
    """
    # 기본 config 설정 (현재는 사용하지 않지만 인터페이스 유지)
    if config is None:
        config = TSMOMConfig()

    # 입력 검증
    required_cols = {"vw_momentum", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Scaled Momentum 계산 (시그널의 원재료)
    # 모멘텀 방향 * 변동성 스케일러 = 변동성 조정된 시그널
    momentum_series: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 모멘텀 방향 추출하고 vol_scalar로 크기 조절
    momentum_direction = np.sign(momentum_series)
    scaled_momentum = momentum_direction * vol_scalar_series

    # 2. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    # 현재 봉의 시그널은 전봉까지의 데이터로 계산된 값을 사용
    signal_shifted: pd.Series = scaled_momentum.shift(1)  # type: ignore[assignment]

    # 3. Direction 계산
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 4. 강도 계산
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # 5. 진입 시그널: 포지션이 0에서 non-zero로 변할 때
    prev_direction = direction.shift(1).fillna(0)

    # Long 진입: direction이 1이 되는 순간 (이전이 0 또는 -1)
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)

    # Short 진입: direction이 -1이 되는 순간 (이전이 0 또는 1)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    # 전체 진입 시그널
    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    # 6. 청산 시그널: 포지션이 non-zero에서 0으로 변할 때
    # 또는 방향이 반전될 때
    to_neutral = (direction == Direction.NEUTRAL) & (
        prev_direction != Direction.NEUTRAL
    )
    reversal = direction * prev_direction < 0  # 부호가 바뀌면 반전

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 디버그: 시그널 통계
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    if len(valid_strength) > 0:
        logger.info(
            "📊 Signal Statistics | Total: %d signals, Long: %d (%.1f%%), Short: %d (%.1f%%)",
            len(valid_strength),
            len(long_signals),
            len(long_signals) / len(valid_strength) * 100,
            len(short_signals),
            len(short_signals) / len(valid_strength) * 100,
        )
        logger.info(
            "🎯 Entry/Exit Events | Long entries: %d, Short entries: %d, Exits: %d, Reversals: %d",
            int(long_entry.sum()),
            int(short_entry.sum()),
            int(exits.sum()),
            int(reversal.sum()),
        )

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )


def generate_signals_with_diagnostics(
    df: pd.DataFrame,
    config: TSMOMConfig | None = None,
    symbol: str = "UNKNOWN",
) -> SignalsWithDiagnostics:
    """VW-TSMOM 시그널 생성 + 진단 데이터 수집.

    generate_signals()와 동일한 시그널 생성 로직을 수행하되,
    진단 데이터를 함께 반환합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: TSMOM 설정
        symbol: 거래 심볼 (진단 로깅용)

    Returns:
        SignalsWithDiagnostics:
            - signals: StrategySignals NamedTuple
            - diagnostics_df: 진단 레코드 DataFrame

    Example:
        >>> result = generate_signals_with_diagnostics(processed_df, config, "BTC/USDT")
        >>> signals = result.signals
        >>> diagnostics = result.diagnostics_df
    """
    # 기본 config 설정
    if config is None:
        config = TSMOMConfig()

    # 입력 검증
    required_cols = {"vw_momentum", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 시그널 생성
    signals = generate_signals(df, config)

    # 진단 DataFrame 생성
    momentum_series: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 벤치마크 수익률 계산
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    benchmark_returns = close_series.pct_change().fillna(0)

    diagnostics_df = pd.DataFrame(
        {
            "symbol": symbol,
            "close_price": df["close"],
            "realized_vol_annualized": df.get("realized_vol", 0.0),
            "benchmark_return": benchmark_returns,
            "raw_momentum": momentum_series,
            "vol_scalar": vol_scalar_series,
            "scaled_momentum": signals.strength,
            "final_target_weight": signals.strength,
            "signal_suppression_reason": "none",
        },
        index=df.index,
    )

    return SignalsWithDiagnostics(signals=signals, diagnostics_df=diagnostics_df)


def get_current_signal(df: pd.DataFrame) -> tuple[Direction, float]:
    """현재(최신) 시그널 반환.

    라이브 트레이딩에서 현재 시점의 시그널을 가져올 때 사용합니다.

    Args:
        df: 전처리된 DataFrame (최신이 마지막)

    Returns:
        (방향, 강도) 튜플

    Example:
        >>> direction, strength = get_current_signal(processed_df)
        >>> if direction == Direction.LONG:
        ...     place_long_order(strength)
    """
    if df.empty:
        return Direction.NEUTRAL, 0.0

    signals = generate_signals(df)

    # 마지막 행 (최신 시그널)
    current_direction = Direction(int(signals.direction.iloc[-1]))
    current_strength = float(signals.strength.iloc[-1])

    return current_direction, current_strength
