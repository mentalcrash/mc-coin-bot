"""VW-TSMOM Signal Generator.

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
VectorBT 및 QuantStats와 호환되는 표준 출력을 제공합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization (No loops)
    - #26 VectorBT Standards: entries/exits as bool Series
    - Shift(1) Rule: 미래 참조 편향 방지
"""

import numpy as np
import pandas as pd

from src.strategy.tsmom.config import TSMOMConfig
from src.strategy.types import Direction, StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: TSMOMConfig | None = None,
) -> StrategySignals:
    """VW-TSMOM 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Important:
        - 입력 DataFrame에는 preprocess()로 계산된 컬럼이 필요합니다.
        - 필수 컬럼: raw_signal
        - entries/exits는 bool Series
        - direction은 -1, 0, 1 값을 가지는 int Series
        - strength는 순수 시그널 강도 (레버리지 제한 미적용)

    Note:
        레버리지 클램핑(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 처리됩니다. 전략은 순수한 시그널만 생성합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
            필수 컬럼: raw_signal
        config: TSMOM 설정 (미사용, 하위 호환성 유지)

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
        >>> signals = generate_signals(processed_df)
        >>> signals.entries  # pd.Series[bool]
        >>> signals.strength  # pd.Series[float] (unbounded)
    """
    # config 파라미터는 하위 호환성을 위해 유지 (미사용)
    _ = config

    # 입력 검증
    if "raw_signal" not in df.columns:
        msg = "Missing required column: 'raw_signal'. Run preprocess() first."
        raise ValueError(msg)

    # 1. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    # 현재 봉의 시그널은 전봉까지의 데이터로 계산된 값을 사용
    signal_series: pd.Series = df["raw_signal"]  # type: ignore[assignment]
    signal_shifted: pd.Series = signal_series.shift(1)  # type: ignore[assignment]

    # 2. 방향 계산 (-1, 0, 1)
    direction_raw = pd.Series(np.sign(signal_shifted), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 3. 강도 계산 (순수 시그널, 레버리지 무제한)
    # PortfolioManagerConfig.max_leverage_cap에서 클램핑 처리
    strength = pd.Series(
        signal_shifted.fillna(0),
        index=df.index,
        name="strength",
    )

    # 4. 진입 시그널: 포지션이 0에서 non-zero로 변할 때
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

    # 5. 청산 시그널: 포지션이 non-zero에서 0으로 변할 때
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

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )


def generate_signals_for_long_only(
    df: pd.DataFrame,
    config: TSMOMConfig | None = None,
) -> StrategySignals:
    """롱 온리 VW-TSMOM 시그널 생성.

    숏 포지션을 허용하지 않는 환경(현물)에서 사용합니다.
    숏 시그널은 중립(현금)으로 처리됩니다.

    Args:
        df: 전처리된 DataFrame
        config: TSMOM 설정

    Returns:
        StrategySignals (롱 온리)
    """
    # 기본 시그널 생성
    signals = generate_signals(df, config)

    # 숏 시그널을 중립으로 변환
    direction_long_only = signals.direction.clip(lower=0)
    strength_long_only = signals.strength.clip(lower=0)

    # 진입/청산 재계산
    prev_direction = direction_long_only.shift(1).fillna(0)
    entries_long_only = (direction_long_only == Direction.LONG) & (
        prev_direction != Direction.LONG
    )
    exits_long_only = (direction_long_only == Direction.NEUTRAL) & (
        prev_direction == Direction.LONG
    )

    return StrategySignals(
        entries=entries_long_only,
        exits=exits_long_only,
        direction=direction_long_only,
        strength=strength_long_only,
    )


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
