"""Adaptive Breakout Signal Generation.

이 모듈은 Adaptive Breakout 전략의 매매 시그널을 생성합니다.
모든 계산은 벡터화된 연산을 사용합니다 (for 루프 금지).

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Broadcasting compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.types import Direction, StrategySignals

if TYPE_CHECKING:
    from src.strategy.breakout.config import AdaptiveBreakoutConfig


def detect_breakout(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    threshold: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """돌파 시그널을 감지합니다.

    Args:
        close: 종가 Series
        upper_band: Donchian Channel 상단 (전봉 기준)
        lower_band: Donchian Channel 하단 (전봉 기준)
        threshold: 돌파 확인 임계값 (선택적)

    Returns:
        (long_breakout, short_breakout) bool Series 튜플

    Note:
        Shift(1) Rule: 전봉의 밴드를 기준으로 현재 종가가 돌파했는지 확인
    """
    # 전봉 밴드 기준 돌파 확인 (Look-Ahead Bias 방지)
    prev_upper = upper_band.shift(1)
    prev_lower = lower_band.shift(1)

    if threshold is not None:
        # 임계값만큼 추가로 돌파해야 진입
        prev_threshold = threshold.shift(1)
        long_breakout: pd.Series = close > (prev_upper + prev_threshold)
        short_breakout: pd.Series = close < (prev_lower - prev_threshold)
    else:
        # 단순 돌파
        long_breakout = close > prev_upper
        short_breakout = close < prev_lower

    return long_breakout, short_breakout


def detect_exit_signals(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
    middle_band: pd.Series,
    direction: pd.Series,
    use_trailing_stop: bool = False,
    trailing_stop_distance: pd.Series | None = None,
    prev_high: pd.Series | None = None,
    prev_low: pd.Series | None = None,
) -> pd.Series:
    """청산 시그널을 감지합니다.

    청산 조건:
        - Long: 가격이 middle_band 또는 lower_band 아래로 하락
        - Short: 가격이 middle_band 또는 upper_band 위로 상승
        - Trailing Stop: 최고/최저점 대비 ATR * multiplier 이상 역행

    Args:
        close: 종가 Series
        upper_band: Donchian Channel 상단
        lower_band: Donchian Channel 하단
        middle_band: Donchian Channel 중심선
        direction: 현재 방향 Series (-1, 0, 1)
        use_trailing_stop: Trailing Stop 사용 여부
        trailing_stop_distance: Trailing Stop 거리 (ATR * multiplier)
        prev_high: 포지션 진입 후 최고가 (Trailing Stop용)
        prev_low: 포지션 진입 후 최저가 (Trailing Stop용)

    Returns:
        청산 시그널 bool Series
    """
    # 전봉 밴드 기준 (Look-Ahead Bias 방지)
    prev_middle = middle_band.shift(1)

    # Long 청산: 가격이 middle 아래로
    long_exit = (direction == Direction.LONG) & (close < prev_middle)

    # Short 청산: 가격이 middle 위로
    short_exit = (direction == Direction.SHORT) & (close > prev_middle)

    exit_signal: pd.Series = long_exit | short_exit

    # Trailing Stop 적용 (선택적)
    if use_trailing_stop and trailing_stop_distance is not None:
        if prev_high is not None:
            # Long Trailing Stop: 최고가 대비 trailing_stop_distance 하락
            long_trailing = (direction == Direction.LONG) & (
                close < prev_high - trailing_stop_distance
            )
            exit_signal = exit_signal | long_trailing

        if prev_low is not None:
            # Short Trailing Stop: 최저가 대비 trailing_stop_distance 상승
            short_trailing = (direction == Direction.SHORT) & (
                close > prev_low + trailing_stop_distance
            )
            exit_signal = exit_signal | short_trailing

    return exit_signal


def apply_cooldown(
    signal: pd.Series,
    cooldown_periods: int,
) -> pd.Series:
    """쿨다운 기간을 적용합니다 (벡터화 구현).

    진입 시그널 후 cooldown_periods 동안 재진입을 방지합니다.
    Rolling window를 활용한 벡터화로 대규모 데이터에서도 빠르게 동작합니다.

    Args:
        signal: 원본 진입 시그널 (bool Series)
        cooldown_periods: 쿨다운 기간 (캔들 수)

    Returns:
        쿨다운이 적용된 시그널 bool Series
    """
    if cooldown_periods <= 0:
        return signal

    # 벡터화된 쿨다운 적용
    # 1. 시그널을 int로 변환 (True=1, False=0)
    signal_int = signal.astype(int)

    # 2. 과거 cooldown_periods 동안 시그널이 있었는지 확인 (현재 제외)
    # rolling sum에서 현재 값을 빼면 "과거 N개" 합계가 됨
    # min_periods=1로 초기 NaN 방지
    past_signals: pd.Series = (
        signal_int.rolling(window=cooldown_periods + 1, min_periods=1).sum()
        - signal_int
    )

    # 3. 과거에 시그널이 있었으면(past_signals > 0) 현재 시그널 억제
    # 단, 현재 시그널이 True이고 과거에 시그널이 없었으면 통과
    cooldown_active = past_signals > 0

    # 4. 현재 시그널이 True이고 쿨다운 중이 아닐 때만 True
    result: pd.Series = signal & ~cooldown_active

    return result


def calculate_position_size(
    vol_scalar: pd.Series,
    direction: pd.Series,
    min_size: float = 0.0,
    max_size: float = 3.0,
) -> pd.Series:
    """포지션 크기를 계산합니다.

    변동성 스케일링을 적용하여 포지션 크기를 결정합니다.

    Args:
        vol_scalar: 변동성 스케일러 (vol_target / realized_vol)
        direction: 방향 Series (-1, 0, 1)
        min_size: 최소 포지션 크기
        max_size: 최대 포지션 크기

    Returns:
        포지션 크기 Series (방향 포함, -max_size ~ +max_size)
    """
    # 절대 크기 계산 및 클램핑
    abs_size = vol_scalar.clip(lower=min_size, upper=max_size)

    # 방향 적용
    position_size: pd.Series = abs_size * direction

    return position_size


def generate_signals(
    df: pd.DataFrame,
    config: AdaptiveBreakoutConfig,
) -> StrategySignals:
    """Adaptive Breakout 매매 시그널을 생성합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: 전략 설정

    Returns:
        StrategySignals NamedTuple (entries, exits, direction, strength)
    """
    # 필요한 컬럼 추출
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    upper_band: pd.Series = df["upper_band"]  # type: ignore[assignment]
    lower_band: pd.Series = df["lower_band"]  # type: ignore[assignment]
    middle_band: pd.Series = df["middle_band"]  # type: ignore[assignment]
    vol_scalar: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    # 임계값 (adaptive_threshold가 True면 동적, 아니면 고정)
    threshold: pd.Series | None = None
    if config.adaptive_threshold and "threshold" in df.columns:
        threshold = df["threshold"]  # type: ignore[assignment]

    # 1. 돌파 시그널 감지
    long_breakout, short_breakout = detect_breakout(
        close_series, upper_band, lower_band, threshold
    )

    # 2. Long-Only 모드 처리
    if config.long_only:
        short_breakout = pd.Series(False, index=df.index)

    # 3. 방향 결정
    direction = pd.Series(Direction.NEUTRAL.value, index=df.index)
    direction = direction.where(~long_breakout, Direction.LONG.value)
    direction = direction.where(~short_breakout, Direction.SHORT.value)

    # 4. 쿨다운 적용
    if config.cooldown_periods > 0:
        entries_raw = long_breakout | short_breakout
        entries_cooled = apply_cooldown(entries_raw, config.cooldown_periods)
        # 쿨다운으로 억제된 시그널의 방향도 NEUTRAL로
        direction = direction.where(entries_cooled | (direction == 0), Direction.NEUTRAL.value)
        long_breakout = long_breakout & entries_cooled
        short_breakout = short_breakout & entries_cooled

    # 5. 진입 시그널
    entries = long_breakout | short_breakout

    # 6. 청산 시그널 (기본: middle band 기준)
    # 실제 청산은 백테스트 엔진에서 처리, 여기서는 기본 청산 조건만 제공
    exits = detect_exit_signals(
        close_series,
        upper_band,
        lower_band,
        middle_band,
        direction,
        use_trailing_stop=False,  # 백테스트 엔진에서 처리
    )

    # 7. 포지션 크기 계산 (변동성 스케일링)
    strength = calculate_position_size(vol_scalar, direction)

    # NaN 처리
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    direction = direction.fillna(Direction.NEUTRAL.value).astype(int)
    strength = strength.fillna(0.0)

    return StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )


def generate_signals_with_diagnostics(
    df: pd.DataFrame,
    config: AdaptiveBreakoutConfig,
) -> tuple[StrategySignals, pd.DataFrame]:
    """진단 데이터와 함께 매매 시그널을 생성합니다.

    Args:
        df: 전처리된 DataFrame (preprocess() 출력)
        config: 전략 설정

    Returns:
        (StrategySignals, diagnostics_df) 튜플
    """
    # 기본 시그널 생성
    signals = generate_signals(df, config)

    # 진단 데이터 수집
    close_series: pd.Series = df["close"]  # type: ignore[assignment]
    upper_band: pd.Series = df["upper_band"]  # type: ignore[assignment]
    lower_band: pd.Series = df["lower_band"]  # type: ignore[assignment]

    # 돌파 유형 결정
    breakout_type = pd.Series("none", index=df.index)
    breakout_type = breakout_type.where(
        ~(signals.direction == Direction.LONG.value), "upper"
    )
    breakout_type = breakout_type.where(
        ~(signals.direction == Direction.SHORT.value), "lower"
    )

    # 밴드까지 거리
    distance_to_upper: pd.Series = (upper_band - close_series) / close_series * 100
    distance_to_lower: pd.Series = (close_series - lower_band) / close_series * 100

    diagnostics_df = pd.DataFrame(
        {
            "close": close_series,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "atr": df.get("atr", np.nan),
            "breakout_type": breakout_type,
            "distance_to_upper": distance_to_upper,
            "distance_to_lower": distance_to_lower,
            "volatility_ratio": df.get("volatility_ratio", 1.0),
            "raw_direction": signals.direction,
            "final_strength": signals.strength,
            "entry": signals.entries,
            "exit": signals.exits,
        },
        index=df.index,
    )

    return signals, diagnostics_df
