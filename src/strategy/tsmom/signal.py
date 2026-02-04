"""VW-TSMOM Signal Generator.

이 모듈은 전처리된 데이터에서 매매 시그널을 생성합니다.
VectorBT 및 QuantStats와 호환되는 표준 출력을 제공합니다.

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
    """VW-TSMOM 시그널 생성.

    전처리된 DataFrame에서 진입/청산 시그널과 강도를 계산합니다.
    Shift(1) Rule을 적용하여 미래 참조 편향을 방지합니다.

    Signal Generation Pipeline:
        1. scaled_momentum 계산: vw_momentum * vol_scalar
        2. Shift(1) 적용: 미래 참조 편향 방지
        3. Deadband 적용: 노이즈 필터링
        4. Trend Filter 적용: 국면 반대 방향 시그널 제거
        5. Entry/Exit 시그널 생성

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
        config: TSMOM 설정 (deadband, use_zscore 등)

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
    # 기본 config 설정
    if config is None:
        config = TSMOMConfig()

    # 입력 검증
    required_cols = {"vw_momentum", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Scaled Momentum 계산 (시그널의 원재료)
    momentum_series: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    if config.use_zscore:
        # Z-Score 모드: 모멘텀 강도 자체가 이미 정규화됨
        # 모멘텀 강도를 직접 사용하고 vol_scalar로 목표 변동성에 맞춰 스케일링
        scaled_momentum = momentum_series * vol_scalar_series
    else:
        # 기존 모드: 방향만 추출하고 vol_scalar로 크기 조절
        momentum_direction = np.sign(momentum_series)
        scaled_momentum = momentum_direction * vol_scalar_series

    # 2. Shift(1) 적용: 전봉 기준 시그널 (미래 참조 편향 방지)
    # 현재 봉의 시그널은 전봉까지의 데이터로 계산된 값을 사용
    signal_shifted: pd.Series = scaled_momentum.shift(1)  # type: ignore[assignment]

    # 3. Deadband 적용 (shift 후): 노이즈 필터링
    # |momentum| < threshold 인 경우 신호를 0으로
    signal_filtered = signal_shifted.copy()

    if config.deadband_threshold > 0:
        # shift된 momentum 값으로 판단해야 함
        momentum_shifted = momentum_series.shift(1)
        deadband_mask = np.abs(momentum_shifted) < config.deadband_threshold
        signal_filtered = pd.Series(
            np.where(deadband_mask, 0, signal_filtered),
            index=df.index,
        )

        # 통계 로깅
        filtered_count = int(deadband_mask.sum())
        total_count = len(momentum_shifted.dropna())
        if total_count > 0:
            filtered_pct = filtered_count / total_count * 100
            logger.info(
                "🚫 Deadband | Threshold: %.2f, Filtered: %d/%d (%.1f%%)",
                config.deadband_threshold,
                filtered_count,
                total_count,
                filtered_pct,
            )

    # 4. Trend Filter 적용 (shift 후): 국면 반대 방향 시그널 제거
    if "trend_regime" in df.columns:
        trend_regime: pd.Series = df["trend_regime"]  # type: ignore[assignment]
        trend_regime_shifted = trend_regime.shift(1)

        # 상승장(shift된)인데 숏 신호(shift된)면 0으로
        signal_filtered_array = np.where(
            (trend_regime_shifted == 1) & (signal_filtered < 0), 0, signal_filtered
        )
        # 하락장(shift된)인데 롱 신호(shift된)면 0으로
        signal_filtered_array = np.where(
            (trend_regime_shifted == -1) & (signal_filtered > 0),
            0,
            signal_filtered_array,
        )
        # numpy array를 Series로 변환
        signal_filtered = pd.Series(signal_filtered_array, index=df.index)

    # 5. Direction 계산 (필터링된 시그널에서)
    direction_raw = pd.Series(np.sign(signal_filtered), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    # 6. 강도 계산 (필터링된 시그널 사용)
    strength = pd.Series(
        signal_filtered.fillna(0),
        index=df.index,
        name="strength",
    )

    # 7. 진입 시그널: 포지션이 0에서 non-zero로 변할 때
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

    # 8. 청산 시그널: 포지션이 non-zero에서 0으로 변할 때
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

    # 🔍 디버그: 시그널 통계
    valid_strength = strength[strength != 0]
    long_signals = strength[strength > 0]
    short_signals = strength[strength < 0]

    logger.info(
        f"📊 Signal Statistics | Total: {len(valid_strength)} signals, Long: {len(long_signals)} ({len(long_signals) / len(valid_strength) * 100 if len(valid_strength) > 0 else 0:.1f}%), Short: {len(short_signals)} ({len(short_signals) / len(valid_strength) * 100 if len(valid_strength) > 0 else 0:.1f}%)",
    )
    logger.info(
        f"🎯 Entry/Exit Events | Long entries: {long_entry.sum()}, Short entries: {short_entry.sum()}, Exits: {exits.sum()}, Reversals: {reversal.sum()}",
    )

    # 샘플 롱/숏 진입 시점
    if long_entry.sum() > 0:
        first_long = long_entry[long_entry].index[0]
        logger.info(
            f"  📈 First Long Entry: {first_long}, Strength: {strength.loc[first_long]:.2f}"
        )
    if short_entry.sum() > 0:
        first_short = short_entry[short_entry].index[0]
        logger.info(
            f"  📉 First Short Entry: {first_short}, Strength: {strength.loc[first_short]:.2f}"
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
    각 필터 단계의 중간 값을 기록하여 Beta Attribution 분석에 사용합니다.

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
    # Lazy import to avoid circular dependency
    from src.strategy.tsmom.diagnostics import collect_diagnostics_from_pipeline

    # 기본 config 설정
    if config is None:
        config = TSMOMConfig()

    # 입력 검증
    required_cols = {"vw_momentum", "vol_scalar"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}. Run preprocess() first."
        raise ValueError(msg)

    # 1. Scaled Momentum 계산 (시그널의 원재료)
    momentum_series: pd.Series = df["vw_momentum"]  # type: ignore[assignment]
    vol_scalar_series: pd.Series = df["vol_scalar"]  # type: ignore[assignment]

    if config.use_zscore:
        scaled_momentum = momentum_series * vol_scalar_series
    else:
        momentum_direction = np.sign(momentum_series)
        scaled_momentum = momentum_direction * vol_scalar_series

    # 2. Shift(1) 적용
    signal_shifted: pd.Series = scaled_momentum.shift(1)  # type: ignore[assignment]

    # 3. Deadband 적용
    signal_after_deadband = signal_shifted.copy()
    deadband_mask = pd.Series(False, index=df.index)

    if config.deadband_threshold > 0:
        momentum_shifted = momentum_series.shift(1)
        deadband_mask = pd.Series(
            np.abs(momentum_shifted) < config.deadband_threshold,
            index=df.index,
        )
        signal_after_deadband = pd.Series(
            np.where(deadband_mask, 0, signal_shifted),
            index=df.index,
        )

    # 📊 진단: Trend Filter 적용 전 시그널 저장
    signal_before_trend = signal_after_deadband.copy()

    # 4. Trend Filter 적용
    signal_after_trend = signal_after_deadband.copy()

    if "trend_regime" in df.columns:
        trend_regime: pd.Series = df["trend_regime"]  # type: ignore[assignment]
        trend_regime_shifted = trend_regime.shift(1)

        signal_filtered_array = np.where(
            (trend_regime_shifted == 1) & (signal_after_deadband < 0),
            0,
            signal_after_deadband,
        )
        signal_filtered_array = np.where(
            (trend_regime_shifted == -1) & (signal_filtered_array > 0),
            0,
            signal_filtered_array,
        )
        signal_after_trend = pd.Series(signal_filtered_array, index=df.index)

    # 5. Direction & Strength 계산
    signal_filtered = signal_after_trend
    direction_raw = pd.Series(np.sign(signal_filtered), index=df.index)
    direction = pd.Series(
        direction_raw.fillna(0).astype(int),
        index=df.index,
        name="direction",
    )

    strength = pd.Series(
        signal_filtered.fillna(0),
        index=df.index,
        name="strength",
    )

    # 6. 진입/청산 시그널 생성
    prev_direction = direction.shift(1).fillna(0)
    long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)

    entries = pd.Series(
        long_entry | short_entry,
        index=df.index,
        name="entries",
    )

    to_neutral = (direction == Direction.NEUTRAL) & (
        prev_direction != Direction.NEUTRAL
    )
    reversal = direction * prev_direction < 0

    exits = pd.Series(
        to_neutral | reversal,
        index=df.index,
        name="exits",
    )

    # 📊 진단 DataFrame 생성
    # NOTE: leverage_capped_weight와 rebalance_mask는 PortfolioManager에서 처리되므로
    # 여기서는 strength를 raw_target_weight로 사용
    diagnostics_df = collect_diagnostics_from_pipeline(
        processed_df=df,
        symbol=symbol,
        signal_before_trend=signal_before_trend,
        signal_after_trend=signal_after_trend,
        signal_after_deadband=signal_after_deadband,
        deadband_mask=deadband_mask,
        final_weights=strength,
    )

    signals = StrategySignals(
        entries=entries,
        exits=exits,
        direction=direction,
        strength=strength,
    )

    return SignalsWithDiagnostics(signals=signals, diagnostics_df=diagnostics_df)


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
