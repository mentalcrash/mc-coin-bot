"""Beta Attribution Analysis.

이 모듈은 TSMOM 전략의 Beta 분해 분석을 수행합니다.
각 필터(Trend Filter, Deadband, Vol Scaling)가 Beta에 미치는 영향을 정량화합니다.

Rules Applied:
    - #12 Data Engineering: Vectorized pandas/numpy 연산
    - #11 Pydantic Modeling: BetaAttributionResult 반환
    - #15 Logging Standards: 분석 결과 로깅
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.models.backtest import BetaAttributionResult

logger = logging.getLogger(__name__)

# 최소 데이터 포인트 수
MIN_DATA_POINTS = 2


def calculate_rolling_beta(
    strategy_returns: pd.Series,  # type: ignore[type-arg]
    benchmark_returns: pd.Series,  # type: ignore[type-arg]
    window: int = 60,
) -> pd.Series:  # type: ignore[type-arg]
    """Rolling Beta를 계산합니다.

    Beta = Cov(strategy, benchmark) / Var(benchmark)

    Args:
        strategy_returns: 전략 수익률 시리즈
        benchmark_returns: 벤치마크 수익률 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Rolling Beta 시리즈
    """
    cov: pd.Series = strategy_returns.rolling(  # type: ignore[type-arg, assignment]
        window=window, min_periods=window // 2
    ).cov(benchmark_returns)
    var: pd.Series = benchmark_returns.rolling(  # type: ignore[type-arg, assignment]
        window=window, min_periods=window // 2
    ).var()

    # 0으로 나누기 방지
    var_safe: pd.Series = var.replace(0, np.nan)  # type: ignore[type-arg, assignment]
    return cov / var_safe  # type: ignore[return-value]


def calculate_overall_beta(
    strategy_returns: pd.Series,  # type: ignore[type-arg]
    benchmark_returns: pd.Series,  # type: ignore[type-arg]
) -> float:
    """전체 기간 Beta를 계산합니다.

    Args:
        strategy_returns: 전략 수익률 시리즈
        benchmark_returns: 벤치마크 수익률 시리즈

    Returns:
        전체 기간 Beta 값
    """
    # NaN 제거 후 정렬
    aligned = pd.DataFrame(
        {"strategy": strategy_returns, "benchmark": benchmark_returns}
    ).dropna()

    if len(aligned) < MIN_DATA_POINTS:
        return 0.0

    s_strat = cast("pd.Series", aligned["strategy"])
    s_bench = cast("pd.Series", aligned["benchmark"])
    cov = float(s_strat.cov(s_bench))
    var = float(s_bench.var())

    if var == 0 or np.isnan(var):
        return 0.0

    return float(cov / var)


def calculate_hypothetical_returns(
    diagnostics_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """각 필터 단계별 가상 수익률을 계산합니다.

    Args:
        diagnostics_df: 진단 레코드 DataFrame
        benchmark_returns: 벤치마크 수익률 시리즈

    Returns:
        각 단계별 가상 수익률 DataFrame:
            - potential_return: 필터 없이 모든 시그널 실행 시
            - return_after_trend: Trend filter 적용 후
            - return_after_deadband: Deadband 적용 후
            - actual_return: 최종 (Vol scaling 적용 후)
    """
    # 인덱스 정렬
    common_index = diagnostics_df.index.intersection(benchmark_returns.index)
    diag = diagnostics_df.loc[common_index].copy()
    bench = benchmark_returns.loc[common_index]

    # 시그널을 shift(1)하여 이전 봉 시그널로 현재 수익률 계산
    # NOTE: diagnostics_df의 시그널은 이미 shift(1) 적용된 값이므로 그대로 사용

    # 1. Potential Return: 필터 없는 raw momentum * benchmark
    # scaled_momentum은 shift(1) 적용된 값
    potential_signal = diag["scaled_momentum"].fillna(0)
    potential_return = potential_signal * bench

    # 2. Return after Trend Filter
    after_trend_signal = diag["signal_after_trend_filter"].fillna(0)
    return_after_trend = after_trend_signal * bench

    # 3. Return after Deadband
    after_deadband_signal = diag["signal_after_deadband"].fillna(0)
    return_after_deadband = after_deadband_signal * bench

    # 4. Actual Return (final_target_weight * benchmark)
    final_weight = diag["final_target_weight"].fillna(0)
    actual_return = final_weight * bench

    return pd.DataFrame(
        {
            "potential_return": potential_return,
            "return_after_trend": return_after_trend,
            "return_after_deadband": return_after_deadband,
            "actual_return": actual_return,
            "benchmark_return": bench,
        },
        index=common_index,
    )


def calculate_beta_attribution(
    diagnostics_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> BetaAttributionResult:
    """Beta 분해 분석을 수행합니다.

    각 필터 단계별 Beta를 계산하고, 어디서 Beta가 손실되었는지 정량화합니다.

    Analysis Flow:
        1. 각 필터 단계별 가상 수익률 계산
        2. 전체 기간 Beta 계산 (각 단계별)
        3. Beta 손실량 계산 (delta)

    Args:
        diagnostics_df: 진단 레코드 DataFrame (generate_signals_with_diagnostics 출력)
        benchmark_returns: 벤치마크 수익률 시리즈
        window: Rolling Beta 계산용 윈도우 크기

    Returns:
        BetaAttributionResult: Beta 분해 분석 결과

    Example:
        >>> result = calculate_beta_attribution(diagnostics_df, benchmark_returns)
        >>> print(result.summary())
        {'potential_beta': '0.850', 'realized_beta': '0.400', ...}
    """
    # 가상 수익률 계산
    returns_df = calculate_hypothetical_returns(diagnostics_df, benchmark_returns)

    if returns_df.empty:
        logger.warning("No valid data for beta attribution analysis")
        return BetaAttributionResult(
            potential_beta=0.0,
            beta_after_trend_filter=0.0,
            beta_after_deadband=0.0,
            realized_beta=0.0,
            lost_to_trend_filter=0.0,
            lost_to_deadband=0.0,
            lost_to_vol_scaling=0.0,
            analysis_window=window,
            total_periods=0,
        )

    bench = cast("pd.Series", returns_df["benchmark_return"])

    # 전체 기간 Beta 계산 (각 단계별)
    potential_beta = calculate_overall_beta(
        cast("pd.Series", returns_df["potential_return"]), bench
    )
    beta_after_trend = calculate_overall_beta(
        cast("pd.Series", returns_df["return_after_trend"]), bench
    )
    beta_after_deadband = calculate_overall_beta(
        cast("pd.Series", returns_df["return_after_deadband"]), bench
    )
    realized_beta = calculate_overall_beta(
        cast("pd.Series", returns_df["actual_return"]), bench
    )

    # Beta 손실량 계산
    lost_to_trend_filter = potential_beta - beta_after_trend
    lost_to_deadband = beta_after_trend - beta_after_deadband
    lost_to_vol_scaling = beta_after_deadband - realized_beta

    # 로깅
    logger.info(
        "Beta Attribution | Potential: %.3f -> Realized: %.3f (%.1f%% retained)",
        potential_beta,
        realized_beta,
        (realized_beta / potential_beta * 100) if potential_beta != 0 else 0,
    )
    logger.info(
        "  Losses | Trend Filter: %.3f, Deadband: %.3f, Vol Scaling: %.3f",
        lost_to_trend_filter,
        lost_to_deadband,
        lost_to_vol_scaling,
    )

    return BetaAttributionResult(
        potential_beta=potential_beta,
        beta_after_trend_filter=beta_after_trend,
        beta_after_deadband=beta_after_deadband,
        realized_beta=realized_beta,
        lost_to_trend_filter=lost_to_trend_filter,
        lost_to_deadband=lost_to_deadband,
        lost_to_vol_scaling=lost_to_vol_scaling,
        analysis_window=window,
        total_periods=len(returns_df),
    )


def calculate_rolling_beta_attribution(
    diagnostics_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """시계열 Beta Attribution을 계산합니다.

    각 시점에서의 Rolling Beta를 계산하여 시간에 따른 Beta 변화를 분석합니다.

    Args:
        diagnostics_df: 진단 레코드 DataFrame
        benchmark_returns: 벤치마크 수익률 시리즈
        window: Rolling 윈도우 크기

    Returns:
        Rolling Beta Attribution DataFrame:
            - potential_beta: 필터 없이 예상되는 Beta
            - beta_after_trend_filter: Trend filter 적용 후 Beta
            - beta_after_deadband: Deadband 적용 후 Beta
            - realized_beta: 실제 Beta
            - lost_to_trend_filter: Trend filter로 인한 손실
            - lost_to_deadband: Deadband로 인한 손실
            - lost_to_vol_scaling: Vol scaling으로 인한 손실

    Example:
        >>> rolling_df = calculate_rolling_beta_attribution(diag_df, bench_ret)
        >>> rolling_df["realized_beta"].plot()  # 시간에 따른 Beta 변화
    """
    # 가상 수익률 계산
    returns_df = calculate_hypothetical_returns(diagnostics_df, benchmark_returns)

    if returns_df.empty:
        return pd.DataFrame()

    bench = cast("pd.Series", returns_df["benchmark_return"])

    # Rolling Beta 계산 (각 단계별)
    potential_beta = calculate_rolling_beta(
        cast("pd.Series", returns_df["potential_return"]), bench, window
    )
    beta_after_trend = calculate_rolling_beta(
        cast("pd.Series", returns_df["return_after_trend"]), bench, window
    )
    beta_after_deadband = calculate_rolling_beta(
        cast("pd.Series", returns_df["return_after_deadband"]), bench, window
    )
    realized_beta = calculate_rolling_beta(
        cast("pd.Series", returns_df["actual_return"]), bench, window
    )

    # Delta 계산
    lost_to_trend_filter = potential_beta - beta_after_trend
    lost_to_deadband = beta_after_trend - beta_after_deadband
    lost_to_vol_scaling = beta_after_deadband - realized_beta

    return pd.DataFrame(
        {
            "potential_beta": potential_beta,
            "beta_after_trend_filter": beta_after_trend,
            "beta_after_deadband": beta_after_deadband,
            "realized_beta": realized_beta,
            "lost_to_trend_filter": lost_to_trend_filter,
            "lost_to_deadband": lost_to_deadband,
            "lost_to_vol_scaling": lost_to_vol_scaling,
        },
        index=returns_df.index,
    )


def summarize_suppression_impact(
    diagnostics_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """시그널 억제 원인별 영향을 요약합니다.

    Args:
        diagnostics_df: 진단 레코드 DataFrame

    Returns:
        억제 원인별 통계:
            - count: 발생 횟수
            - percentage: 전체 대비 비율
            - avg_potential_weight: 억제되지 않았을 경우 평균 비중
    """
    if diagnostics_df.empty:
        return {}

    total = len(diagnostics_df)
    result: dict[str, dict[str, float]] = {}

    # 억제 원인별 그룹화
    grouped = diagnostics_df.groupby("signal_suppression_reason")

    for reason, group in grouped:
        count = len(group)
        percentage = count / total * 100

        # 억제되지 않았을 경우 예상 비중
        if reason == "none":
            avg_weight = float(group["final_target_weight"].abs().mean())
        else:
            # 억제된 경우: scaled_momentum을 사용 (억제 전 값)
            avg_weight = float(group["scaled_momentum"].abs().mean())

        result[str(reason)] = {
            "count": float(count),
            "percentage": percentage,
            "avg_potential_weight": avg_weight,
        }

    return result
