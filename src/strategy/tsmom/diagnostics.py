"""VW-TSMOM Signal Pipeline Diagnostics (Pure TSMOM).

이 모듈은 TSMOM 전략의 시그널 파이프라인 상태를 진단하고 로깅합니다.
Pure TSMOM + Vol Target 구현에 맞게 단순화되었습니다.

Rules Applied:
    - #15 Logging Standards: get_trading_logger(), 구조화된 로깅
    - #12 Data Engineering: Vectorized DataFrame 처리
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.logging.context import get_trading_logger

if TYPE_CHECKING:
    from loguru import Logger


def get_diagnostic_logger(symbol: str) -> Logger:
    """진단용 로거를 반환합니다.

    Args:
        symbol: 거래 심볼 (예: "BTC/USDT")

    Returns:
        진단 컨텍스트가 바인딩된 Logger
    """
    return get_trading_logger(
        strategy="TSMOM",
        symbol=symbol,
        diagnostic_type="signal_pipeline",
    )


def log_diagnostic_summary(diagnostics_df: pd.DataFrame, symbol: str) -> None:
    """진단 DataFrame의 요약 통계를 로깅합니다.

    Args:
        diagnostics_df: 진단 레코드 DataFrame
        symbol: 거래 심볼
    """
    log = get_diagnostic_logger(symbol)

    total = len(diagnostics_df)
    if total == 0:
        log.warning("No diagnostic records to summarize")
        return

    # 시그널 방향별 카운트
    if "final_target_weight" in diagnostics_df.columns:
        weights = diagnostics_df["final_target_weight"]
        long_count = int((weights > 0).sum())
        short_count = int((weights < 0).sum())
        neutral_count = int((weights == 0).sum())

        log.info(
            "Diagnostic Summary | Total: {total} candles | "
            + "Long: {long} | Short: {short} | Neutral: {neutral}",
            total=total,
            long=long_count,
            short=short_count,
            neutral=neutral_count,
        )


def collect_diagnostics_from_signals(
    processed_df: pd.DataFrame,
    symbol: str,
    final_weights: pd.Series,
) -> pd.DataFrame:
    """시그널에서 진단 DataFrame을 생성합니다.

    Pure TSMOM에서는 필터가 없으므로 단순한 진단 데이터만 수집합니다.

    Args:
        processed_df: 전처리된 OHLCV DataFrame
        symbol: 거래 심볼
        final_weights: 최종 시그널 강도 Series

    Returns:
        진단 레코드 DataFrame (DatetimeIndex)

    Example:
        >>> diagnostics_df = collect_diagnostics_from_signals(
        ...     processed_df=df,
        ...     symbol="BTC/USDT",
        ...     final_weights=signals.strength,
        ... )
    """
    # 벤치마크 수익률 계산 (close 기준 수익률)
    close_series: pd.Series = processed_df["close"]  # type: ignore[assignment]
    benchmark_returns = close_series.pct_change().fillna(0)

    # DataFrame 생성
    diagnostics_df = pd.DataFrame(
        {
            "symbol": symbol,
            "close_price": processed_df["close"],
            "realized_vol_annualized": processed_df.get("realized_vol", 0.0),
            "benchmark_return": benchmark_returns,
            "raw_momentum": processed_df.get("vw_momentum", 0.0),
            "vol_scalar": processed_df.get("vol_scalar", 1.0),
            "scaled_momentum": final_weights,
            "final_target_weight": final_weights,
        },
        index=processed_df.index,
    )

    return diagnostics_df
