"""VW-TSMOM Signal Pipeline Diagnostics.

이 모듈은 TSMOM 전략의 시그널 파이프라인 상태를 진단하고 로깅합니다.
매 캔들마다의 시그널 생성 과정을 추적하여 Beta 손실 원인을 분석합니다.

Rules Applied:
    - #11 Pydantic Modeling: SignalDiagnosticRecord 사용
    - #15 Logging Standards: get_trading_logger(), 구조화된 로깅
    - #12 Data Engineering: Vectorized DataFrame 처리
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.logging.context import get_trading_logger
from src.models.backtest import SignalDiagnosticRecord, SignalSuppressionReason

if TYPE_CHECKING:
    from datetime import datetime

    from loguru import Logger


# =============================================================================
# Logging Utilities
# =============================================================================


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


def log_signal_diagnostic(record: SignalDiagnosticRecord) -> None:
    """진단 레코드를 구조화된 로그로 출력합니다.

    Args:
        record: 시그널 진단 레코드

    Example:
        >>> log_signal_diagnostic(record)
        # DEBUG | Signal diagnostic | momentum=0.15 | vol_scalar=0.62 | ...
    """
    log = get_diagnostic_logger(record.symbol)

    log.debug(
        (
            "Signal diagnostic | momentum={raw_mom:.4f} | vol_scalar={vol_scl:.2f} | "
            "trend_regime={trend} | final_weight={weight:.4f} | suppressed_by={reason}"
        ),
        raw_mom=record.raw_momentum,
        vol_scl=record.vol_scalar,
        trend=record.trend_regime,
        weight=record.final_target_weight,
        reason=record.signal_suppression_reason,
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

    # 억제 원인별 카운트
    suppression_counts = diagnostics_df["signal_suppression_reason"].value_counts()

    log.info(
        (
            "Diagnostic Summary | Total: {total} candles | "
            "Trend Filter: {trend} | Deadband: {deadband} | "
            "Vol Scaling: {vol} | Normal: {normal}"
        ),
        total=total,
        trend=int(suppression_counts.get("trend_filter") or 0),
        deadband=int(suppression_counts.get("deadband") or 0),
        vol=int(suppression_counts.get("low_volatility") or 0),
        normal=int(suppression_counts.get("none") or 0),
    )


# =============================================================================
# Diagnostic Collector Class
# =============================================================================


class DiagnosticCollector:
    """시그널 파이프라인 진단 데이터 수집기.

    매 캔들마다의 시그널 생성 과정을 추적하고 DataFrame으로 반환합니다.
    백테스트에서 Beta Attribution 분석에 사용됩니다.

    Attributes:
        symbol: 거래 심볼
        records: 수집된 진단 레코드 목록

    Example:
        >>> collector = DiagnosticCollector("BTC/USDT")
        >>> for idx, row in processed_df.iterrows():
        ...     record = collector.collect_from_row(row, signal_data)
        >>> diagnostics_df = collector.to_dataframe()
    """

    def __init__(self, symbol: str) -> None:
        """DiagnosticCollector를 초기화합니다.

        Args:
            symbol: 거래 심볼
        """
        self.symbol = symbol
        self._records: list[dict[str, object]] = []

    def collect(
        self,
        *,
        timestamp: datetime,
        close_price: Decimal | float,
        realized_vol_annualized: float,
        benchmark_return: float,
        raw_momentum: float,
        vol_scalar: float,
        scaled_momentum: float,
        trend_regime: int,
        signal_before_trend_filter: float,
        signal_after_trend_filter: float,
        deadband_applied: bool,
        signal_after_deadband: float,
        raw_target_weight: float,
        leverage_capped_weight: float,
        final_target_weight: float,
        rebalance_triggered: bool,
        stop_loss_triggered: bool = False,
    ) -> SignalDiagnosticRecord:
        """단일 캔들의 진단 데이터를 수집합니다.

        Args:
            timestamp: 캔들 타임스탬프
            close_price: 종가
            realized_vol_annualized: 실현 변동성 (연율화)
            benchmark_return: 벤치마크 수익률
            raw_momentum: 원시 모멘텀
            vol_scalar: 변동성 스케일러
            scaled_momentum: 스케일링된 모멘텀
            trend_regime: 추세 국면 (1, -1, 0)
            signal_before_trend_filter: Trend filter 적용 전 시그널
            signal_after_trend_filter: Trend filter 적용 후 시그널
            deadband_applied: Deadband 적용 여부
            signal_after_deadband: Deadband 적용 후 시그널
            raw_target_weight: 원시 목표 비중
            leverage_capped_weight: 레버리지 제한 후 비중
            final_target_weight: 최종 목표 비중
            rebalance_triggered: 리밸런싱 발동 여부
            stop_loss_triggered: 스탑 로스 발동 여부

        Returns:
            생성된 SignalDiagnosticRecord
        """
        # 시그널 억제 원인 결정
        suppression_reason = self._determine_suppression_reason(
            signal_before_trend_filter=signal_before_trend_filter,
            signal_after_trend_filter=signal_after_trend_filter,
            deadband_applied=deadband_applied,
            signal_after_deadband=signal_after_deadband,
            raw_target_weight=raw_target_weight,
            leverage_capped_weight=leverage_capped_weight,
            final_target_weight=final_target_weight,
            rebalance_triggered=rebalance_triggered,
            stop_loss_triggered=stop_loss_triggered,
        )

        # Decimal 변환
        if not isinstance(close_price, Decimal):
            close_price = Decimal(str(close_price))

        record = SignalDiagnosticRecord(
            timestamp=timestamp,
            symbol=self.symbol,
            close_price=close_price,
            realized_vol_annualized=realized_vol_annualized,
            benchmark_return=benchmark_return,
            raw_momentum=raw_momentum,
            vol_scalar=vol_scalar,
            scaled_momentum=scaled_momentum,
            trend_regime=trend_regime,  # type: ignore[arg-type]
            signal_before_trend_filter=signal_before_trend_filter,
            signal_after_trend_filter=signal_after_trend_filter,
            deadband_applied=deadband_applied,
            signal_after_deadband=signal_after_deadband,
            raw_target_weight=raw_target_weight,
            leverage_capped_weight=leverage_capped_weight,
            final_target_weight=final_target_weight,
            rebalance_triggered=rebalance_triggered,
            stop_loss_triggered=stop_loss_triggered,
            signal_suppression_reason=suppression_reason,
        )

        # 내부 저장 (DataFrame 생성용)
        self._records.append(record.model_dump())

        return record

    def _determine_suppression_reason(
        self,
        *,
        signal_before_trend_filter: float,
        signal_after_trend_filter: float,
        deadband_applied: bool,
        signal_after_deadband: float,
        raw_target_weight: float,
        leverage_capped_weight: float,
        final_target_weight: float,
        rebalance_triggered: bool,
        stop_loss_triggered: bool,
    ) -> SignalSuppressionReason:
        """시그널 억제 원인을 결정합니다.

        우선순위:
            1. stop_loss - 스탑 로스 발동
            2. trend_filter - Trend filter로 시그널이 0이 됨
            3. deadband - Deadband로 시그널이 억제됨
            4. leverage_cap - 레버리지 제한으로 비중 축소
            5. rebalance_threshold - 리밸런싱 미달
            6. none - 정상 시그널

        Returns:
            억제 원인 Literal 값
        """
        if stop_loss_triggered:
            return "stop_loss"

        # Trend filter로 시그널이 0이 됨
        if signal_before_trend_filter != 0 and signal_after_trend_filter == 0:
            return "trend_filter"

        # Deadband로 시그널이 억제됨
        if deadband_applied and signal_after_deadband == 0:
            return "deadband"

        # 레버리지 제한으로 비중 축소
        if raw_target_weight != 0 and leverage_capped_weight < raw_target_weight:
            return "leverage_cap"

        # 리밸런싱 threshold 미달
        if not rebalance_triggered and final_target_weight != 0:
            return "rebalance_threshold"

        return "none"

    def to_dataframe(self) -> pd.DataFrame:
        """수집된 진단 데이터를 DataFrame으로 반환합니다.

        Returns:
            진단 레코드 DataFrame (timestamp를 인덱스로 사용)
        """
        if not self._records:
            return pd.DataFrame()

        df = pd.DataFrame(self._records)
        df = df.set_index("timestamp")
        return df

    def clear(self) -> None:
        """수집된 레코드를 초기화합니다."""
        self._records.clear()

    def __len__(self) -> int:
        """수집된 레코드 수를 반환합니다."""
        return len(self._records)


# =============================================================================
# Vectorized Diagnostic Collection (for Backtesting)
# =============================================================================


def collect_diagnostics_from_pipeline(
    *,
    processed_df: pd.DataFrame,
    symbol: str,
    signal_before_trend: pd.Series,
    signal_after_trend: pd.Series,
    signal_after_deadband: pd.Series,
    deadband_mask: pd.Series,
    final_weights: pd.Series,
    leverage_capped_weights: pd.Series | None = None,
    rebalance_mask: pd.Series | None = None,
    stop_loss_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """시그널 파이프라인에서 진단 DataFrame을 벡터화하여 생성합니다.

    백테스트에서 각 필터 단계의 중간 값을 전달받아
    전체 진단 DataFrame을 한 번에 생성합니다.

    Args:
        processed_df: 전처리된 OHLCV DataFrame
        symbol: 거래 심볼
        signal_before_trend: Trend filter 적용 전 시그널 Series
        signal_after_trend: Trend filter 적용 후 시그널 Series
        signal_after_deadband: Deadband 적용 후 시그널 Series
        deadband_mask: Deadband 적용 여부 bool Series
        final_weights: 최종 목표 비중 Series
        leverage_capped_weights: 레버리지 제한 후 비중 (None이면 final_weights 사용)
        rebalance_mask: 리밸런싱 발동 여부 bool Series (None이면 True로 가정)
        stop_loss_mask: 스탑 로스 발동 여부 bool Series (None이면 False로 가정)

    Returns:
        진단 레코드 DataFrame (DatetimeIndex)

    Example:
        >>> diagnostics_df = collect_diagnostics_from_pipeline(
        ...     processed_df=df,
        ...     symbol="BTC/USDT",
        ...     signal_before_trend=signal_shifted,
        ...     signal_after_trend=signal_filtered,
        ...     signal_after_deadband=signal_final,
        ...     deadband_mask=deadband_applied,
        ...     final_weights=strength,
        ... )
    """
    # 기본값 설정
    if leverage_capped_weights is None:
        leverage_capped_weights = final_weights
    if rebalance_mask is None:
        rebalance_mask = pd.Series(True, index=processed_df.index)
    if stop_loss_mask is None:
        stop_loss_mask = pd.Series(False, index=processed_df.index)

    # 벤치마크 수익률 계산 (close 기준 수익률)
    close_series: pd.Series = processed_df["close"]  # type: ignore[assignment]
    benchmark_returns = close_series.pct_change().fillna(0)

    # Trend regime 추출 (없으면 0으로 설정)
    if "trend_regime" in processed_df.columns:
        trend_regime = processed_df["trend_regime"].fillna(0).astype(int)
    else:
        trend_regime = pd.Series(0, index=processed_df.index)

    # 억제 원인 결정 (벡터화)
    suppression_reasons = _determine_suppression_reasons_vectorized(
        signal_before_trend=signal_before_trend,
        signal_after_trend=signal_after_trend,
        deadband_mask=deadband_mask,
        signal_after_deadband=signal_after_deadband,
        raw_weights=final_weights,  # NOTE: raw_target_weight는 별도 계산 필요
        leverage_capped_weights=leverage_capped_weights,
        rebalance_mask=rebalance_mask,
        stop_loss_mask=stop_loss_mask,
    )

    # DataFrame 생성
    diagnostics_df = pd.DataFrame(
        {
            "symbol": symbol,
            "close_price": processed_df["close"],
            "realized_vol_annualized": processed_df.get("realized_vol", 0.0),
            "benchmark_return": benchmark_returns,
            "raw_momentum": processed_df.get("vw_momentum", 0.0),
            "vol_scalar": processed_df.get("vol_scalar", 1.0),
            "scaled_momentum": signal_before_trend,
            "trend_regime": trend_regime,
            "signal_before_trend_filter": signal_before_trend,
            "signal_after_trend_filter": signal_after_trend,
            "deadband_applied": deadband_mask,
            "signal_after_deadband": signal_after_deadband,
            "raw_target_weight": final_weights,  # NOTE: 실제로는 스케일링 전 값
            "leverage_capped_weight": leverage_capped_weights,
            "final_target_weight": final_weights,
            "rebalance_triggered": rebalance_mask,
            "stop_loss_triggered": stop_loss_mask,
            "signal_suppression_reason": suppression_reasons,
        },
        index=processed_df.index,
    )

    return diagnostics_df


def _determine_suppression_reasons_vectorized(
    *,
    signal_before_trend: pd.Series,
    signal_after_trend: pd.Series,
    deadband_mask: pd.Series,
    signal_after_deadband: pd.Series,
    raw_weights: pd.Series,
    leverage_capped_weights: pd.Series,
    rebalance_mask: pd.Series,
    stop_loss_mask: pd.Series,
) -> pd.Series:
    """벡터화된 억제 원인 결정.

    Returns:
        억제 원인 문자열 Series
    """
    reasons = pd.Series("none", index=signal_before_trend.index)

    # 우선순위 역순으로 적용 (낮은 우선순위부터)

    # 5. rebalance_threshold
    mask = (~rebalance_mask) & (signal_after_deadband != 0)
    reasons = reasons.where(~mask, "rebalance_threshold")

    # 4. leverage_cap
    mask = (raw_weights != 0) & (leverage_capped_weights < np.abs(raw_weights))
    reasons = reasons.where(~mask, "leverage_cap")

    # 3. deadband
    mask = deadband_mask & (signal_after_deadband == 0)
    reasons = reasons.where(~mask, "deadband")

    # 2. trend_filter
    mask = (signal_before_trend != 0) & (signal_after_trend == 0)
    reasons = reasons.where(~mask, "trend_filter")

    # 1. stop_loss (최우선)
    reasons = reasons.where(~stop_loss_mask, "stop_loss")

    return reasons
