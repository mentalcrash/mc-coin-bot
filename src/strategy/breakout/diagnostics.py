"""Adaptive Breakout Signal Pipeline Diagnostics.

이 모듈은 Adaptive Breakout 전략의 시그널 파이프라인 상태를 진단하고 로깅합니다.
매 캔들마다의 시그널 생성 과정을 추적하여 전략 성과 원인을 분석합니다.

Rules Applied:
    - #11 Pydantic Modeling: SignalDiagnosticRecord 사용
    - #15 Logging Standards: get_trading_logger(), 구조화된 로깅
    - #12 Data Engineering: Vectorized DataFrame 처리
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.logging.context import get_trading_logger

if TYPE_CHECKING:
    from datetime import datetime

    from loguru import Logger

# =============================================================================
# Type Definitions
# =============================================================================

type BreakoutSuppressionReason = Literal[
    "none",  # 정상 시그널
    "volatility_filter",  # 변동성 필터로 억제
    "false_breakout",  # 가짜 돌파 (즉시 복귀)
    "cooldown",  # 쿨다운 기간 중
    "leverage_cap",  # 레버리지 제한
    "rebalance_threshold",  # 리밸런싱 미달
    "long_only",  # Long-Only 모드로 Short 억제
]


# =============================================================================
# Pydantic Models
# =============================================================================


class BreakoutDiagnosticRecord(BaseModel):
    """Breakout 전략 진단 레코드.

    단일 캔들에 대한 시그널 파이프라인 진단 정보를 저장합니다.
    """

    model_config = ConfigDict(frozen=True)

    # 기본 정보
    timestamp: datetime
    symbol: str
    close_price: Decimal

    # Donchian Channel 정보
    upper_band: float
    lower_band: float
    middle_band: float

    # ATR 정보
    atr_value: float
    threshold: float

    # 변동성 정보
    realized_vol: float
    volatility_ratio: float
    vol_scalar: float

    # 돌파 정보
    breakout_type: str  # "upper", "lower", "none"
    distance_to_band: float  # 돌파 거리 (%)

    # 시그널 정보
    raw_signal: int  # -1, 0, 1
    final_signal: int  # -1, 0, 1
    raw_strength: float
    final_strength: float

    # 억제 정보
    signal_suppression_reason: BreakoutSuppressionReason


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
        strategy="AdaptiveBreakout",
        symbol=symbol,
        diagnostic_type="signal_pipeline",
    )


def log_signal_diagnostic(record: BreakoutDiagnosticRecord) -> None:
    """진단 레코드를 구조화된 로그로 출력합니다.

    Args:
        record: 시그널 진단 레코드
    """
    log = get_diagnostic_logger(record.symbol)

    log.debug(
        (
            "Signal diagnostic | breakout={breakout} | atr={atr:.4f} | "
            "vol_ratio={vol_ratio:.2f} | strength={strength:.4f} | suppressed_by={reason}"
        ),
        breakout=record.breakout_type,
        atr=record.atr_value,
        vol_ratio=record.volatility_ratio,
        strength=record.final_strength,
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
            "Volatility Filter: {vol_filter} | Cooldown: {cooldown} | "
            "Long-Only: {long_only} | Normal: {normal}"
        ),
        total=total,
        vol_filter=int(suppression_counts.get("volatility_filter") or 0),
        cooldown=int(suppression_counts.get("cooldown") or 0),
        long_only=int(suppression_counts.get("long_only") or 0),
        normal=int(suppression_counts.get("none") or 0),
    )


# =============================================================================
# Diagnostic Collector Class
# =============================================================================


class BreakoutDiagnosticCollector:
    """Breakout 전략 진단 데이터 수집기.

    매 캔들마다의 시그널 생성 과정을 추적하고 DataFrame으로 반환합니다.
    백테스트에서 전략 성과 분석에 사용됩니다.

    Attributes:
        symbol: 거래 심볼
        records: 수집된 진단 레코드 목록

    Example:
        >>> collector = BreakoutDiagnosticCollector("BTC/USDT")
        >>> collector.collect(
        ...     timestamp=dt,
        ...     close_price=Decimal("50000"),
        ...     ...
        ... )
        >>> diagnostics_df = collector.to_dataframe()
    """

    def __init__(self, symbol: str) -> None:
        """BreakoutDiagnosticCollector를 초기화합니다.

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
        upper_band: float,
        lower_band: float,
        middle_band: float,
        atr_value: float,
        threshold: float,
        realized_vol: float,
        volatility_ratio: float,
        vol_scalar: float,
        breakout_type: str,
        distance_to_band: float,
        raw_signal: int,
        final_signal: int,
        raw_strength: float,
        final_strength: float,
        suppression_reason: BreakoutSuppressionReason = "none",
    ) -> BreakoutDiagnosticRecord:
        """단일 캔들의 진단 데이터를 수집합니다.

        Args:
            timestamp: 캔들 타임스탬프
            close_price: 종가
            upper_band: Donchian Channel 상단
            lower_band: Donchian Channel 하단
            middle_band: Donchian Channel 중심선
            atr_value: ATR 값
            threshold: 돌파 임계값
            realized_vol: 실현 변동성
            volatility_ratio: 변동성 비율
            vol_scalar: 변동성 스케일러
            breakout_type: 돌파 유형 ("upper", "lower", "none")
            distance_to_band: 밴드까지 거리 (%)
            raw_signal: 원시 시그널 (-1, 0, 1)
            final_signal: 최종 시그널 (-1, 0, 1)
            raw_strength: 원시 시그널 강도
            final_strength: 최종 시그널 강도
            suppression_reason: 시그널 억제 원인

        Returns:
            생성된 BreakoutDiagnosticRecord
        """
        # Decimal 변환
        if not isinstance(close_price, Decimal):
            close_price = Decimal(str(close_price))

        record = BreakoutDiagnosticRecord(
            timestamp=timestamp,
            symbol=self.symbol,
            close_price=close_price,
            upper_band=upper_band,
            lower_band=lower_band,
            middle_band=middle_band,
            atr_value=atr_value,
            threshold=threshold,
            realized_vol=realized_vol,
            volatility_ratio=volatility_ratio,
            vol_scalar=vol_scalar,
            breakout_type=breakout_type,
            distance_to_band=distance_to_band,
            raw_signal=raw_signal,
            final_signal=final_signal,
            raw_strength=raw_strength,
            final_strength=final_strength,
            signal_suppression_reason=suppression_reason,
        )

        # 내부 저장 (DataFrame 생성용)
        self._records.append(record.model_dump())

        return record

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
    signals_direction: pd.Series,
    signals_strength: pd.Series,
    config_long_only: bool = False,
    cooldown_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """시그널 파이프라인에서 진단 DataFrame을 벡터화하여 생성합니다.

    백테스트에서 각 필터 단계의 중간 값을 전달받아
    전체 진단 DataFrame을 한 번에 생성합니다.

    Args:
        processed_df: 전처리된 OHLCV DataFrame
        symbol: 거래 심볼
        signals_direction: 최종 방향 Series
        signals_strength: 최종 시그널 강도 Series
        config_long_only: Long-Only 모드 여부
        cooldown_mask: 쿨다운 중 여부 bool Series

    Returns:
        진단 레코드 DataFrame (DatetimeIndex)
    """
    # 기본값 설정
    if cooldown_mask is None:
        cooldown_mask = pd.Series(False, index=processed_df.index)

    # 필요한 컬럼 추출
    close_series: pd.Series = processed_df["close"]  # type: ignore[assignment]
    upper_band: pd.Series = processed_df["upper_band"]  # type: ignore[assignment]
    lower_band: pd.Series = processed_df["lower_band"]  # type: ignore[assignment]

    # 돌파 유형 결정 (벡터화)
    breakout_type = pd.Series("none", index=processed_df.index)
    breakout_type = breakout_type.where(signals_direction != 1, "upper")
    breakout_type = breakout_type.where(signals_direction != -1, "lower")

    # 밴드까지 거리 계산
    distance_to_upper: pd.Series = (upper_band - close_series) / close_series * 100
    distance_to_lower: pd.Series = (close_series - lower_band) / close_series * 100

    # 돌파 방향에 따른 거리 선택
    distance_to_band = pd.Series(0.0, index=processed_df.index)
    distance_to_band = distance_to_band.where(signals_direction != 1, distance_to_upper)
    distance_to_band = distance_to_band.where(signals_direction != -1, distance_to_lower)

    # 억제 원인 결정 (벡터화)
    suppression_reasons = _determine_suppression_reasons_vectorized(
        signals_direction=signals_direction,
        config_long_only=config_long_only,
        cooldown_mask=cooldown_mask,
        vol_scalar=processed_df.get("vol_scalar", pd.Series(1.0, index=processed_df.index)),  # type: ignore[arg-type]
    )

    # DataFrame 생성
    diagnostics_df = pd.DataFrame(
        {
            "symbol": symbol,
            "close_price": close_series,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "middle_band": processed_df.get("middle_band", np.nan),
            "atr_value": processed_df.get("atr", np.nan),
            "threshold": processed_df.get("threshold", np.nan),
            "realized_vol": processed_df.get("realized_vol", np.nan),
            "volatility_ratio": processed_df.get("volatility_ratio", 1.0),
            "vol_scalar": processed_df.get("vol_scalar", 1.0),
            "breakout_type": breakout_type,
            "distance_to_band": distance_to_band,
            "raw_signal": signals_direction,  # 단순화: raw와 final 동일
            "final_signal": signals_direction,
            "raw_strength": signals_strength,
            "final_strength": signals_strength,
            "signal_suppression_reason": suppression_reasons,
        },
        index=processed_df.index,
    )

    return diagnostics_df


def _determine_suppression_reasons_vectorized(
    *,
    signals_direction: pd.Series,
    config_long_only: bool,
    cooldown_mask: pd.Series,
    vol_scalar: pd.Series,
    min_vol_scalar: float = 0.1,
) -> pd.Series:
    """벡터화된 억제 원인 결정.

    Args:
        signals_direction: 최종 방향 Series
        config_long_only: Long-Only 모드 여부
        cooldown_mask: 쿨다운 중 여부
        vol_scalar: 변동성 스케일러
        min_vol_scalar: 최소 변동성 스케일러

    Returns:
        억제 원인 문자열 Series
    """
    reasons = pd.Series("none", index=signals_direction.index)

    # Long-Only 모드에서 Short 억제
    if config_long_only:
        # Short 시그널이 있었을 가능성 (현재는 추적 불가, 단순화)
        pass

    # 쿨다운 중
    mask = cooldown_mask & (signals_direction == 0)
    reasons = reasons.where(~mask, "cooldown")

    # 변동성 필터
    mask = (vol_scalar < min_vol_scalar) & (signals_direction == 0)
    reasons = reasons.where(~mask, "volatility_filter")

    return reasons
