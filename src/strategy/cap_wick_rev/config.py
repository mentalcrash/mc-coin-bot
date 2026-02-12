"""Capitulation Wick Reversal 전략 설정.

레버리지 청산 캐스케이드 후 가격 과잉반응 -> 48-72h 회복 패턴 포착.
3중 필터(ATR spike + Volume surge + Wick ratio) + confirmation bars.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CapWickRevConfig(BaseModel):
    """Capitulation Wick Reversal 전략 설정.

    레버리지 청산 캐스케이드로 인한 가격 과잉반응을 3중 필터로 감지하고
    48-72h(12-18 bars @4H) 회복 패턴을 포착합니다.

    Signal Formula:
        1. ATR spike: current_atr / rolling_median_atr > atr_spike_threshold
        2. Volume surge: current_vol / rolling_median_vol > vol_surge_threshold
        3. Wick ratio (bearish): lower_wick / bar_range > wick_ratio_threshold
           Wick ratio (bullish): upper_wick / bar_range > wick_ratio_threshold
        4. Capitulation (long): ATR spike + Vol surge + large lower wick + close near low
        5. Euphoria (short): ATR spike + Vol surge + large upper wick + close near high
        6. Confirmation: wait N bars after signal for price to stabilize
        7. Exit: timeout OR ATR normalizes

    Attributes:
        atr_window: ATR 계산 기간 (캔들 수)
        atr_spike_threshold: ATR spike 감지 비율 (current / median)
        vol_surge_window: Volume surge rolling window (캔들 수)
        vol_surge_threshold: Volume surge 감지 비율 (current / median)
        wick_ratio_threshold: Wick ratio 임계값 (wick / range)
        close_position_threshold: Close near low/high 판정 임계값
        confirmation_bars: 확인 대기 바 수
        exit_timeout_bars: 타임아웃 청산 바 수
        mom_lookback: 변동성 계산 lookback (캔들 수)
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수 (4H: 2190)
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 기간 (trailing stop용)
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Capitulation Wick 파라미터
    # =========================================================================
    atr_window: int = Field(
        default=30,
        ge=10,
        le=100,
        description="ATR spike 기준선 rolling median window (캔들 수)",
    )
    atr_spike_threshold: float = Field(
        default=2.0,
        ge=1.2,
        le=5.0,
        description="ATR spike 감지 비율 (current_atr / median_atr)",
    )
    vol_surge_window: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Volume surge rolling median window (캔들 수)",
    )
    vol_surge_threshold: float = Field(
        default=2.0,
        ge=1.2,
        le=5.0,
        description="Volume surge 감지 비율 (current_vol / median_vol)",
    )
    wick_ratio_threshold: float = Field(
        default=0.5,
        ge=0.2,
        le=0.8,
        description="Wick ratio 임계값 (wick_length / bar_range)",
    )
    close_position_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Close near low/high 판정 임계값 (0=low, 1=high)",
    )
    confirmation_bars: int = Field(
        default=2,
        ge=0,
        le=6,
        description="확인 대기 바 수 (0이면 즉시 진입)",
    )
    exit_timeout_bars: int = Field(
        default=18,
        ge=6,
        le=48,
        description="타임아웃 청산 바 수 (18 = 3일 @4H)",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 계산 lookback (캔들 수)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프 (0으로 나누기 방지)",
    )
    annualization_factor: float = Field(
        default=2190.0,
        gt=0,
        description="연환산 계수 (4H: 6*365=2190)",
    )

    # =========================================================================
    # 옵션
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (Trailing Stop용)",
    )

    # =========================================================================
    # 숏 모드 설정
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
    )
    hedge_threshold: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.05,
        description="헤지 숏 활성화 드로다운 임계값 (예: -0.07 = -7%)",
    )
    hedge_strength_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="헤지 숏 강도 비율 (롱 대비, 예: 0.8 = 80%)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증."""
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return (
            max(
                self.atr_window,
                self.vol_surge_window,
                self.mom_lookback,
                self.atr_period,
            )
            + self.confirmation_bars
            + 10
        )
