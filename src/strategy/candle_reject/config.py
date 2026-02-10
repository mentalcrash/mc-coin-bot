"""Candlestick Rejection Momentum Strategy Configuration.

4H candle의 rejection wick으로 방향성 모멘텀 시그널을 생성합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 -> 중립)
        HEDGE_ONLY: 헤지 목적 숏만 (드로다운 임계값 초과 시)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CandleRejectConfig(BaseModel):
    """Candlestick Rejection Momentum 전략 설정.

    4H candle의 긴 꼬리(rejection wick)로 가격 거부를 감지합니다.
    거부 방향의 반대 = 시장의 진정한 방향 -> directional signal.

    Signal Formula:
        1. Bar anatomy: upper_wick, lower_wick, body, range
        2. Rejection ratio: bull_reject = lower_wick / range (long lower wick -> buy)
        3. Entry: bull_reject > threshold AND volume_zscore > vol_threshold
        4. Consecutive boost: 2+ consecutive rejections -> weight * 1.5
        5. Vol-target sizing: weight * vol_target / realized_vol
        6. Exit: body_position reversal OR timeout

    Attributes:
        rejection_threshold: Rejection ratio 임계값 (wick / range)
        volume_zscore_threshold: Volume Z-score 임계값
        volume_zscore_window: Volume Z-score rolling window
        consecutive_boost: 연속 rejection 시 conviction 배수
        consecutive_min: 부스트 활성화 최소 연속 횟수
        exit_timeout_bars: 타임아웃 청산 (bar 수)
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Candlestick Rejection 파라미터
    # =========================================================================
    rejection_threshold: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Rejection ratio 임계값 (wick_length / range)",
    )
    volume_zscore_threshold: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Volume Z-score 임계값",
    )
    volume_zscore_window: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Volume Z-score rolling window (캔들 수)",
    )
    consecutive_boost: float = Field(
        default=1.5,
        ge=1.0,
        le=2.0,
        description="연속 rejection 시 conviction 배수",
    )
    consecutive_min: int = Field(
        default=2,
        ge=2,
        le=5,
        description="부스트 활성화 최소 연속 rejection 횟수",
    )
    exit_timeout_bars: int = Field(
        default=12,
        ge=4,
        le=48,
        description="타임아웃 청산 (bar 수, 12 = 4H * 12 = 2일)",
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
        default=ShortMode.HEDGE_ONLY,
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
        """필요한 워밍업 기간 (캔들 수).

        volume_zscore_window가 가장 긴 rolling window.
        """
        return max(self.volume_zscore_window, self.atr_period) + 1
