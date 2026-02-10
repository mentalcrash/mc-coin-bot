"""Volume Climax Reversal Strategy Configuration.

극단적 거래량 스파이크(Climax)를 감지하여 단기 반전을 포착합니다.

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


class VolClimaxConfig(BaseModel):
    """Volume Climax Reversal 전략 설정.

    극단적 거래량 스파이크(climax)는 집단적 항복(capitulation) 또는
    과열(euphoria)을 의미 → 에너지 소진 → 단기 반전.

    Signal Formula:
        1. vol_zscore = (vol - rolling_mean) / rolling_std
        2. climax = vol_zscore > climax_threshold
        3. OBV divergence: obv_direction != price_direction
        4. Bullish reversal: climax + price_down + close_near_low (capitulation)
        5. Bearish reversal: climax + price_up + close_near_high (euphoria)
        6. Divergence boost: strength * divergence_boost
        7. Exit: vol_zscore < exit_vol_zscore OR timeout

    Attributes:
        vol_zscore_window: Volume Z-score rolling 윈도우
        climax_threshold: Climax 감지 Z-score 임계값
        obv_lookback: OBV 방향 lookback
        divergence_boost: Divergence 확신 승수
        close_position_threshold: Close near low/high 임계값
        exit_vol_zscore: 청산 volume Z-score 임계값
        exit_timeout_bars: 청산 timeout (캔들 수)
        mom_lookback: 모멘텀 lookback (vol scalar용)
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
    # Volume Climax 파라미터
    # =========================================================================
    vol_zscore_window: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Volume Z-score rolling 윈도우 (캔들 수)",
    )
    climax_threshold: float = Field(
        default=2.5,
        ge=1.5,
        le=5.0,
        description="Climax 감지 Z-score 임계값",
    )
    obv_lookback: int = Field(
        default=6,
        ge=3,
        le=20,
        description="OBV 방향 lookback (캔들 수)",
    )
    divergence_boost: float = Field(
        default=1.3,
        ge=1.0,
        le=2.0,
        description="Divergence 확신 승수",
    )
    close_position_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Close near low/high 판정 임계값 (0=low, 1=high)",
    )
    exit_vol_zscore: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="청산 volume Z-score 임계값",
    )
    exit_timeout_bars: int = Field(
        default=18,
        ge=6,
        le=48,
        description="청산 timeout (캔들 수, 18 = 3일 @4H)",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="모멘텀 lookback (vol scalar 계산용)",
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

        if self.exit_vol_zscore >= self.climax_threshold:
            msg = (
                f"exit_vol_zscore ({self.exit_vol_zscore}) must be "
                f"< climax_threshold ({self.climax_threshold})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return (
            max(self.vol_zscore_window, self.obv_lookback, self.mom_lookback, self.atr_period) + 1
        )
