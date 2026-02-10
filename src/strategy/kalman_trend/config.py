"""Adaptive Kalman Trend Strategy Configuration.

칼만 필터로 가격에서 노이즈를 분리합니다.

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


class KalmanTrendConfig(BaseModel):
    """Adaptive Kalman Trend 전략 설정.

    칼만 필터로 가격의 state(smoothed price)와 velocity를 추정합니다.
    Adaptive Q parameter로 변동성 레짐에 자동 적응합니다.

    Signal Formula:
        1. Kalman filter: state = [price, velocity], adaptive Q
        2. velocity > vel_threshold → LONG
        3. velocity < -vel_threshold → SHORT (HEDGE_ONLY)
        4. |velocity| < vel_threshold → FLAT
        5. strength = direction * vol_scalar

    Kalman Filter Model:
        State: [price, velocity]
        Transition: F = [[1, dt], [0, 1]]
        Observation: H = [[1, 0]]
        Q = base_q * (realized_vol / long_term_vol) * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
        R = observation_noise

    Attributes:
        base_q: 기본 프로세스 노이즈 Q
        observation_noise: 관측 노이즈 R
        vel_threshold: 속도 시그널 임계값
        vol_lookback: Realized Vol 계산 윈도우
        long_term_vol_lookback: 장기 Vol 계산 윈도우
        mom_lookback: 모멘텀 방향 lookback
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
    # Kalman Filter 파라미터
    # =========================================================================
    base_q: float = Field(
        default=0.01,
        ge=0.001,
        le=1.0,
        description="기본 프로세스 노이즈 Q",
    )
    observation_noise: float = Field(
        default=1.0,
        ge=0.01,
        le=100.0,
        description="관측 노이즈 R",
    )
    vel_threshold: float = Field(
        default=0.5,
        ge=0.01,
        le=5.0,
        description="속도 시그널 임계값",
    )

    # =========================================================================
    # Adaptive Q 파라미터
    # =========================================================================
    vol_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Realized Vol 계산 윈도우 (캔들 수)",
    )
    long_term_vol_lookback: int = Field(
        default=120,
        ge=40,
        le=500,
        description="장기 Vol 계산 윈도우 (캔들 수)",
    )

    # =========================================================================
    # 모멘텀 확인 파라미터
    # =========================================================================
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="모멘텀 방향 lookback (캔들 수)",
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
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
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
        if self.vol_lookback >= self.long_term_vol_lookback:
            msg = (
                f"vol_lookback ({self.vol_lookback}) must be "
                f"< long_term_vol_lookback ({self.long_term_vol_lookback})"
            )
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.long_term_vol_lookback, self.mom_lookback, self.atr_period) + 1
