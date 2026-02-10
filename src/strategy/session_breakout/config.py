"""Session Breakout Strategy Configuration.

Asian session (00-08 UTC) range breakout을 포착하는 1H 전략입니다.

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


class SessionBreakoutConfig(BaseModel):
    """Session Breakout 전략 설정.

    Asian session (00:00-08:00 UTC)의 high/low range를 계산하고,
    trade window에서 breakout 방향을 추종합니다.

    Signal Formula:
        1. asian_high/low = 00-08 UTC 구간의 max(high)/min(low)
        2. range_pctl = asian_range의 rolling percentile
        3. squeeze = range_pctl < threshold (좁은 range = breakout 준비)
        4. Long: close > asian_high & squeeze & trade_window
        5. Short: close < asian_low & squeeze & trade_window
        6. Exit: exit_hour 도달 시 청산

    Attributes:
        asian_start_hour: Asian session 시작 시각 (UTC)
        asian_end_hour: Asian session 종료 시각 (UTC)
        trade_end_hour: 거래 허용 종료 시각 (UTC)
        exit_hour: 강제 청산 시각 (UTC)
        range_pctl_window: Range percentile 윈도우 (1H bars)
        range_pctl_threshold: Squeeze 판정 percentile 임계값
        tp_multiplier: Take-profit 배수 (asian range 대비)
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수 (1H: 8760)
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Session 파라미터
    # =========================================================================
    asian_start_hour: int = Field(
        default=0,
        ge=0,
        le=23,
        description="Asian session 시작 시각 (UTC, 0 = 자정)",
    )
    asian_end_hour: int = Field(
        default=8,
        ge=1,
        le=23,
        description="Asian session 종료 시각 (UTC, exclusive)",
    )
    trade_end_hour: int = Field(
        default=20,
        ge=1,
        le=23,
        description="거래 허용 종료 시각 (UTC)",
    )
    exit_hour: int = Field(
        default=22,
        ge=1,
        le=23,
        description="강제 청산 시각 (UTC)",
    )

    # =========================================================================
    # Range 파라미터
    # =========================================================================
    range_pctl_window: int = Field(
        default=720,
        ge=48,
        le=2160,
        description="Range percentile 윈도우 (1H bars, 720 = 30일)",
    )
    range_pctl_threshold: float = Field(
        default=50.0,
        ge=10.0,
        le=90.0,
        description="Squeeze 판정 percentile 임계값 (이 이하면 squeeze)",
    )
    tp_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Take-profit 배수 (asian range 대비)",
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
        default=8760.0,
        gt=0,
        description="연환산 계수 (1H: 8760 = 24*365)",
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
        if self.asian_end_hour <= self.asian_start_hour:
            msg = (
                f"asian_end_hour ({self.asian_end_hour}) must be > "
                f"asian_start_hour ({self.asian_start_hour})"
            )
            raise ValueError(msg)
        if self.exit_hour <= self.trade_end_hour:
            msg = f"exit_hour ({self.exit_hour}) must be > trade_end_hour ({self.trade_end_hour})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (1H bars)."""
        return self.range_pctl_window + 1
