"""Range Compression Breakout Strategy Configuration.

NR7 패턴과 range ratio를 사용하여 vol compression 후 breakout을 포착합니다.

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


class RangeSqueezeConfig(BaseModel):
    """Range Compression Breakout 전략 설정.

    NR7 패턴(N일 중 최소 range)과 range ratio(현재/평균 range)를 사용하여
    vol compression을 감지하고, breakout 방향을 추종합니다.

    Signal Formula:
        1. daily_range = high - low
        2. avg_range = daily_range.rolling(lookback).mean()
        3. range_ratio = daily_range / avg_range
        4. is_nr = daily_range == daily_range.rolling(nr_period).min()
        5. squeeze = is_nr OR range_ratio < squeeze_threshold
        6. direction = sign(close - open) when squeeze, else 0
        7. strength = direction * vol_scalar

    Attributes:
        nr_period: NR7 패턴 기간
        lookback: 평균 range 윈도우
        squeeze_threshold: squeeze 판정 range ratio
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
    # Range Compression 파라미터
    # =========================================================================
    nr_period: int = Field(
        default=7,
        ge=3,
        le=20,
        description="NR 패턴 기간 (NR7 = 7일 중 최소 range)",
    )
    lookback: int = Field(
        default=20,
        ge=10,
        le=60,
        description="평균 range 윈도우 (캔들 수)",
    )
    squeeze_threshold: float = Field(
        default=0.5,
        ge=0.2,
        le=0.9,
        description="Squeeze 판정 range ratio 임계값 (이 이하면 squeeze)",
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
        description="연환산 계수 (일봉: 365, 4시간봉: 2190)",
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
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.lookback, self.nr_period, self.atr_period) + 1
