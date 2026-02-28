"""Dual Momentum Strategy Configuration.

Dual Momentum (XSMOM 변형) 전략 설정을 정의합니다.
12H 타임프레임에 최적화된 횡단면 모멘텀 전략입니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode

# 12H 연환산 계수: 365일 * 2 (하루 2개 12H bar)
_ANNUALIZATION_12H = 730.0


class DualMomConfig(BaseModel):
    """Dual Momentum 전략 설정.

    Cross-sectional ranking은 IntraPodAllocator(DUAL_MOMENTUM)에서 수행.
    전략 자체는 per-symbol momentum signal + vol-target sizing만 담당.

    Signal Formula:
        1. rolling_return = lookback 기간 수익률
        2. vol_scalar = vol_target / realized_vol
        3. direction = sign(rolling_return) (Long-Only: 양수만)
        4. strength = direction * vol_scalar

    Attributes:
        lookback: Rolling return 계산 기간 (캔들 수)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수 (12H: 730)
        use_log_returns: 로그 수익률 사용 여부
        short_mode: 숏 포지션 처리 모드
    """

    model_config = ConfigDict(frozen=True)

    lookback: int = Field(
        default=42,
        ge=5,
        le=120,
        description="Rolling return lookback 기간 (캔들 수)",
    )
    vol_window: int = Field(
        default=30,
        ge=5,
        le=120,
        description="변동성 계산 윈도우 (캔들 수)",
    )
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
        description="최소 변동성 클램프",
    )
    annualization_factor: float = Field(
        default=_ANNUALIZATION_12H,
        gt=0,
        description="연환산 계수 (12H: 730)",
    )
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부",
    )
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드 (기본: Long-Only)",
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
        return max(self.lookback, self.vol_window) + 1
