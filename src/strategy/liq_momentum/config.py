"""Liquidity-Adjusted Momentum Strategy Configuration.

Amihud illiquidity와 relative volume으로 유동성 상태를 분류하고,
TSMOM conviction을 스케일링하는 1H 전략입니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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


class LiqMomentumConfig(BaseModel):
    """Liquidity-Adjusted Momentum 전략 설정.

    Amihud illiquidity와 relative volume으로 유동성 상태를 분류하고,
    low liquidity에서 conviction 확대, high liquidity에서 축소합니다.

    Signal Formula:
        1. rel_vol = volume / volume.rolling(rel_vol_window).median()
        2. amihud = (|return| / volume).rolling(amihud_window).mean()
        3. amihud_pctl = amihud.rolling(amihud_pctl_window).rank(pct=True)
        4. liq_state: LOW (rel_vol < 0.5 OR amihud_pctl > 0.75) / HIGH / NORMAL
        5. mom_signal = sign(returns.rolling(mom_lookback).sum())
        6. strength = mom_signal * vol_scalar * liq_multiplier * weekend_multiplier

    Attributes:
        rel_vol_window: Relative volume 윈도우 (1H bars)
        amihud_window: Amihud 계산 윈도우 (1H bars)
        amihud_pctl_window: Amihud percentile 윈도우 (1H bars)
        rel_vol_low: 저유동성 relative volume 임계값
        rel_vol_high: 고유동성 relative volume 임계값
        amihud_pctl_high: 저유동성 Amihud percentile 임계값
        mom_lookback: Momentum lookback (1H bars, 24 = 1일)
        low_liq_multiplier: 저유동성 conviction 배수
        weekend_multiplier: 주말 conviction 배수
        amihud_pctl_low: (derived) 1 - amihud_pctl_high
        high_liq_multiplier: (derived) 2 - low_liq_multiplier
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # 유동성 파라미터
    # =========================================================================
    rel_vol_window: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Relative volume 윈도우 (1H bars, 168 = 7일)",
    )
    amihud_window: int = Field(
        default=24,
        ge=6,
        le=168,
        description="Amihud illiquidity 윈도우 (1H bars)",
    )
    amihud_pctl_window: int = Field(
        default=720,
        ge=48,
        le=2160,
        description="Amihud percentile 윈도우 (1H bars, 720 = 30일)",
    )
    rel_vol_low: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="저유동성 relative volume 임계값",
    )
    rel_vol_high: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="고유동성 relative volume 임계값",
    )
    amihud_pctl_high: float = Field(
        default=0.75,
        ge=0.5,
        le=0.99,
        description="저유동성 Amihud percentile 임계값 (높을수록 illiquid)",
    )
    # =========================================================================
    # Momentum 파라미터
    # =========================================================================
    mom_lookback: int = Field(
        default=24,
        ge=3,
        le=168,
        description="Momentum lookback (1H bars, 24 = 1일)",
    )

    # =========================================================================
    # Conviction 스케일링
    # =========================================================================
    low_liq_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="저유동성 conviction 배수 (>1 = 확대)",
    )
    weekend_multiplier: float = Field(
        default=1.2,
        ge=0.5,
        le=2.0,
        description="주말 conviction 배수",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.30,
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

    # =========================================================================
    # Derived (computed) fields — 핵심 파라미터에서 자동 도출
    # =========================================================================
    @computed_field  # type: ignore[prop-decorator]
    @property
    def amihud_pctl_low(self) -> float:
        """고유동성 Amihud percentile 임계값 (1 - amihud_pctl_high)."""
        return 1.0 - self.amihud_pctl_high

    @computed_field  # type: ignore[prop-decorator]
    @property
    def high_liq_multiplier(self) -> float:
        """고유동성 conviction 배수 (2 - low_liq_multiplier, 대칭 스케일링)."""
        return 2.0 - self.low_liq_multiplier

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증."""
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.rel_vol_low >= self.rel_vol_high:
            msg = f"rel_vol_low ({self.rel_vol_low}) must be < rel_vol_high ({self.rel_vol_high})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (1H bars)."""
        return max(self.amihud_pctl_window, self.rel_vol_window, self.atr_period) + 1
