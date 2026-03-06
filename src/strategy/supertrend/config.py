"""SuperTrend 전략 설정.

ATR 기반 동적 지지/저항선으로 추세 전환을 감지하는 추세추종 전략.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 → 중립)
        HEDGE_ONLY: 헤지 목적 숏만 (드로다운 임계값 초과 시)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class SuperTrendConfig(BaseModel):
    """SuperTrend 전략 설정.

    Attributes:
        atr_period: ATR 계산 기간.
        multiplier: ATR 배수 (클수록 둔감).
        adx_period: ADX 계산 기간.
        adx_threshold: ADX 최소 기준값 (0이면 필터 비활성).
        short_mode: 숏 포지션 처리 모드.
    """

    model_config = ConfigDict(frozen=True)

    atr_period: int = Field(default=10, ge=5, le=50)
    multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    adx_period: int = Field(default=14, ge=5, le=50)
    adx_threshold: float = Field(default=0.0, ge=0.0, le=80.0)
    risk_per_trade: float = Field(default=0.0, ge=0.0, le=0.1)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=10.0)
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
    )

    # ==========================================================================
    # Pyramiding (분할 진입)
    # ==========================================================================
    use_pyramiding: bool = Field(
        default=False,
        description="분할 진입 활성화 여부",
    )
    pyramid_stage1_pct: float = Field(
        default=0.40,
        ge=0.1,
        le=1.0,
        description="1차 진입 비중 (시그널 발생)",
    )
    pyramid_stage2_pct: float = Field(
        default=0.35,
        ge=0.0,
        le=0.6,
        description="2차 추가 비중 (신고점 돌파)",
    )
    pyramid_stage3_pct: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="3차 추가 비중 (추세 강화)",
    )
    pyramid_high_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description="신고점 판단 Donchian 룩백 기간",
    )
    pyramid_adx_strong: float = Field(
        default=30.0,
        ge=20.0,
        le=60.0,
        description="추세 강화 ADX 임계값 (stage 3)",
    )

    @property
    def use_adx_filter(self) -> bool:
        """ADX 필터 활성 여부."""
        return self.adx_threshold > 0.0

    @property
    def use_risk_sizing(self) -> bool:
        """ATR 기반 리스크 사이징 활성 여부."""
        return self.risk_per_trade > 0.0

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        base = self.atr_period * 3 + 10
        if self.use_adx_filter:
            base = max(base, self.adx_period * 3 + 10)
        return base
