"""VWAP Disposition Momentum Strategy Configuration.

Rolling VWAP를 시장 참여자의 평균 취득가 proxy로 사용합니다.

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


class VWAPDispositionConfig(BaseModel):
    """VWAP Disposition Momentum 전략 설정.

    Rolling VWAP를 시장 참여자의 평균 취득가(cost basis)로 사용하여,
    미실현 이익/손실 수준(Capital Gains Overhang)에 따른 매도/매수 압력을 예측합니다.

    Signal Formula:
        1. vwap = rolling_vwap(price, volume, window)
        2. cgo = (close - vwap) / vwap
        3. cgo < -overhang_low AND volume_spike → LONG (항복 매도 후 반등)
        4. cgo > +overhang_high AND volume_decline → SHORT (차익 실현 압력)
        5. middle zone → follow momentum direction
        6. strength = direction * vol_scalar

    Attributes:
        vwap_window: Rolling VWAP 윈도우
        overhang_high: 과도한 미실현 이익 임계값
        overhang_low: 과도한 미실현 손실 임계값
        vol_ratio_window: Volume ratio 계산 윈도우
        vol_spike_threshold: Volume spike 기준
        vol_decline_threshold: Volume decline 기준
        use_volume_confirm: Volume confirmation 사용 여부
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
    # VWAP 파라미터
    # =========================================================================
    vwap_window: int = Field(
        default=720,
        ge=100,
        le=2000,
        description="Rolling VWAP 윈도우 (캔들 수, 720=120일@4H)",
    )

    # =========================================================================
    # CGO 임계값
    # =========================================================================
    overhang_high: float = Field(
        default=0.15,
        ge=0.05,
        le=0.50,
        description="과도한 미실현 이익 임계값 (SHORT)",
    )
    overhang_low: float = Field(
        default=0.10,
        ge=0.03,
        le=0.30,
        description="과도한 미실현 손실 임계값 (LONG)",
    )

    # =========================================================================
    # Volume confirmation 파라미터
    # =========================================================================
    vol_ratio_window: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Volume ratio 계산 윈도우",
    )
    vol_spike_threshold: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Volume spike 기준 (평균 대비)",
    )
    vol_decline_threshold: float = Field(
        default=0.7,
        ge=0.3,
        le=1.0,
        description="Volume decline 기준 (평균 대비)",
    )
    use_volume_confirm: bool = Field(
        default=True,
        description="Volume confirmation 사용 여부",
    )

    # =========================================================================
    # 모멘텀 파라미터
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
        return max(self.vwap_window, self.vol_ratio_window, self.mom_lookback, self.atr_period) + 1
