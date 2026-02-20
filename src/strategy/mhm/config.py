"""MHM (Multi-Horizon Momentum) 전략 설정.

다중 룩백 기간의 모멘텀을 역변동성 가중 합산하여
robust한 추세 시그널 생성. ML 불필요 — 순수 벡터화 전략.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MHMConfig(BaseModel):
    """MHM (Multi-Horizon Momentum) 전략 설정.

    5개 horizon의 momentum을 역변동성 가중 합산하여 방향 결정.
    horizon별 부호 일치(agreement) 수로 conviction 조절.

    Attributes:
        lookback_1: 초단기 모멘텀 lookback.
        lookback_2: 단기 모멘텀 lookback.
        lookback_3: 중기 모멘텀 lookback.
        lookback_4: 중장기 모멘텀 lookback.
        lookback_5: 장기 모멘텀 lookback.
        agreement_threshold: 진입에 필요한 최소 부호 일치 수 (1~5).
        vol_target: 연환산 변동성 타겟.
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한.
        annualization_factor: 1D TF 연환산 계수.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준.
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Multi-Horizon Lookbacks ---
    lookback_1: int = Field(default=5, ge=2, le=30)
    lookback_2: int = Field(default=10, ge=3, le=60)
    lookback_3: int = Field(default=21, ge=5, le=120)
    lookback_4: int = Field(default=63, ge=20, le=252)
    lookback_5: int = Field(default=126, ge=40, le=504)

    # --- Agreement Threshold ---
    agreement_threshold: int = Field(
        default=3,
        ge=1,
        le=5,
        description="진입에 필요한 최소 부호 일치 수.",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> MHMConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        lookbacks = [
            self.lookback_1,
            self.lookback_2,
            self.lookback_3,
            self.lookback_4,
            self.lookback_5,
        ]
        for i in range(len(lookbacks) - 1):
            if lookbacks[i] >= lookbacks[i + 1]:
                msg = (
                    f"Lookbacks must be strictly increasing: "
                    f"lookback_{i + 1}={lookbacks[i]} >= lookback_{i + 2}={lookbacks[i + 1]}"
                )
                raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.lookback_5, self.vol_window) + 10
