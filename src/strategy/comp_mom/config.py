"""Composite Momentum 전략 설정.

OHLCV 5변수 3축 직교 분해: 가격 모멘텀(방향) x 거래량(참여도) x GK변동성(환경).
연속 곱셈으로 3축 정렬 시 강한 신호, 불일치 시 자연 감쇄.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CompMomConfig(BaseModel):
    """Composite Momentum 전략 설정.

    Attributes:
        mom_period: 가격 모멘텀(ROC) 계산 기간 (bars).
        mom_zscore_window: 모멘텀 rolling z-score 윈도우.
        vol_zscore_window: 거래량 rolling z-score 윈도우.
        gk_window: GK 변동성 rolling 평균 윈도우.
        gk_zscore_window: GK 변동성 z-score 윈도우.
        composite_threshold: 복합 점수 진입 임계값.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    mom_period: int = Field(default=20, ge=5, le=120)
    mom_zscore_window: int = Field(default=60, ge=10, le=200)
    vol_zscore_window: int = Field(default=60, ge=10, le=200)
    gk_window: int = Field(default=20, ge=5, le=100)
    gk_zscore_window: int = Field(default=60, ge=10, le=200)
    composite_threshold: float = Field(default=0.5, ge=0.0, le=5.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CompMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.mom_period + self.mom_zscore_window,
                self.vol_zscore_window,
                self.gk_window + self.gk_zscore_window,
                self.vol_window,
            )
            + 10
        )
