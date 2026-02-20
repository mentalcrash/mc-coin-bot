"""Residual Momentum 전략 설정.

시장 factor(BTC return) + 변동성 factor를 회귀 제거한 잔차의 모멘텀으로
자산 고유 정보의 느린 확산(slow diffusion)을 포착.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class ResidualMomConfig(BaseModel):
    """Residual Momentum 전략 설정.

    Rolling OLS로 자산 수익률을 시장(BTC) 수익률에 대해 회귀하여
    잔차를 추출하고, 잔차의 rolling 합으로 momentum을 측정한다.

    Attributes:
        regression_window: Rolling OLS 회귀 window (일 수).
        residual_lookback: 잔차 모멘텀 측정 기간 (rolling sum window).
        entry_threshold: long/short 진입 z-score 임계값.
        exit_threshold: 포지션 청산 z-score 임계값.
        zscore_window: 잔차 모멘텀 z-score 정규화 window.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: TF별 연환산 계수 (1D=365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    regression_window: int = Field(default=60, ge=20, le=252)
    residual_lookback: int = Field(default=21, ge=5, le=120)
    entry_threshold: float = Field(default=1.0, ge=0.1, le=3.0)
    exit_threshold: float = Field(default=0.3, ge=0.0, le=2.0)
    zscore_window: int = Field(default=60, ge=10, le=252)

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
    def _validate_cross_fields(self) -> ResidualMomConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.exit_threshold >= self.entry_threshold:
            msg = (
                f"exit_threshold ({self.exit_threshold}) must be < "
                f"entry_threshold ({self.entry_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(self.regression_window, self.zscore_window, self.vol_window)
            + self.residual_lookback
            + 10
        )
