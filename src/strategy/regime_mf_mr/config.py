"""Regime-Gated Multi-Factor Mean Reversion 전략 설정.

Ranging 레짐에서만 활성화되는 멀티팩터 평균회귀 전략.
BB + Z-score + MR score + RSI + Volume 확인으로 MR alpha 추출.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class RegimeMfMrConfig(BaseModel):
    """Regime-Gated Multi-Factor MR 전략 설정.

    크립토 MR alpha가 ranging 레짐에서만 존재한다는 가설.
    기존 MR 전략 전멸의 원인(trending 구간 손실)을 p_ranging 확률
    게이팅으로 근본 차단. 멀티팩터(BB + zscore + MR score + RSI)에
    Volume 확인을 결합.

    Attributes:
        bb_period: Bollinger Band 기간
        bb_std: Bollinger Band 표준편차 배수
        zscore_window: Z-score rolling window
        mr_score_window: Mean Reversion Score window
        mr_score_std: MR Score 표준편차 배수
        rsi_period: RSI 기간
        rsi_oversold: RSI 과매도 임계값
        rsi_overbought: RSI 과매수 임계값
        volume_ma_period: Volume MA 기간
        volume_threshold: 평균 대비 volume 배수 임계값
        min_factor_agreement: 최소 팩터 합의 수 (4개 팩터 중)
        regime_gate_threshold: ranging 확률 게이팅 임계값
        vol_target: 연환산 변동성 타겟
        vol_window: 변동성 계산 rolling window
        min_volatility: 변동성 하한
        annualization_factor: TF별 연환산 계수
        short_mode: 숏 포지션 허용 모드
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율
    """

    model_config = ConfigDict(frozen=True)

    # --- Multi-Factor MR Parameters ---
    bb_period: int = Field(default=20, ge=10, le=100)
    bb_std: float = Field(default=2.0, ge=1.0, le=4.0)
    zscore_window: int = Field(default=20, ge=10, le=100)
    mr_score_window: int = Field(default=20, ge=10, le=100)
    mr_score_std: float = Field(default=2.0, ge=1.0, le=4.0)
    rsi_period: int = Field(default=14, ge=5, le=50)
    rsi_oversold: float = Field(default=30.0, ge=10.0, le=45.0)
    rsi_overbought: float = Field(default=70.0, ge=55.0, le=90.0)

    # --- Volume Confirmation ---
    volume_ma_period: int = Field(default=20, ge=5, le=100)
    volume_threshold: float = Field(default=1.0, ge=0.5, le=3.0)

    # --- Factor Agreement ---
    min_factor_agreement: int = Field(default=2, ge=1, le=4)

    # --- Regime Gate ---
    regime_gate_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # --- Regime-Adaptive Vol Targets ---
    trending_vol_target: float = Field(default=0.0, ge=0.0, le=1.0)
    ranging_vol_target: float = Field(default=0.35, ge=0.05, le=1.0)
    volatile_vol_target: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.rsi_oversold >= self.rsi_overbought:
            msg = f"rsi_oversold ({self.rsi_oversold}) must be < rsi_overbought ({self.rsi_overbought})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        largest_window = max(
            self.bb_period,
            self.zscore_window,
            self.mr_score_window,
            self.rsi_period,
            self.volume_ma_period,
            self.vol_window,
        )
        return largest_window + 10
