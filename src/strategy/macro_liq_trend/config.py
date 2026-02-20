"""Macro-Liquidity Adaptive Trend 전략 설정.

글로벌 유동성(DXY, VIX, Stablecoin, SPY)과 가격 모멘텀 정렬로 크립토 방향 예측.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MacroLiqTrendConfig(BaseModel):
    """Macro-Liquidity Adaptive Trend 전략 설정.

    Attributes:
        dxy_roc_period: DXY ROC 계산 lookback (일).
        vix_roc_period: VIX ROC 계산 lookback (일).
        spy_roc_period: SPY ROC 계산 lookback (일).
        stab_change_period: Stablecoin 변화율 계산 lookback (일).
        zscore_window: Macro composite z-score rolling window.
        liq_long_threshold: Long 진입 z-score 임계값.
        liq_short_threshold: Short 진입 z-score 임계값.
        price_mom_period: 가격 모멘텀 lookback (SMA 기간, 일).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수.
        atr_period: ATR 계산 기간.
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Macro Parameters ---
    dxy_roc_period: int = Field(default=20, ge=5, le=120)
    vix_roc_period: int = Field(default=20, ge=5, le=120)
    spy_roc_period: int = Field(default=20, ge=5, le=120)
    stab_change_period: int = Field(default=14, ge=5, le=90)
    zscore_window: int = Field(default=90, ge=20, le=365)

    # --- Thresholds ---
    liq_long_threshold: float = Field(default=0.5, ge=0.0, le=3.0)
    liq_short_threshold: float = Field(default=-0.5, ge=-3.0, le=0.0)

    # --- Price Momentum ---
    price_mom_period: int = Field(default=50, ge=10, le=200)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> MacroLiqTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.liq_short_threshold >= self.liq_long_threshold:
            msg = (
                f"liq_short_threshold ({self.liq_short_threshold}) "
                f"must be < liq_long_threshold ({self.liq_long_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        largest_window = max(
            self.dxy_roc_period,
            self.vix_roc_period,
            self.spy_roc_period,
            self.stab_change_period,
            self.zscore_window,
            self.price_mom_period,
            self.vol_window,
        )
        return largest_window + 10
