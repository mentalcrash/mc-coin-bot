"""Macro-Context-Trend 12H 전략 설정.

12H EMA 추세 시그널 + 1D 매크로 리스크 선호도 컨텍스트 사이징.
Multi-Source Architecture (12H OHLCV signal + 1D Macro context).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class MacroContextTrendConfig(BaseModel):
    """Macro-Context-Trend 12H 전략 설정.

    Attributes:
        ema_fast: 빠른 EMA lookback (bars).
        ema_slow: 느린 EMA lookback (bars).
        trend_confirm_bars: 추세 확인 최소 연속 bars.
        macro_risk_weight: 매크로 컨텍스트 사이징 가중치 (0~1).
        macro_window: 매크로 지표 rolling window.
        macro_min_weight: 매크로 필터 최소 사이징 가중치.
        macro_max_weight: 매크로 필터 최대 사이징 가중치.
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

    # --- EMA Trend Parameters ---
    ema_fast: int = Field(default=12, ge=3, le=50)
    ema_slow: int = Field(default=26, ge=10, le=100)
    trend_confirm_bars: int = Field(default=2, ge=1, le=10)

    # --- Macro Context Parameters ---
    macro_risk_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    macro_window: int = Field(default=30, ge=5, le=120)
    macro_min_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    macro_max_weight: float = Field(default=1.0, ge=0.5, le=2.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Options ---
    atr_period: int = Field(default=14, ge=5, le=50)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> MacroContextTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.ema_fast >= self.ema_slow:
            msg = f"ema_fast ({self.ema_fast}) must be < ema_slow ({self.ema_slow})"
            raise ValueError(msg)
        if self.macro_min_weight > self.macro_max_weight:
            msg = f"macro_min_weight ({self.macro_min_weight}) must be <= macro_max_weight ({self.macro_max_weight})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.ema_slow, self.vol_window, self.macro_window) + 10
