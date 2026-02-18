"""Hurst-Adaptive Strategy Configuration.

Hurst Exponent + Efficiency Ratio 기반 레짐 감지.
추세 → 모멘텀 추종, 횡보 → 평균회귀.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class HurstAdaptiveConfig(BaseModel):
    """Hurst-Adaptive 전략 설정.

    Sweep 대상: hurst_window, er_period, vol_target (3개)
    """

    model_config = ConfigDict(frozen=True)

    # --- Sweep parameters ---
    hurst_window: int = Field(default=100, ge=50, le=200, description="Hurst exponent window")
    er_period: int = Field(default=20, ge=10, le=50, description="Efficiency Ratio period")
    vol_target: float = Field(default=0.35, ge=0.10, le=0.60, description="Annualized vol target")

    # --- Fixed parameters ---
    hurst_trend_threshold: float = Field(
        default=0.55, ge=0.50, le=0.65, description="Hurst trending threshold"
    )
    hurst_mr_threshold: float = Field(
        default=0.45, ge=0.35, le=0.50, description="Hurst mean-reversion threshold"
    )
    er_trend_threshold: float = Field(
        default=0.40, ge=0.20, le=0.60, description="ER trending threshold"
    )
    trend_mom_lookback: int = Field(default=20, ge=5, le=60, description="Trend momentum lookback")
    mr_window: int = Field(default=20, ge=5, le=60, description="Mean reversion score window")
    mr_std_mult: float = Field(default=2.0, ge=1.0, le=3.0, description="MR z-score clip")
    vol_window: int = Field(default=30, ge=5, le=200, description="Volatility window")
    min_volatility: float = Field(default=0.05, ge=0.01, le=0.50, description="Min vol clamp")
    annualization_factor: float = Field(default=365.0, description="Annualization factor")
    short_mode: ShortMode = ShortMode.HEDGE_ONLY
    hedge_threshold: float = Field(
        default=-0.05, le=0.0, description="Drawdown threshold for hedge shorts"
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """hurst_trend > hurst_mr 보장."""
        if self.hurst_trend_threshold <= self.hurst_mr_threshold:
            msg = (
                f"hurst_trend_threshold ({self.hurst_trend_threshold}) must be > "
                f"hurst_mr_threshold ({self.hurst_mr_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요 최소 캔들 수."""
        return max(self.hurst_window, self.vol_window, self.trend_mom_lookback, self.mr_window) + 1
