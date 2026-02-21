"""Stablecoin Composition Shift Strategy Configuration.

USDT/USDC 구성비 변화율 기반 리스크 선호도 시그널.
Sweep 대상: roc_short_window, roc_long_window (2개)
"""

from pydantic import BaseModel, ConfigDict, Field

from src.strategy.tsmom.config import ShortMode


class StabCompConfig(BaseModel):
    """Stablecoin Composition Shift 전략 설정."""

    model_config = ConfigDict(frozen=True)

    # --- Sweep parameters ---
    roc_short_window: int = Field(
        default=7, ge=3, le=14, description="Short-term ROC window (days)"
    )
    roc_long_window: int = Field(
        default=30, ge=14, le=60, description="Long-term ROC window (days)"
    )

    # --- Fixed parameters ---
    vol_target: float = Field(default=0.30, ge=0.10, le=0.60, description="Annualized vol target")
    vol_window: int = Field(default=30, ge=5, le=200, description="Volatility window")
    min_volatility: float = Field(default=0.05, ge=0.01, le=0.50, description="Min vol clamp")
    annualization_factor: float = Field(default=365.0, description="Annualization factor")
    short_mode: ShortMode = ShortMode.FULL

    def warmup_periods(self) -> int:
        """필요 최소 캔들 수."""
        return self.roc_long_window + self.vol_window + 10
