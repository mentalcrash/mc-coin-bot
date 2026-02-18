"""OI-Price Divergence Strategy Configuration.

OI-가격 괴리 + Funding Rate z-score 기반 숏스퀴즈/롱청산 감지.
BTC/ETH 전용 derivatives 전략.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class OiDivergeConfig(BaseModel):
    """OI-Price Divergence 전략 설정.

    Sweep 대상: divergence_window, fr_zscore_threshold, vol_target (3개)
    """

    model_config = ConfigDict(frozen=True)

    # --- Sweep parameters ---
    divergence_window: int = Field(
        default=14, ge=7, le=30, description="OI-price rolling correlation window"
    )
    fr_zscore_threshold: float = Field(
        default=1.5, ge=0.5, le=3.0, description="Funding rate z-score entry threshold"
    )
    vol_target: float = Field(default=0.30, ge=0.10, le=0.60, description="Annualized vol target")

    # --- Fixed parameters ---
    divergence_threshold: float = Field(
        default=-0.3, le=0.0, description="OI-price divergence threshold (negative = diverging)"
    )
    fr_ma_window: int = Field(default=3, ge=1, le=10, description="Funding rate MA window")
    fr_zscore_window: int = Field(
        default=90, ge=30, le=180, description="Funding rate z-score normalization window"
    )
    oi_momentum_period: int = Field(default=14, ge=5, le=30, description="OI momentum period")
    vol_window: int = Field(default=30, ge=5, le=200, description="Volatility window")
    min_volatility: float = Field(default=0.05, ge=0.01, le=0.50, description="Min vol clamp")
    annualization_factor: float = Field(default=365.0, description="Annualization factor")
    short_mode: ShortMode = ShortMode.FULL

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """divergence_threshold <= 0 보장."""
        if self.divergence_threshold > 0:
            msg = f"divergence_threshold must be <= 0, got {self.divergence_threshold}"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요 최소 캔들 수."""
        return max(self.fr_zscore_window, self.vol_window, self.divergence_window) + 1
