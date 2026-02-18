"""On-chain Accumulation Strategy Configuration.

MVRV + Exchange Flow + Stablecoin 3지표 다수결.
BTC/ETH 전용, Long-only.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class OnchainAccumConfig(BaseModel):
    """On-chain Accumulation 전략 설정.

    Sweep 대상: mvrv_undervalued, flow_threshold, stablecoin_roc_threshold (3개)
    """

    model_config = ConfigDict(frozen=True)

    # --- Sweep parameters ---
    mvrv_undervalued: float = Field(
        default=1.0, ge=0.5, le=1.5, description="MVRV undervalued threshold"
    )
    flow_threshold: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Net flow z-score threshold"
    )
    stablecoin_roc_threshold: float = Field(
        default=0.02, ge=0.005, le=0.05, description="Stablecoin ROC threshold"
    )

    # --- Fixed parameters ---
    mvrv_overvalued: float = Field(
        default=3.0, ge=2.0, le=5.0, description="MVRV overvalued threshold"
    )
    flow_zscore_window: int = Field(
        default=90, ge=30, le=180, description="Net flow z-score window"
    )
    stablecoin_roc_window: int = Field(
        default=14, ge=7, le=30, description="Stablecoin pct_change window"
    )
    vol_target: float = Field(default=0.30, ge=0.10, le=0.60, description="Annualized vol target")
    vol_window: int = Field(default=30, ge=5, le=200, description="Volatility window")
    min_volatility: float = Field(default=0.05, ge=0.01, le=0.50, description="Min vol clamp")
    annualization_factor: float = Field(default=365.0, description="Annualization factor")
    short_mode: ShortMode = ShortMode.DISABLED

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """mvrv_undervalued < mvrv_overvalued 보장."""
        if self.mvrv_undervalued >= self.mvrv_overvalued:
            msg = (
                f"mvrv_undervalued ({self.mvrv_undervalued}) must be < "
                f"mvrv_overvalued ({self.mvrv_overvalued})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요 최소 캔들 수."""
        return max(self.flow_zscore_window, self.vol_window) + 1
