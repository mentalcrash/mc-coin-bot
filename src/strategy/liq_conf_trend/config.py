"""Liquidity-Confirmed Trend 전략 설정.

On-chain 유동성 복합지수(stablecoin supply + TVL)가 가격 모멘텀 방향을 확인할 때만 진입.
F&G 극단 override로 행동편향 포착. 3개 글로벌 on-chain 소스 활용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class LiqConfTrendConfig(BaseModel):
    """Liquidity-Confirmed Trend 전략 설정.

    Attributes:
        mom_lookback: 가격 모멘텀 계산 lookback (1D bars).
        stablecoin_roc_window: Stablecoin supply ROC 계산 window.
        tvl_roc_window: TVL ROC 계산 window.
        liq_score_threshold: 유동성 composite score 진입 임계값 (0~2).
        fg_fear_threshold: F&G fear extreme (이하 → contrarian long).
        fg_greed_threshold: F&G greed extreme (이상 → contrarian short).
        fg_ma_window: F&G smoothing MA window (1D bars).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Momentum Parameters ---
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Price momentum lookback (1D bars).",
    )

    # --- Liquidity Parameters ---
    stablecoin_roc_window: int = Field(
        default=14,
        ge=5,
        le=60,
        description="Stablecoin supply ROC window.",
    )
    tvl_roc_window: int = Field(
        default=14,
        ge=5,
        le=60,
        description="TVL ROC window.",
    )
    liq_score_threshold: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Min liquidity score for confirmation (1 = any, 2 = both).",
    )

    # --- Fear & Greed Override Parameters ---
    fg_fear_threshold: int = Field(
        default=20,
        ge=5,
        le=35,
        description="Fear extreme threshold (below → contrarian long).",
    )
    fg_greed_threshold: int = Field(
        default=80,
        ge=65,
        le=95,
        description="Greed extreme threshold (above → contrarian short).",
    )
    fg_ma_window: int = Field(
        default=14,
        ge=5,
        le=60,
        description="F&G smoothing MA window (1D bars).",
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
    def _validate_cross_fields(self) -> LiqConfTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.fg_fear_threshold >= self.fg_greed_threshold:
            msg = (
                f"fg_fear_threshold ({self.fg_fear_threshold}) "
                f"must be < fg_greed_threshold ({self.fg_greed_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.mom_lookback,
                self.stablecoin_roc_window,
                self.tvl_roc_window,
                self.fg_ma_window,
                self.vol_window,
            )
            + 10
        )
