"""Carry-Sentiment Gate 전략 설정.

Funding Rate carry premium을 F&G sentiment gate로 타이밍 개선.
구조적 레버리지 불균형(BIS WP 1087) + 행동편향 극단 contrarian.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CarrySentConfig(BaseModel):
    """Carry-Sentiment Gate 전략 설정.

    FR carry: 양의 FR → short (캐리 수취), 음의 FR → long.
    F&G gate: F&G 중립 구간에서만 carry 활성화.
    극단 F&G: fear 극단 → force long, greed 극단 → force short (contrarian override).

    Attributes:
        fr_lookback: Funding rate rolling mean window (1D bars).
        fr_zscore_window: FR z-score normalization window.
        fr_entry_threshold: 진입을 위한 최소 |avg_FR|.
        fg_fear_threshold: Fear 극단 임계값 (이하 → contrarian long).
        fg_greed_threshold: Greed 극단 임계값 (이상 → contrarian short).
        fg_gate_low: F&G gate 하한 (이상에서 carry 활성).
        fg_gate_high: F&G gate 상한 (이하에서 carry 활성).
        fg_ma_window: F&G smoothing window (1D bars).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Funding Rate Parameters ---
    fr_lookback: int = Field(
        default=3,
        ge=1,
        le=30,
        description="FR rolling mean window (1D bars).",
    )
    fr_zscore_window: int = Field(
        default=90,
        ge=10,
        le=365,
        description="FR z-score normalization window.",
    )
    fr_entry_threshold: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.01,
        description="Minimum |avg_FR| for carry entry.",
    )

    # --- Fear & Greed Parameters ---
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
    fg_gate_low: int = Field(
        default=30,
        ge=15,
        le=45,
        description="F&G gate lower bound (carry active above).",
    )
    fg_gate_high: int = Field(
        default=70,
        ge=55,
        le=85,
        description="F&G gate upper bound (carry active below).",
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
    def _validate_cross_fields(self) -> CarrySentConfig:
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
        if self.fg_gate_low >= self.fg_gate_high:
            msg = f"fg_gate_low ({self.fg_gate_low}) must be < fg_gate_high ({self.fg_gate_high})"
            raise ValueError(msg)
        if self.fg_fear_threshold >= self.fg_gate_low:
            msg = (
                f"fg_fear_threshold ({self.fg_fear_threshold}) "
                f"must be < fg_gate_low ({self.fg_gate_low}) to avoid zone overlap"
            )
            raise ValueError(msg)
        if self.fg_gate_high >= self.fg_greed_threshold:
            msg = (
                f"fg_gate_high ({self.fg_gate_high}) "
                f"must be < fg_greed_threshold ({self.fg_greed_threshold}) to avoid zone overlap"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_zscore_window, self.fg_ma_window, self.vol_window) + 10
