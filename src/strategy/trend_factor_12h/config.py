"""Trend Factor Multi-Horizon 전략 설정.

JFQA 2024 Trend Factor: 5-horizon(5/10/20/40/80) risk-adjusted return 합산.
ret_h/vol_h additive ensemble = multi-scale momentum consensus. OHLCV-only.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class TrendFactorConfig(BaseModel):
    """Trend Factor Multi-Horizon 전략 설정.

    5개 horizon에서 risk-adjusted return(ret/vol)을 합산하여
    multi-scale momentum consensus를 측정한다.

    Signal Logic:
        1. 각 horizon h에 대해 trend_factor_h = ret_h / vol_h 계산
        2. trend_factor = sum(trend_factor_1, ..., trend_factor_5)
        3. direction = sign(trend_factor) if |trend_factor| >= entry_threshold
        4. strength = direction * tanh(|trend_factor| * tanh_scale) * vol_scalar

    Attributes:
        horizon_1: 최단기 lookback (bars).
        horizon_2: 단기 lookback (bars).
        horizon_3: 중기 lookback (bars).
        horizon_4: 장기 lookback (bars).
        horizon_5: 최장기 lookback (bars).
        entry_threshold: trend_factor 진입 임계값.
        tanh_scale: tanh 스케일링 계수 (strength 연속화).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    horizon_1: int = Field(default=5, ge=2, le=30)
    horizon_2: int = Field(default=10, ge=3, le=60)
    horizon_3: int = Field(default=20, ge=5, le=120)
    horizon_4: int = Field(default=40, ge=10, le=200)
    horizon_5: int = Field(default=80, ge=20, le=400)
    entry_threshold: float = Field(default=0.5, ge=0.0, le=5.0)
    tanh_scale: float = Field(default=0.3, gt=0.0, le=2.0)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> TrendFactorConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        horizons = [
            self.horizon_1,
            self.horizon_2,
            self.horizon_3,
            self.horizon_4,
            self.horizon_5,
        ]
        for i in range(len(horizons) - 1):
            if horizons[i] >= horizons[i + 1]:
                msg = f"Horizons must be strictly increasing: {horizons[i]} >= {horizons[i + 1]}"
                raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.horizon_5, self.vol_window) + 10
