"""CVD Divergence 8H 전략 설정.

Price-CVD (Cumulative Volume Delta) 괴리를 감지하여 추세 반전/지속 신호 포착.
CVD 데이터는 Coinalyze daily buy_volume (deriv_ext), merge_asof로 8H에 forward-fill.
BTC/ETH 전용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CvdDiverge8hConfig(BaseModel):
    """CVD Divergence 8H 전략 설정.

    Attributes:
        cvd_lookback: Divergence 감지 lookback (bars, ~7일 at 8H).
        cvd_ma_window: CVD EMA smoothing window.
        price_ma_window: 가격 EMA smoothing window.
        divergence_threshold: Divergence z-score 진입 임계값.
        trend_ema_window: 추세 확인 EMA window (21일 at 8H = 63 bars).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 8H TF 연환산 계수 (1095).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    cvd_lookback: int = Field(default=21, ge=5, le=100)
    cvd_ma_window: int = Field(default=14, ge=3, le=60)
    price_ma_window: int = Field(default=14, ge=3, le=60)
    divergence_threshold: float = Field(default=0.5, ge=0.0, le=3.0)
    trend_ema_window: int = Field(default=63, ge=10, le=200)

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=1095.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CvdDiverge8hConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.trend_ema_window, self.cvd_lookback, self.vol_window) + 10
