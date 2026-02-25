"""VRP-Regime Trend 전략 설정.

BTC/ETH 옵션시장 VRP(IV-RV spread)를 레짐 indicator로 활용.
고VRP + GK vol 확인 → trend-following 강화, VRP collapse 시 방어.
8H TF. Deribit DVOL + 8H Garman-Klass RV 결합.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VrpRegimeTrendConfig(BaseModel):
    """VRP-Regime Trend 전략 설정.

    Attributes:
        gk_rv_window: Garman-Klass Realized Volatility rolling window (bars).
        vrp_ma_window: VRP 시계열 이동평균 smoothing window.
        vrp_zscore_window: VRP z-score 계산용 rolling window.
        vrp_high_z: VRP z-score 고VRP 진입 임계값 (>= → Long 강화).
        vrp_low_z: VRP z-score 저VRP 전환 임계값 (<= → Short/방어).
        trend_ema_fast: 추세 판단 빠른 EMA period.
        trend_ema_slow: 추세 판단 느린 EMA period.
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
    gk_rv_window: int = Field(default=30, ge=5, le=120)
    vrp_ma_window: int = Field(default=14, ge=3, le=60)
    vrp_zscore_window: int = Field(default=90, ge=20, le=365)
    vrp_high_z: float = Field(default=0.5, ge=0.0, le=3.0)
    vrp_low_z: float = Field(default=-0.5, ge=-3.0, le=0.0)
    trend_ema_fast: int = Field(default=12, ge=3, le=50)
    trend_ema_slow: int = Field(default=36, ge=10, le=200)

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
    def _validate_cross_fields(self) -> VrpRegimeTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.vrp_high_z <= self.vrp_low_z:
            msg = f"vrp_high_z ({self.vrp_high_z}) must be > vrp_low_z ({self.vrp_low_z})"
            raise ValueError(msg)
        if self.trend_ema_fast >= self.trend_ema_slow:
            msg = (
                f"trend_ema_fast ({self.trend_ema_fast}) must be < "
                f"trend_ema_slow ({self.trend_ema_slow})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.gk_rv_window + self.vrp_ma_window + self.vrp_zscore_window,
                self.trend_ema_slow,
                self.vol_window,
            )
            + 10
        )
