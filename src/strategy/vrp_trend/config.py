"""VRP-Trend 전략 설정.

DVOL(IV) vs RV 스프레드(VRP)와 추세 확인으로 시장 방향 예측.
고VRP=과공포 프리미엄 수취(롱), 저VRP=실제위험(숏).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VrpTrendConfig(BaseModel):
    """VRP-Trend 전략 설정.

    Attributes:
        rv_window: Realized Volatility 계산 rolling window (bars).
        vrp_ma_window: VRP 시계열의 이동평균 smoothing window.
        vrp_zscore_window: VRP z-score 계산용 rolling window.
        vrp_entry_z: VRP z-score 진입 임계값 (>= 이면 Long).
        vrp_exit_z: VRP z-score 청산/숏 전환 임계값 (<= 이면 Short).
        trend_sma_window: 추세 확인 SMA window.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 1D TF 연환산 계수 (365).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- Strategy-Specific Parameters ---
    rv_window: int = Field(default=30, ge=5, le=120)
    vrp_ma_window: int = Field(default=14, ge=3, le=60)
    vrp_zscore_window: int = Field(default=90, ge=20, le=365)
    vrp_entry_z: float = Field(default=0.5, ge=0.0, le=3.0)
    vrp_exit_z: float = Field(default=-0.5, ge=-3.0, le=0.0)
    trend_sma_window: int = Field(default=50, ge=10, le=200)

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
    def _validate_cross_fields(self) -> VrpTrendConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.vrp_entry_z <= self.vrp_exit_z:
            msg = f"vrp_entry_z ({self.vrp_entry_z}) must be > vrp_exit_z ({self.vrp_exit_z})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return (
            max(
                self.rv_window + self.vrp_ma_window + self.vrp_zscore_window,
                self.trend_sma_window,
                self.vol_window,
            )
            + 10
        )
