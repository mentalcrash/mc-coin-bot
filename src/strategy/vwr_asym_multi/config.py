"""VWR Asymmetric Multi-Scale 전략 설정.

12H Volume-Weighted Returns 다중스케일 앙상블 + 비대칭 임계값(long != short).
정보 비대칭(volume weight = 기관 방향성) + crypto 구조적 drift 비대칭성 활용.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VwrAsymMultiConfig(BaseModel):
    """VWR Asymmetric Multi-Scale 전략 설정.

    3-scale VWR(10/21/42) 앙상블 + 비대칭 long/short 임계값으로
    crypto의 구조적 long drift와 short squeeze 비대칭을 반영한다.

    Signal Logic:
        1. 각 lookback(10/21/42)에서 volume-weighted return 계산
        2. VWR z-score = (vwr - rolling_mean) / rolling_std
        3. consensus = mean(zscore_short, zscore_mid, zscore_long)
        4. direction = +1 if consensus > long_threshold,
                       -1 if consensus < -short_threshold (비대칭),
                        0 otherwise
        5. strength = |consensus| * vol_scalar

    Attributes:
        lookback_short: 단기 VWR lookback (bars).
        lookback_mid: 중기 VWR lookback (bars).
        lookback_long: 장기 VWR lookback (bars).
        zscore_window: VWR z-score rolling window.
        long_threshold: 롱 진입 consensus z-score 임계값.
        short_threshold: 숏 진입 consensus z-score 절대값 임계값 (long보다 높아 비대칭).
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
    lookback_short: int = Field(default=10, ge=3, le=50)
    lookback_mid: int = Field(default=21, ge=5, le=100)
    lookback_long: int = Field(default=42, ge=10, le=200)
    zscore_window: int = Field(default=60, ge=20, le=200)
    long_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=3.0,
        description="롱 진입 consensus z-score 임계값",
    )
    short_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=3.0,
        description="숏 진입 consensus z-score 절대값 임계값 (비대칭: long보다 높음)",
    )

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
    def _validate_cross_fields(self) -> VwrAsymMultiConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.lookback_short < self.lookback_mid < self.lookback_long):
            msg = (
                f"lookback_short ({self.lookback_short}) < "
                f"lookback_mid ({self.lookback_mid}) < "
                f"lookback_long ({self.lookback_long}) 필수"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        # lookback_long + zscore_window + shift(1) 여유
        return max(self.lookback_long + self.zscore_window, self.vol_window) + 10
