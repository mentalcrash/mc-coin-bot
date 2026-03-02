"""CCI Consensus Multi-Scale Trend 전략 설정.

CCI(Commodity Channel Index) x 3스케일(20/60/150) consensus voting.
CCI는 MAD 정규화 기반으로 BB의 std와 수학적으로 직교적.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class CciConsensusConfig(BaseModel):
    """CCI Consensus Multi-Scale Trend 전략 설정.

    CCI(MAD 정규화) x 3스케일(20/60/150)의 consensus voting으로 추세 방향 도출.
    3개 sub-signal 중 2/3 이상 합의(majority) 시 진입.

    Signal Logic:
        1. 각 스케일에 대해 CCI 계산
        2. CCI > cci_upper → +1 (long vote), CCI < cci_lower → -1 (short vote), else 0
        3. consensus = mean(3 sub-signals)
        4. |consensus| >= entry_threshold → direction = sign(consensus)
        5. strength = |consensus| * vol_scalar

    Attributes:
        scale_short: 단기 CCI 기간 (bars).
        scale_mid: 중기 CCI 기간 (bars).
        scale_long: 장기 CCI 기간 (bars).
        cci_upper: CCI 과매수 임계값 (> 0).
        cci_lower: CCI 과매도 임계값 (< 0).
        entry_threshold: consensus 진입 임계값 (0~1). 0.34 = 3개 중 2개 합의.
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
    """

    model_config = ConfigDict(frozen=True)

    # --- CCI Scale Parameters ---
    scale_short: int = Field(default=20, ge=5, le=100)
    scale_mid: int = Field(default=60, ge=10, le=300)
    scale_long: int = Field(default=150, ge=20, le=500)

    # --- CCI Threshold Parameters ---
    cci_upper: float = Field(
        default=100.0,
        ge=50.0,
        le=300.0,
        description="CCI 과매수 임계값. +100 = 표준 overbought",
    )
    cci_lower: float = Field(
        default=-100.0,
        ge=-300.0,
        le=-50.0,
        description="CCI 과매도 임계값. -100 = 표준 oversold",
    )

    # --- Consensus Entry ---
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.34 = 3개 중 2개 합의 (2/3 ≈ 0.667 > 0.34)",
    )

    # --- Vol-Target Parameters ---
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=730.0, gt=0.0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.DISABLED)
    hedge_threshold: float = Field(default=-0.07, le=0.0)
    hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> CciConsensusConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.scale_short < self.scale_mid < self.scale_long):
            msg = (
                f"scale_short ({self.scale_short}) < scale_mid ({self.scale_mid}) "
                f"< scale_long ({self.scale_long}) 필수"
            )
            raise ValueError(msg)
        if not (self.cci_upper > 0 > self.cci_lower):
            msg = f"cci_upper ({self.cci_upper}) > 0 > cci_lower ({self.cci_lower}) 필수"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.scale_long, self.vol_window) + 10
