"""Donchian Filtered 전략 설정.

Donch-Multi 3-scale consensus + funding rate crowd filter.
과열 포지셔닝 시 진입 억제로 false breakout 필터링.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.donch_multi.config import ShortMode


class DonchFilteredConfig(BaseModel):
    """Donchian Filtered 전략 설정.

    Donch-Multi 파라미터를 그대로 포함하고,
    funding rate z-score 기반 crowd filter 파라미터를 추가한다.

    Crowd Filter Logic:
        - fr_zscore > threshold AND consensus LONG → 롱 과열, 진입 억제
        - fr_zscore < -threshold AND consensus SHORT → 숏 과열, 진입 억제
        - 그 외 → donch-multi 시그널 그대로 통과

    Attributes:
        lookback_short: 단기 Donchian lookback (bars).
        lookback_mid: 중기 Donchian lookback (bars).
        lookback_long: 장기 Donchian lookback (bars).
        entry_threshold: consensus 진입 임계값 (0~1).
        vol_target: 연환산 변동성 타겟 (0~1).
        vol_window: 변동성 계산 rolling window.
        min_volatility: 변동성 하한 (0 나눗셈 방지).
        annualization_factor: 12H TF 연환산 계수 (730).
        short_mode: 숏 포지션 허용 모드.
        hedge_threshold: HEDGE_ONLY 활성화 drawdown 기준 (<=0).
        hedge_strength_ratio: HEDGE_ONLY 숏 강도 감쇄 비율.
        fr_ma_window: funding rate MA smoothing window.
        fr_zscore_window: z-score 정규화 window.
        fr_suppress_threshold: 과열 억제 z-score 임계값 (sweep 대상).
    """

    model_config = ConfigDict(frozen=True)

    # --- Donch-Multi Parameters (동일) ---
    lookback_short: int = Field(default=20, ge=5, le=100)
    lookback_mid: int = Field(default=40, ge=10, le=200)
    lookback_long: int = Field(default=80, ge=20, le=400)
    entry_threshold: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="consensus 진입 임계값. 0.34 = 3개 중 2개 합의 필요",
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

    # --- Funding Rate Crowd Filter ---
    fr_ma_window: int = Field(
        default=3, ge=1, le=10, description="Funding rate MA smoothing window"
    )
    fr_zscore_window: int = Field(
        default=90, ge=30, le=180, description="Funding rate z-score normalization window"
    )
    fr_suppress_threshold: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="과열 억제 z-score 임계값 (sweep 대상)",
    )

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> DonchFilteredConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if not (self.lookback_short < self.lookback_mid < self.lookback_long):
            msg = (
                f"lookback_short ({self.lookback_short}) < lookback_mid ({self.lookback_mid}) "
                f"< lookback_long ({self.lookback_long}) 필수"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.lookback_long, self.vol_window, self.fr_zscore_window) + 10
