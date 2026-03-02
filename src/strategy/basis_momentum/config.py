"""Basis-Momentum 전략 설정.

FR 변화율(1st derivative) z-score 기반 모멘텀.
FR level = lagging (시장 이미 반영), FR acceleration = leading (심리 전환 포착).

basis_mom = delta(FR, N) / std(delta_FR, M)
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class BasisMomentumConfig(BaseModel):
    """Basis-Momentum 전략 설정.

    핵심 아이디어: FR 수준이 아닌 FR 변화율(가속도)의 z-score를 추적.
    19개+ FR 레벨 전략이 전멸한 이유: FR 수준은 이미 가격에 반영.
    FR 가속도는 선행 — 심리 전환의 초기 신호.

    Signal Logic:
        1. fr_change = FR - FR.shift(fr_change_window) — N-period FR 변화
        2. fr_change_std = rolling std of 1-period FR diffs — 정규화 분모
        3. basis_mom = fr_change / fr_change_std — FR 모멘텀 z-score
        4. basis_mom > entry_zscore → LONG, < -entry_zscore → SHORT
        5. |basis_mom| < exit_zscore → FLAT (중립 복귀)

    Attributes:
        fr_change_window: FR 변화 lookback (bars). 6 x 12H = 3일.
        fr_std_window: FR diff rolling std window (정규화용).
        entry_zscore: z-score 진입 임계값.
        exit_zscore: z-score 청산 임계값 (중립 복귀).
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
    fr_change_window: int = Field(default=6, ge=2, le=50)
    fr_std_window: int = Field(default=30, ge=5, le=200)
    entry_zscore: float = Field(default=1.5, ge=0.5, le=5.0)
    exit_zscore: float = Field(default=0.5, ge=0.0, le=3.0)

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
    def _validate_cross_fields(self) -> BasisMomentumConfig:
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.exit_zscore >= self.entry_zscore:
            msg = f"exit_zscore ({self.exit_zscore}) must be < entry_zscore ({self.entry_zscore})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        return max(self.fr_change_window + self.fr_std_window, self.vol_window) + 10
