"""Asymmetric Semivariance MR 전략 설정.

방향별 semivariance 비율로 공포/탐욕 과잉반응 감지 -> contrarian 진입.
Downside semivariance 예측력이 total variance 대비 우수 (Kyei-Mensah 2024).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class AsymSemivarMRConfig(BaseModel):
    """Asymmetric Semivariance MR 전략 설정.

    Rolling window에서 upside/downside semivariance를 계산하고,
    비율(semivar ratio)의 극단값에서 contrarian 진입합니다.

    Signal Formula:
        1. downside_semivar = E[min(r, 0)^2] over rolling window
        2. upside_semivar = E[max(r, 0)^2] over rolling window
        3. semivar_ratio = downside_semivar / (downside_semivar + upside_semivar)
        4. z_score = (ratio - rolling_mean) / rolling_std
        5. Long (capitulation): z > entry_zscore (downside fear spike)
        6. Short (euphoria): z < -entry_zscore (upside greed spike)
        7. Exit: |z| < exit_zscore OR timeout

    Attributes:
        semivar_window: Semivariance 계산 rolling window (캔들 수)
        zscore_window: Z-score 계산 rolling window (캔들 수)
        entry_zscore: 진입 Z-score 임계값
        exit_zscore: 청산 Z-score 임계값
        exit_timeout_bars: 타임아웃 청산 바 수
        mom_lookback: 변동성 계산 lookback (캔들 수)
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수 (4H: 2190)
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Semivariance 파라미터
    # =========================================================================
    semivar_window: int = Field(
        default=60,
        ge=20,
        le=300,
        description="Semivariance 계산 rolling window (캔들 수, 60 = 4H*60 = 10일)",
    )
    zscore_window: int = Field(
        default=120,
        ge=30,
        le=500,
        description="Semivar ratio Z-score 계산 rolling window (캔들 수)",
    )
    entry_zscore: float = Field(
        default=1.5,
        ge=0.5,
        le=4.0,
        description="진입 Z-score 임계값",
    )
    exit_zscore: float = Field(
        default=0.5,
        ge=0.1,
        le=1.5,
        description="청산 Z-score 임계값",
    )
    exit_timeout_bars: int = Field(
        default=30,
        ge=10,
        le=60,
        description="타임아웃 청산 바 수 (30 = 5일 @4H)",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 계산 lookback (캔들 수)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프 (0으로 나누기 방지)",
    )
    annualization_factor: float = Field(
        default=2190.0,
        gt=0,
        description="연환산 계수 (4H: 6*365=2190)",
    )

    # =========================================================================
    # 옵션
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (Trailing Stop용)",
    )

    # =========================================================================
    # 숏 모드 설정
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
    )
    hedge_threshold: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.05,
        description="헤지 숏 활성화 드로다운 임계값 (예: -0.07 = -7%)",
    )
    hedge_strength_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="헤지 숏 강도 비율 (롱 대비, 예: 0.8 = 80%)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증."""
        if self.exit_zscore >= self.entry_zscore:
            msg = f"exit_zscore ({self.exit_zscore}) must be < entry_zscore ({self.entry_zscore})"
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.semivar_window, self.zscore_window, self.mom_lookback, self.atr_period) + 10
