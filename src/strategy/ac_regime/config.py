"""Autocorrelation Regime-Adaptive Strategy Configuration.

Rolling autocorrelation으로 trending/MR regime을 감지하고 자동 전환합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 -> 중립)
        HEDGE_ONLY: 헤지 목적 숏만 (드로다운 임계값 초과 시)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class ACRegimeConfig(BaseModel):
    """Autocorrelation Regime-Adaptive 전략 설정.

    Returns의 serial correlation 부호로 regime을 분류합니다.
    양수 AC → trending (momentum), 음수 AC → mean-reverting (contrarian).

    Signal Formula:
        1. rho = rolling_autocorrelation(returns, lag, window)
        2. sig_bound = significance_z / sqrt(window)
        3. Trending: rho > 0 AND |rho| > sig_bound → follow momentum
        4. Mean-reverting: rho < 0 AND |rho| > sig_bound → contrarian
        5. Neutral: else → flat
        6. strength = direction * vol_scalar

    Attributes:
        ac_window: Rolling autocorrelation 윈도우
        ac_lag: Autocorrelation lag
        significance_z: Bartlett bound z-score
        mom_lookback: 모멘텀 방향 lookback
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Autocorrelation 파라미터
    # =========================================================================
    ac_window: int = Field(
        default=60,
        ge=20,
        le=252,
        description="Rolling autocorrelation 윈도우 (캔들 수)",
    )
    ac_lag: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Autocorrelation lag",
    )
    significance_z: float = Field(
        default=1.96,
        ge=1.0,
        le=3.0,
        description="Bartlett bound z-score (유의수준 결정)",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="모멘텀 방향 lookback (캔들 수)",
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
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
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
        if self.ac_lag >= self.ac_window:
            msg = f"ac_lag ({self.ac_lag}) must be < ac_window ({self.ac_window})"
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.ac_window, self.mom_lookback, self.atr_period) + 1
