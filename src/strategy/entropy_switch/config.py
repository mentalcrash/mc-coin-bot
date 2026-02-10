"""Entropy Regime Switch Strategy Configuration.

Shannon Entropy로 시장 예측가능성을 측정합니다.

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


class EntropySwitchConfig(BaseModel):
    """Entropy Regime Switch 전략 설정.

    Shannon Entropy로 시장 예측가능성을 측정하여,
    낮은 엔트로피(규칙적 패턴)에서만 추세추종 진입.

    Signal Formula:
        1. entropy = scipy.stats.entropy(histogram(returns, bins))
        2. Low entropy + positive momentum → LONG
        3. Low entropy + negative momentum → SHORT (HEDGE_ONLY)
        4. High entropy → FLAT
        5. strength = direction * vol_scalar

    Attributes:
        entropy_window: Entropy 계산 윈도우 (캔들 수)
        entropy_bins: Entropy 계산 히스토그램 빈 수
        entropy_low_threshold: 낮은 엔트로피 임계값 (진입 조건)
        entropy_high_threshold: 높은 엔트로피 임계값 (거래 중단)
        mom_lookback: 모멘텀 방향 lookback
        adx_period: ADX 계산 기간
        adx_threshold: ADX 보조 필터 임계값
        use_adx_filter: ADX 필터 사용 여부
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
    # Entropy 파라미터
    # =========================================================================
    entropy_window: int = Field(
        default=120,
        ge=30,
        le=500,
        description="Entropy 계산 윈도우 (캔들 수)",
    )
    entropy_bins: int = Field(
        default=10,
        ge=5,
        le=30,
        description="히스토그램 빈 수",
    )
    entropy_low_threshold: float = Field(
        default=1.8,
        ge=0.5,
        le=3.0,
        description="낮은 엔트로피 임계값 (진입 조건)",
    )
    entropy_high_threshold: float = Field(
        default=2.2,
        ge=1.0,
        le=3.5,
        description="높은 엔트로피 임계값 (거래 중단)",
    )

    # =========================================================================
    # 모멘텀 파라미터
    # =========================================================================
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="모멘텀 방향 lookback (캔들 수)",
    )

    # =========================================================================
    # ADX 보조 필터
    # =========================================================================
    adx_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ADX 계산 기간",
    )
    adx_threshold: float = Field(
        default=20.0,
        ge=10.0,
        le=40.0,
        description="ADX 보조 필터 임계값",
    )
    use_adx_filter: bool = Field(
        default=True,
        description="ADX 필터 사용 여부",
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
        default=ShortMode.HEDGE_ONLY,
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
        if self.entropy_low_threshold >= self.entropy_high_threshold:
            msg = (
                f"entropy_low_threshold ({self.entropy_low_threshold}) "
                f"must be < entropy_high_threshold ({self.entropy_high_threshold})"
            )
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.entropy_window, self.mom_lookback, self.adx_period, self.atr_period) + 1
