"""Hurst/ER Regime Strategy Configuration.

Efficiency Ratio + R/S Hurst exponent 기반 regime 판별 전략의 설정을 정의합니다.
Trending regime에서 momentum following, mean-reverting regime에서 z-score fading.

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


class HurstRegimeConfig(BaseModel):
    """Hurst/ER Regime 전략 설정.

    Efficiency Ratio와 R/S Hurst exponent를 결합하여 시장 regime
    (trending / mean-reverting / neutral)을 판별하고, regime별 최적 전략을 적용합니다.

    Signal Formula:
        1. ER > er_trend_threshold OR Hurst > hurst_trend_threshold → Trending regime
        2. ER < er_mr_threshold OR Hurst < hurst_mr_threshold → Mean-Reversion regime
        3. Trending: momentum following (sign(cumulative returns) * vol_scalar)
        4. MR: z-score fading (-sign(z_score) * vol_scalar)
        5. Neutral: reduced momentum (sign(momentum) * 0.5 * vol_scalar)

    Attributes:
        er_lookback: Efficiency Ratio 계산 lookback
        hurst_window: Rolling Hurst exponent 윈도우
        mom_lookback: 모멘텀 lookback (trending regime)
        mr_lookback: Mean-reversion lookback (z-score 계산)
        er_trend_threshold: ER 기반 trending regime 임계값
        er_mr_threshold: ER 기반 mean-reversion regime 임계값
        hurst_trend_threshold: Hurst 기반 trending regime 임계값
        hurst_mr_threshold: Hurst 기반 mean-reversion regime 임계값
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율

    Example:
        >>> config = HurstRegimeConfig(
        ...     er_lookback=20,
        ...     hurst_window=100,
        ...     er_trend_threshold=0.6,
        ...     er_mr_threshold=0.3,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Regime 판별 파라미터
    # =========================================================================
    er_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Efficiency Ratio lookback (캔들 수)",
    )
    hurst_window: int = Field(
        default=100,
        ge=50,
        le=252,
        description="Rolling Hurst exponent 윈도우 (캔들 수)",
    )

    # =========================================================================
    # 전략별 Lookback
    # =========================================================================
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="모멘텀 lookback (trending regime, 캔들 수)",
    )
    mr_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Mean-reversion lookback (z-score 계산, 캔들 수)",
    )

    # =========================================================================
    # Regime 임계값
    # =========================================================================
    er_trend_threshold: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="ER 기반 trending regime 임계값 (이 이상이면 trending)",
    )
    er_mr_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.6,
        description="ER 기반 mean-reversion regime 임계값 (이 이하면 MR)",
    )
    hurst_trend_threshold: float = Field(
        default=0.55,
        ge=0.50,
        le=0.70,
        description="Hurst 기반 trending regime 임계값 (이 이상이면 trending)",
    )
    hurst_mr_threshold: float = Field(
        default=0.45,
        ge=0.30,
        le=0.50,
        description="Hurst 기반 mean-reversion regime 임계값 (이 이하면 MR)",
    )

    # =========================================================================
    # 변동성 파라미터
    # =========================================================================
    vol_window: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.40,
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
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
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
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        if self.er_trend_threshold <= self.er_mr_threshold:
            msg = (
                f"er_trend_threshold ({self.er_trend_threshold}) must be > "
                f"er_mr_threshold ({self.er_mr_threshold})"
            )
            raise ValueError(msg)

        if self.hurst_trend_threshold <= self.hurst_mr_threshold:
            msg = (
                f"hurst_trend_threshold ({self.hurst_trend_threshold}) must be > "
                f"hurst_mr_threshold ({self.hurst_mr_threshold})"
            )
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        전략 계산을 시작하기 전 필요한 최소 데이터 양입니다.
        Rolling 계산의 초기 NaN을 피하기 위해 사용됩니다.

        Returns:
            필요한 캔들 수
        """
        return (
            max(
                self.hurst_window,
                self.er_lookback,
                self.mom_lookback,
                self.mr_lookback,
                self.vol_window,
                self.atr_period,
            )
            + 1
        )
