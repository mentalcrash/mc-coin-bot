"""Vol Structure Regime Strategy Configuration.

이 모듈은 Vol Structure Regime 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
Short/long vol ratio와 normalized momentum으로 3 regime을 분류합니다.

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


class VolStructureConfig(BaseModel):
    """Vol Structure Regime 전략 설정.

    Short/long volatility ratio와 normalized momentum을 사용하여
    expansion / neutral / contraction 3가지 regime을 분류하고
    각 regime별 포지셔닝 전략을 적용합니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. vol_ratio = vol_short / vol_long
        2. norm_momentum = returns_sum / returns_std (rolling)
        3. Expansion: vol_ratio > threshold AND abs(norm_mom) > threshold → follow trend
        4. Contraction: vol_ratio < threshold AND abs(norm_mom) < threshold → flat
        5. Neutral: moderate signal (±0.5 * direction)
        6. strength = direction * vol_scalar

    Attributes:
        vol_short_window: 단기 변동성 윈도우
        vol_long_window: 장기 변동성 윈도우
        mom_window: 모멘텀 계산 윈도우
        expansion_vol_ratio: Expansion regime vol ratio 임계값
        contraction_vol_ratio: Contraction regime vol ratio 임계값
        expansion_mom_threshold: Expansion regime 모멘텀 임계값
        contraction_mom_threshold: Contraction regime 모멘텀 임계값
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율

    Example:
        >>> config = VolStructureConfig(
        ...     vol_short_window=10,
        ...     vol_long_window=60,
        ...     expansion_vol_ratio=1.2,
        ...     contraction_vol_ratio=0.8,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Volatility Structure 파라미터
    # =========================================================================
    vol_short_window: int = Field(
        default=10,
        ge=5,
        le=30,
        description="단기 변동성 윈도우 (캔들 수)",
    )
    vol_long_window: int = Field(
        default=60,
        ge=30,
        le=120,
        description="장기 변동성 윈도우 (캔들 수)",
    )
    mom_window: int = Field(
        default=20,
        ge=10,
        le=60,
        description="모멘텀 계산 윈도우 (캔들 수)",
    )

    # =========================================================================
    # Regime 임계값
    # =========================================================================
    expansion_vol_ratio: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Expansion regime vol ratio 임계값 (이 이상이면 expansion)",
    )
    contraction_vol_ratio: float = Field(
        default=0.8,
        ge=0.3,
        le=1.0,
        description="Contraction regime vol ratio 임계값 (이 이하면 contraction)",
    )
    expansion_mom_threshold: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="Expansion regime 모멘텀 임계값 (abs(norm_mom) > 이 값)",
    )
    contraction_mom_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=1.5,
        description="Contraction regime 모멘텀 임계값 (abs(norm_mom) < 이 값)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
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
        # vol_long_window > vol_short_window
        if self.vol_long_window <= self.vol_short_window:
            msg = (
                f"vol_long_window ({self.vol_long_window}) must be > "
                f"vol_short_window ({self.vol_short_window})"
            )
            raise ValueError(msg)

        # expansion_vol_ratio > contraction_vol_ratio
        if self.expansion_vol_ratio <= self.contraction_vol_ratio:
            msg = (
                f"expansion_vol_ratio ({self.expansion_vol_ratio}) must be > "
                f"contraction_vol_ratio ({self.contraction_vol_ratio})"
            )
            raise ValueError(msg)

        # expansion_mom_threshold > contraction_mom_threshold
        if self.expansion_mom_threshold <= self.contraction_mom_threshold:
            msg = (
                f"expansion_mom_threshold ({self.expansion_mom_threshold}) must be > "
                f"contraction_mom_threshold ({self.contraction_mom_threshold})"
            )
            raise ValueError(msg)

        # vol_target >= min_volatility
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
                self.vol_long_window,
                self.mom_window,
                self.atr_period,
            )
            + 1
        )
