"""Vol-Regime Adaptive Strategy Configuration.

이 모듈은 Vol-Regime Adaptive 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
변동성 regime별 파라미터 자동 전환을 위한 모든 설정이 포함됩니다.

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


class VolRegimeConfig(BaseModel):
    """Vol-Regime Adaptive 전략 설정.

    변동성 regime(high/normal/low)별 TSMOM 파라미터를 자동 전환하여
    시장 상황에 적응하는 전략의 모든 파라미터를 정의합니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. vol_pct = rolling percentile rank of realized volatility
        2. regime_strength = select(high/normal/low momentum * vol_scalar)
        3. direction = sign(regime_strength)
        4. strength = regime_strength (변동성 스케일링 포함)

    Attributes:
        vol_lookback: 변동성 추정 윈도우
        vol_rank_lookback: percentile rank 계산 윈도우
        high_vol_threshold: 고변동성 regime 임계값 (초과 시 high vol)
        low_vol_threshold: 저변동성 regime 임계값 (미만 시 low vol)
        high_vol_lookback: 고변동성 regime TSMOM lookback
        high_vol_target: 고변동성 regime 목표 변동성
        normal_lookback: 일반 regime TSMOM lookback
        normal_vol_target: 일반 regime 목표 변동성
        low_vol_lookback: 저변동성 regime TSMOM lookback
        low_vol_target: 저변동성 regime 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율

    Example:
        >>> config = VolRegimeConfig(
        ...     high_vol_threshold=0.8,
        ...     low_vol_threshold=0.2,
        ...     high_vol_lookback=60,
        ...     normal_lookback=30,
        ...     low_vol_lookback=14,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Volatility Regime 파라미터
    # =========================================================================
    vol_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 추정 윈도우 (캔들 수)",
    )
    vol_rank_lookback: int = Field(
        default=252,
        ge=60,
        le=500,
        description="Percentile rank 계산 윈도우 (캔들 수)",
    )
    high_vol_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="고변동성 regime 임계값 (이 이상이면 high vol)",
    )
    low_vol_threshold: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="저변동성 regime 임계값 (이 이하면 low vol)",
    )

    # =========================================================================
    # Regime별 TSMOM 파라미터
    # =========================================================================
    # High vol regime: 보수적 (긴 lookback, 낮은 vol target)
    high_vol_lookback: int = Field(
        default=60,
        ge=10,
        le=120,
        description="고변동성 regime TSMOM lookback (캔들 수)",
    )
    high_vol_target: float = Field(
        default=0.15,
        ge=0.05,
        le=0.50,
        description="고변동성 regime 목표 변동성",
    )

    # Normal regime: 중간
    normal_lookback: int = Field(
        default=30,
        ge=10,
        le=90,
        description="일반 regime TSMOM lookback (캔들 수)",
    )
    normal_vol_target: float = Field(
        default=0.30,
        ge=0.05,
        le=1.0,
        description="일반 regime 목표 변동성",
    )

    # Low vol regime: 공격적 (짧은 lookback, 높은 vol target)
    low_vol_lookback: int = Field(
        default=14,
        ge=6,
        le=60,
        description="저변동성 regime TSMOM lookback (캔들 수)",
    )
    low_vol_target: float = Field(
        default=0.50,
        ge=0.10,
        le=1.0,
        description="저변동성 regime 목표 변동성",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
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
        # high_vol_threshold > low_vol_threshold
        if self.high_vol_threshold <= self.low_vol_threshold:
            msg = (
                f"high_vol_threshold ({self.high_vol_threshold}) must be > "
                f"low_vol_threshold ({self.low_vol_threshold})"
            )
            raise ValueError(msg)

        # 모든 vol_target >= min_volatility
        for target_name, target_val in [
            ("high_vol_target", self.high_vol_target),
            ("normal_vol_target", self.normal_vol_target),
            ("low_vol_target", self.low_vol_target),
        ]:
            if target_val < self.min_volatility:
                msg = (
                    f"{target_name} ({target_val}) should be >= "
                    f"min_volatility ({self.min_volatility})"
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
                self.vol_rank_lookback,
                self.high_vol_lookback,
                self.normal_lookback,
                self.atr_period,
            )
            + 1
        )
