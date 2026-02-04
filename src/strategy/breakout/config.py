"""Adaptive Breakout Strategy Configuration.

이 모듈은 Adaptive Breakout 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
모든 파라미터는 타입 안전하게 검증됩니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdaptiveBreakoutConfig(BaseModel):
    """Adaptive Breakout 전략 설정.

    Donchian Channel 기반 돌파 전략의 모든 파라미터를 정의합니다.
    ATR(Average True Range)을 활용하여 변동성에 적응하는 임계값을 사용합니다.

    Key Concepts:
        - Donchian Channel: N일간 고가의 최고점, 저가의 최저점으로 형성된 채널
        - ATR: 변동성 측정 지표, 돌파 확인 및 포지션 사이징에 활용
        - Adaptive Threshold: ATR * k_value로 계산된 동적 임계값

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Attributes:
        channel_period: Donchian Channel 계산 기간 (캔들 수)
        atr_period: ATR 계산 기간 (캔들 수)
        k_value: ATR 배수 (돌파 확인 임계값)
        volatility_lookback: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성 (0.0~1.0, 예: 0.15 = 15%)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)
        annualization_factor: 연환산 계수 (일봉: 365, 시간봉: 8760)
        adaptive_threshold: 변동성 기반 동적 임계값 사용 여부
        long_only: Long-Only 모드 (Short 시그널 비활성화)

    Example:
        >>> config = AdaptiveBreakoutConfig(
        ...     channel_period=20,
        ...     k_value=1.5,
        ...     atr_period=14,
        ... )
    """

    model_config = ConfigDict(frozen=True)  # 불변 객체

    # =========================================================================
    # Donchian Channel 파라미터
    # =========================================================================
    channel_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Donchian Channel 계산 기간 (캔들 수)",
    )

    # =========================================================================
    # ATR (Average True Range) 파라미터
    # =========================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (캔들 수)",
    )
    k_value: float = Field(
        default=0.5,
        ge=0.5,
        le=5.0,
        description="ATR 배수 (돌파 확인 임계값, 높을수록 보수적). 암호화폐는 0.5~1.0 권장",
    )

    # =========================================================================
    # 변동성 스케일링 파라미터
    # =========================================================================
    volatility_lookback: int = Field(
        default=20,
        ge=10,
        le=100,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.0~1.0)",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프 (0으로 나누기 방지)",
    )

    # =========================================================================
    # 시간 프레임 관련
    # =========================================================================
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
    )

    # =========================================================================
    # 전략 옵션
    # =========================================================================
    adaptive_threshold: bool = Field(
        default=True,
        description="변동성 기반 동적 임계값 사용 여부",
    )
    long_only: bool = Field(
        default=False,
        description="Long-Only 모드 (Short 시그널을 비활성화)",
    )
    # NOTE: Trailing Stop은 PortfolioManagerConfig에서 관리합니다.
    # 포지션 관리(stop-loss, trailing stop)는 Portfolio 레이어의 책임입니다.

    # =========================================================================
    # 필터 옵션
    # =========================================================================
    cooldown_periods: int = Field(
        default=0,
        ge=0,
        le=10,
        description="진입 후 재진입 금지 기간 (캔들 수, 0이면 비활성화)",
    )
    min_breakout_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최소 돌파 강도 (밴드 대비 돌파 비율, 0이면 비활성화)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # vol_target이 min_volatility보다 크거나 같아야 함
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        # k_value가 너무 낮으면 false breakout 다수 발생
        if self.k_value < 1.0 and not self.adaptive_threshold:
            msg = (
                "k_value < 1.0 without adaptive_threshold may cause "
                "excessive false breakouts"
            )
            raise ValueError(msg)

        return self

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> AdaptiveBreakoutConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "15m", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 AdaptiveBreakoutConfig

        Example:
            >>> config = AdaptiveBreakoutConfig.for_timeframe("1h", k_value=2.0)
        """
        # 타임프레임별 연환산 계수
        annualization_map: dict[str, float] = {
            "1m": 525600.0,  # 60 * 24 * 365
            "5m": 105120.0,  # 12 * 24 * 365
            "15m": 35040.0,  # 4 * 24 * 365
            "1h": 8760.0,  # 24 * 365
            "4h": 2190.0,  # 6 * 365
            "1d": 365.0,
        }

        # 타임프레임별 기본 channel_period (대략 20일 기준)
        channel_map: dict[str, int] = {
            "1m": 30,  # 30분
            "5m": 60,  # 5시간
            "15m": 96,  # 24시간
            "1h": 24,  # 24시간 (1일)
            "4h": 30,  # 5일
            "1d": 20,  # 20일 (표준)
        }

        annualization = annualization_map.get(timeframe, 365.0)
        channel_period = channel_map.get(timeframe, 20)

        return cls(
            channel_period=channel_period,
            volatility_lookback=channel_period,
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> AdaptiveBreakoutConfig:
        """보수적 설정 (긴 기간, 높은 k_value).

        - 긴 channel_period로 노이즈 필터링
        - 높은 k_value로 확실한 돌파만 진입
        - 낮은 vol_target으로 작은 포지션

        Returns:
            보수적 파라미터의 AdaptiveBreakoutConfig
        """
        return cls(
            channel_period=30,
            atr_period=20,
            k_value=2.0,
            volatility_lookback=30,
            vol_target=0.30,
        )

    @classmethod
    def aggressive(cls) -> AdaptiveBreakoutConfig:
        """공격적 설정 (짧은 기간, 낮은 k_value).

        - 짧은 channel_period로 빠른 반응
        - 낮은 k_value로 더 많은 진입 기회
        - 높은 vol_target으로 큰 포지션

        Returns:
            공격적 파라미터의 AdaptiveBreakoutConfig
        """
        return cls(
            channel_period=10,
            atr_period=10,
            k_value=1.0,
            volatility_lookback=10,
            vol_target=0.50,
        )

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        전략 계산을 시작하기 전 필요한 최소 데이터 양입니다.
        Rolling 계산의 초기 NaN을 피하기 위해 사용됩니다.

        Returns:
            필요한 캔들 수
        """
        return max(self.channel_period, self.atr_period, self.volatility_lookback) + 1
