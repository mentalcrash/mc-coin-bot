"""XSMOM Strategy Configuration.

이 모듈은 XSMOM 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
모든 파라미터는 타입 안전하게 검증됩니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class XSMOMConfig(BaseModel):
    """XSMOM (Cross-Sectional Momentum) 전략 설정.

    코인별 rolling return과 vol-target sizing을 사용하는 횡단면 모멘텀 전략입니다.
    Cross-sectional ranking은 멀티에셋 백테스트 레벨에서 수행됩니다.

    Signal Formula:
        1. rolling_return = lookback 기간 수익률 (log or simple)
        2. vol_scalar = vol_target / realized_vol
        3. direction = sign(rolling_return)
        4. strength = direction * vol_scalar
        5. holding_period로 시그널 리밸런싱 주기 제어

    Attributes:
        lookback: Rolling return 계산 기간 (캔들 수)
        holding_period: 시그널 유지 기간 (리밸런싱 주기)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성 (0.0~1.0)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)
        annualization_factor: 연환산 계수 (일봉: 365)
        use_log_returns: 로그 수익률 사용 여부
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = XSMOMConfig(
        ...     lookback=21,
        ...     holding_period=7,
        ...     vol_target=0.35,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # 모멘텀 계산 파라미터
    lookback: int = Field(
        default=21,
        ge=5,
        le=120,
        description="Rolling return lookback 기간 (캔들 수)",
    )
    holding_period: int = Field(
        default=7,
        ge=1,
        le=30,
        description="시그널 유지 기간 (리밸런싱 주기)",
    )

    # 변동성 파라미터
    vol_window: int = Field(
        default=30,
        ge=5,
        le=120,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.35,
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

    # 시간 프레임 관련
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
    )

    # 옵션
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )

    # 숏 모드 설정
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Rolling 계산의 초기 NaN을 피하기 위해 사용됩니다.

        Returns:
            필요한 캔들 수
        """
        return max(self.lookback, self.vol_window) + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> XSMOMConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 XSMOMConfig

        Example:
            >>> config = XSMOMConfig.for_timeframe("1h", vol_target=0.20)
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "2h": 4380.0,
            "3h": 2920.0,
            "4h": 2190.0,
            "6h": 1460.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 365.0)

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> XSMOMConfig:
        """보수적 설정 (긴 lookback, 낮은 변동성 타겟).

        Returns:
            보수적 파라미터의 XSMOMConfig
        """
        return cls(
            lookback=60,
            holding_period=14,
            vol_window=60,
            vol_target=0.15,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> XSMOMConfig:
        """공격적 설정 (짧은 lookback, 높은 변동성 타겟).

        Returns:
            공격적 파라미터의 XSMOMConfig
        """
        return cls(
            lookback=10,
            holding_period=3,
            vol_window=15,
            vol_target=0.50,
            min_volatility=0.05,
        )
