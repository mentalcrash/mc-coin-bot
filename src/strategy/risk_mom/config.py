"""Risk-Managed Momentum Strategy Configuration.

Barroso-Santa-Clara (2015) variance scaling을 적용한 모멘텀 전략 설정입니다.
TSMOM의 vol_scalar 대신 realized variance 기반 BSC scaling을 사용합니다.

References:
    Barroso, P. & Santa-Clara, P. (2015).
    "Momentum has its moments." Journal of Financial Economics.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class RiskMomConfig(BaseModel):
    """Risk-Managed Momentum 전략 설정.

    TSMOM + BSC variance scaling 전략의 파라미터를 정의합니다.
    vol_scalar(= vol_target / realized_vol) 대신
    bsc_scaling(= vol_target^2 / realized_var)으로 포지션 크기를 조절합니다.

    Signal Formula:
        1. vw_momentum = 거래량 가중 수익률 (lookback 기간)
        2. realized_var = returns.rolling(var_window).var()
        3. bsc_scaling = vol_target^2 / max(realized_var, min_variance)
        4. direction = sign(vw_momentum)
        5. strength = direction * bsc_scaling

    Attributes:
        lookback: 모멘텀 계산 기간 (캔들 수)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        var_window: 분산 계산 윈도우 (BSC, 약 6개월)
        vol_target: 연간 목표 변동성 (0.0~1.0)
        min_volatility: 최소 변동성 클램프
        min_variance: 최소 분산 클램프 (BSC 0 나누기 방지)
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율

    Example:
        >>> config = RiskMomConfig(
        ...     lookback=30,
        ...     var_window=126,
        ...     vol_target=0.30,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # 모멘텀 계산 파라미터
    lookback: int = Field(
        default=30,
        ge=6,
        le=365,
        description="모멘텀 계산 기간 (캔들 수)",
    )

    # 변동성 파라미터
    vol_window: int = Field(
        default=30,
        ge=6,
        le=365,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    var_window: int = Field(
        default=126,
        ge=60,
        le=365,
        description="분산 계산 윈도우 (BSC, 약 6개월)",
    )
    vol_target: float = Field(
        default=0.30,
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
    min_variance: float = Field(
        default=0.0001,
        ge=0.00001,
        le=0.01,
        description="최소 분산 클램프 (BSC 0 나누기 방지)",
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
            ValueError: vol_target < min_volatility일 경우
        """
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        var_window이 가장 긴 rolling 윈도우이므로 이를 기준으로 계산합니다.

        Returns:
            필요한 캔들 수
        """
        return max(self.lookback, self.var_window, self.vol_window) + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> RiskMomConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "15m", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 RiskMomConfig
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        lookback_map: dict[str, int] = {
            "1m": 60,
            "5m": 48,
            "15m": 48,
            "1h": 24,
            "4h": 24,
            "1d": 7,
        }

        annualization = annualization_map.get(timeframe, 8760.0)
        lookback = lookback_map.get(timeframe, 24)

        return cls(
            lookback=lookback,
            vol_window=lookback,
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> RiskMomConfig:
        """보수적 설정 (긴 lookback, 낮은 변동성 타겟).

        Returns:
            보수적 파라미터의 RiskMomConfig
        """
        return cls(
            lookback=48,
            vol_window=48,
            var_window=180,
            vol_target=0.10,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> RiskMomConfig:
        """공격적 설정 (짧은 lookback, 높은 변동성 타겟).

        Returns:
            공격적 파라미터의 RiskMomConfig
        """
        return cls(
            lookback=12,
            vol_window=12,
            var_window=63,
            vol_target=0.20,
            min_volatility=0.05,
        )
