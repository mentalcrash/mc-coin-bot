"""VW-TSMOM Strategy Configuration.

이 모듈은 VW-TSMOM 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
모든 파라미터는 타입 안전하게 검증됩니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TSMOMConfig(BaseModel):
    """VW-TSMOM 전략 설정.

    Volume-Weighted Time Series Momentum 전략의 모든 파라미터를 정의합니다.
    학술 연구(SSRN #4825389)에 기반한 기본값을 사용합니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Attributes:
        lookback: 모멘텀 계산 기간 (캔들 수, 보통 시간봉 기준)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성 (0.0~1.0, 예: 0.15 = 15%)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)
        annualization_factor: 연환산 계수 (시간봉 기준: 24*365 = 8760)
        use_log_returns: 로그 수익률 사용 여부
        momentum_smoothing: 모멘텀 스무딩 윈도우 (선택적)

    Example:
        >>> config = TSMOMConfig(
        ...     lookback=24,
        ...     vol_window=24,
        ...     vol_target=0.15,
        ... )
    """

    model_config = ConfigDict(frozen=True)  # 불변 객체

    # 모멘텀 계산 파라미터
    lookback: int = Field(
        default=30,  # 30일 (일봉)
        ge=6,
        le=365,  # 최대 1년 (일봉 기준)
        description="모멘텀 계산 기간 (캔들 수)",
    )

    # 변동성 파라미터
    vol_window: int = Field(
        default=30,  # 30일 (일봉)
        ge=6,
        le=365,
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

    # 시간 프레임 관련
    annualization_factor: float = Field(
        default=365.0,  # 일봉 기준
        gt=0,
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
    )

    # 옵션
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )
    momentum_smoothing: int | None = Field(
        default=None,
        ge=2,
        le=24,
        description="모멘텀 스무딩 윈도우 (선택적, EMA 적용)",
    )

    # 🆕 Z-Score 정규화 & 앙상블 옵션
    use_zscore: bool = Field(
        default=True,
        description="Z-Score 정규화 사용 여부 (신호 품질 향상)",
    )
    ensemble_windows: tuple[int, ...] = Field(
        default=(10, 20, 40),  # 일봉 기준: 10일, 20일, 40일
        description="앙상블 룩백 윈도우 (캔들 수). 여러 타임프레임 평균으로 휩쏘 감소.",
    )
    zscore_clip: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Z-Score 클리핑 범위 (-clip ~ +clip)",
    )

    # 🆕 Trend Filter & Deadband (휩쏘 방지)
    use_trend_filter: bool = Field(
        default=True,
        description="국면 필터 사용 여부 (상승장: Long Only, 하락장: Short Only)",
    )
    trend_ma_period: int = Field(
        default=50,
        ge=20,
        le=500,
        description="추세 판단용 이동평균 기간 (일봉 기준, 기본 50일)",
    )
    deadband_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.5,
        description="불감대 임계값 (|신호| < threshold면 중립 유지)",
    )

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        """윈도우 크기 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 윈도우 크기가 비합리적일 경우
        """
        # 모멘텀 스무딩이 lookback보다 크면 안 됨
        if (
            self.momentum_smoothing is not None
            and self.momentum_smoothing > self.lookback
        ):
            msg = (
                f"momentum_smoothing ({self.momentum_smoothing}) must be "
                f"<= lookback ({self.lookback})"
            )
            raise ValueError(msg)

        # vol_target이 min_volatility보다 크거나 같아야 함
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> "TSMOMConfig":
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "15m", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 TSMOMConfig

        Example:
            >>> config = TSMOMConfig.for_timeframe("1h", vol_target=0.20)
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

        # 타임프레임별 기본 lookback (대략 1일치)
        lookback_map: dict[str, int] = {
            "1m": 60,  # 1시간
            "5m": 48,  # 4시간
            "15m": 48,  # 12시간
            "1h": 24,  # 24시간
            "4h": 24,  # 4일
            "1d": 7,  # 1주일
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
    def conservative(cls) -> "TSMOMConfig":
        """보수적 설정 (긴 lookback, 낮은 변동성 타겟).

        Note:
            레버리지 제한은 PortfolioManagerConfig.conservative()를 함께 사용하세요.

        Returns:
            보수적 파라미터의 TSMOMConfig
        """
        return cls(
            lookback=48,
            vol_window=48,
            vol_target=0.10,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> "TSMOMConfig":
        """공격적 설정 (짧은 lookback, 높은 변동성 타겟).

        Note:
            레버리지 제한은 PortfolioManagerConfig.aggressive()를 함께 사용하세요.

        Returns:
            공격적 파라미터의 TSMOMConfig
        """
        return cls(
            lookback=12,
            vol_window=12,
            vol_target=0.20,
            min_volatility=0.05,
        )

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        전략 계산을 시작하기 전 필요한 최소 데이터 양입니다.
        Rolling 계산의 초기 NaN을 피하기 위해 사용됩니다.

        Returns:
            필요한 캔들 수
        """
        return max(self.lookback, self.vol_window) + 1
