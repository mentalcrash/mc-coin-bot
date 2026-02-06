"""VW-TSMOM Strategy Configuration.

이 모듈은 VW-TSMOM 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
모든 파라미터는 타입 안전하게 검증됩니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 → 중립)
        HEDGE_ONLY: 헤지 목적 숏만 (드로다운 임계값 초과 시)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class TSMOMConfig(BaseModel):
    """VW-TSMOM 전략 설정 (Pure TSMOM + Vol Target).

    Volume-Weighted Time Series Momentum 전략의 핵심 파라미터만 정의합니다.
    학술 연구(SSRN #4825389)에 기반한 순수한 TSMOM 구현입니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. vw_momentum = 거래량 가중 수익률 (lookback 기간)
        2. vol_scalar = vol_target / realized_vol
        3. direction = sign(vw_momentum)
        4. strength = direction * vol_scalar

    Attributes:
        lookback: 모멘텀 계산 기간 (캔들 수)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성 (0.0~1.0, 예: 0.40 = 40%)
        min_volatility: 최소 변동성 클램프 (0으로 나누기 방지)
        annualization_factor: 연환산 계수 (일봉: 365)
        use_log_returns: 로그 수익률 사용 여부
        momentum_smoothing: 모멘텀 스무딩 윈도우 (선택적)

    Example:
        >>> config = TSMOMConfig(
        ...     lookback=30,
        ...     vol_window=30,
        ...     vol_target=0.40,
        ... )
    """

    model_config = ConfigDict(frozen=True)  # 불변 객체

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
    momentum_smoothing: int | None = Field(
        default=None,
        ge=2,
        le=24,
        description="모멘텀 스무딩 윈도우 (선택적, EMA 적용)",
    )

    # 숏 모드 설정 (long_only 대체)
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

    # 횡보장 필터 (ADX 기반)
    use_sideways_filter: bool = Field(
        default=False,
        description="횡보장 필터 활성화 (ADX 기반)",
    )
    adx_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ADX 계산 기간",
    )
    adx_threshold: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="ADX 임계값 (이 이하면 횡보장으로 판단)",
    )
    sideways_position_scale: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="횡보장에서 포지션 스케일 (0=현금, 1=유지)",
    )

    # 다중 타임프레임 필터 (Phase 2)
    mtf_filter: "MTFFilterConfig | None" = Field(
        default=None,
        description="다중 타임프레임 필터 설정 (None=비활성)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # 모멘텀 스무딩이 lookback보다 크면 안 됨
        if self.momentum_smoothing is not None and self.momentum_smoothing > self.lookback:
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

    def effective_short_mode(self) -> ShortMode:
        """실제 적용될 숏 모드 반환.

        Returns:
            적용될 ShortMode
        """
        return self.short_mode

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


class MTFFilterMode(StrEnum):
    """다중 타임프레임 필터 모드.

    Attributes:
        CONSENSUS: 상위 TF와 동일 방향일 때만 진입 허용 (가장 보수적)
        VETO: 상위 TF가 반대 방향일 때만 필터링 (중립은 통과)
        WEIGHTED: 상위 TF 방향에 따라 강도 조절 (가장 유연)
    """

    CONSENSUS = "consensus"
    VETO = "veto"
    WEIGHTED = "weighted"


class MTFFilterConfig(BaseModel):
    """다중 타임프레임 필터 설정.

    상위 타임프레임의 추세 방향을 참조하여
    하위 타임프레임 시그널을 필터링합니다.

    Example:
        >>> config = MTFFilterConfig(
        ...     enabled=True,
        ...     higher_timeframe="1W",
        ...     mode=MTFFilterMode.CONSENSUS,
        ... )

    Attributes:
        enabled: MTF 필터 활성화 여부
        higher_timeframe: 상위 타임프레임 (예: "1W" for Weekly)
        mode: 필터 모드 (CONSENSUS/VETO/WEIGHTED)
        lookback_htf: 상위 TF 모멘텀 계산 lookback (캔들 수)
        weight_aligned: 동일 방향 시 가중치 (WEIGHTED 모드)
        weight_neutral: 상위 TF 중립 시 가중치 (WEIGHTED 모드)
        weight_against: 반대 방향 시 가중치 (WEIGHTED 모드, 0=필터링)
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="MTF 필터 활성화 여부",
    )
    higher_timeframe: str = Field(
        default="1W",
        description="상위 타임프레임 (1D→1W, 4h→1D)",
    )
    mode: MTFFilterMode = Field(
        default=MTFFilterMode.CONSENSUS,
        description="필터 모드 (CONSENSUS/VETO/WEIGHTED)",
    )
    lookback_htf: int = Field(
        default=4,
        ge=2,
        le=12,
        description="상위 TF 모멘텀 lookback (주간 기준 약 1개월)",
    )

    # WEIGHTED 모드용 가중치
    weight_aligned: float = Field(
        default=1.2,
        ge=0.5,
        le=2.0,
        description="동일 방향 시 강도 배수",
    )
    weight_neutral: float = Field(
        default=1.0,
        ge=0.0,
        le=1.5,
        description="상위 TF 중립 시 강도 유지",
    )
    weight_against: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="반대 방향 시 강도 (0=필터링)",
    )
