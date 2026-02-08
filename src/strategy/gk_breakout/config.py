"""GK Volatility Breakout Strategy Configuration.

Garman-Klass 변동성 압축 후 Donchian 채널 돌파 전략 설정.
GK variance로 변동성 압축 구간을 감지하고, Donchian Channel 돌파 시 진입합니다.

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


class GKBreakoutConfig(BaseModel):
    """GK Volatility Breakout 전략 설정.

    Garman-Klass variance를 활용하여 변동성 압축 구간을 감지하고,
    Donchian Channel 돌파 시 진입하는 전략입니다.

    Key Concepts:
        - GK Variance: OHLC 4가지 가격을 모두 활용하는 효율적 변동성 추정치
        - Vol Ratio: 단기/장기 GK variance 비율 (1 미만 = 압축)
        - Donchian Channel: N일간 고가 최고점/저가 최저점으로 형성된 채널

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Attributes:
        gk_lookback: GK variance rolling window (캔들 수)
        compression_threshold: vol ratio 압축 임계값 (1 미만)
        breakout_lookback: Donchian Channel lookback (캔들 수)
        atr_period: ATR 계산 기간
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율

    Example:
        >>> config = GKBreakoutConfig(
        ...     gk_lookback=20,
        ...     compression_threshold=0.75,
        ...     breakout_lookback=20,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # GK Variance 파라미터
    # =========================================================================
    gk_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="GK variance rolling window (캔들 수)",
    )
    compression_threshold: float = Field(
        default=0.75,
        ge=0.3,
        le=1.0,
        description="vol ratio 압축 임계값 (단기/장기 비율이 이 미만이면 압축)",
    )

    # =========================================================================
    # Donchian Channel 파라미터
    # =========================================================================
    breakout_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Donchian Channel lookback (캔들 수)",
    )

    # =========================================================================
    # ATR 파라미터
    # =========================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (trailing stop 참조용)",
    )

    # =========================================================================
    # 변동성 스케일링 파라미터
    # =========================================================================
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
        description="연간 목표 변동성 (0.30 = 30%)",
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
    # 옵션
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )

    # =========================================================================
    # Short Mode
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
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        GK variance의 long window (gk_lookback * 2)가 가장 긴 윈도우입니다.

        Returns:
            필요한 캔들 수
        """
        return (
            max(self.gk_lookback * 2, self.breakout_lookback, self.vol_window, self.atr_period) + 1
        )
