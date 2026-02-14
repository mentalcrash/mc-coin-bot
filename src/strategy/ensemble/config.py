"""Ensemble Strategy Configuration.

여러 전략의 시그널을 집계(Aggregation)하여 앙상블 포트폴리오를 구성하는 설정.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing, StrEnum
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AggregationMethod(StrEnum):
    """시그널 집계 방법.

    Attributes:
        EQUAL_WEIGHT: 동일 가중 평균
        INVERSE_VOLATILITY: 변동성 역비례 가중
        MAJORITY_VOTE: 다수결 합의
        STRATEGY_MOMENTUM: 최근 성과 기반 모멘텀 가중
    """

    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    MAJORITY_VOTE = "majority_vote"
    STRATEGY_MOMENTUM = "strategy_momentum"


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 -> 중립)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    FULL = 2


class SubStrategySpec(BaseModel):
    """서브 전략 사양.

    앙상블에 포함할 개별 전략의 이름, 파라미터, 정적 가중치를 정의합니다.

    Attributes:
        name: Registry에 등록된 전략 이름 (예: "tsmom", "donchian-ensemble")
        params: 전략 생성 파라미터
        weight: 정적 가중치 (equal_weight 이외 method에서 초기값)
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Registry에 등록된 전략 이름")
    params: dict[str, Any] = Field(default_factory=dict, description="전략 생성 파라미터")
    weight: float = Field(default=1.0, gt=0.0, description="정적 가중치")


class EnsembleConfig(BaseModel):
    """앙상블 전략 설정.

    여러 서브 전략의 시그널을 집계하여 하나의 포지션을 결정합니다.

    Attributes:
        strategies: 서브 전략 목록 (최소 2개)
        aggregation: 시그널 집계 방법
        vol_lookback: inverse_volatility 방법의 변동성 lookback
        momentum_lookback: strategy_momentum 방법의 모멘텀 lookback
        top_n: strategy_momentum에서 선택할 상위 전략 수
        min_agreement: majority_vote에서 최소 합의 비율
        vol_target: 연간 목표 변동성
        vol_window: 변동성 계산 윈도우
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드
    """

    model_config = ConfigDict(frozen=True)

    strategies: tuple[SubStrategySpec, ...] = Field(
        min_length=2,
        description="서브 전략 목록 (최소 2개)",
    )
    aggregation: AggregationMethod = Field(
        default=AggregationMethod.EQUAL_WEIGHT,
        description="시그널 집계 방법",
    )

    # Inverse Volatility 파라미터
    vol_lookback: int = Field(
        default=63,
        ge=5,
        le=504,
        description="inverse_volatility 방법의 변동성 lookback",
    )

    # Strategy Momentum 파라미터
    momentum_lookback: int = Field(
        default=126,
        ge=10,
        le=504,
        description="strategy_momentum 방법의 모멘텀 lookback",
    )
    top_n: int = Field(
        default=3,
        ge=1,
        description="strategy_momentum에서 선택할 상위 전략 수",
    )

    # Majority Vote 파라미터
    min_agreement: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="majority_vote에서 최소 합의 비율",
    )

    # Volatility Scaling
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성",
    )
    vol_window: int = Field(
        default=30,
        ge=5,
        le=252,
        description="변동성 계산 윈도우",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
    )

    # Short Mode
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드 (DISABLED/FULL)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: top_n > 전략 수이거나 vol_target < 0.05일 경우
        """
        if (
            self.aggregation == AggregationMethod.STRATEGY_MOMENTUM
            and self.top_n > len(self.strategies)
        ):
            msg = (
                f"top_n ({self.top_n}) cannot exceed number of strategies ({len(self.strategies)})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간.

        aggregation lookback + 여유분 (sub-strategy warmup은 별도 합산).

        Returns:
            aggregation에 필요한 최소 캔들 수
        """
        agg_lookback = 0
        if self.aggregation == AggregationMethod.INVERSE_VOLATILITY:
            agg_lookback = self.vol_lookback
        elif self.aggregation == AggregationMethod.STRATEGY_MOMENTUM:
            agg_lookback = self.momentum_lookback
        return agg_lookback + self.vol_window + 1
