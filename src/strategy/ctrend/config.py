"""CTREND Strategy Configuration.

ML Elastic Net Trend Factor 전략의 설정을 정의하는 Pydantic 모델입니다.
28개 기술적 지표를 Elastic Net 회귀로 결합하여 수익률을 예측합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class CTRENDConfig(BaseModel):
    """CTREND 전략 설정 (ML Elastic Net Trend Factor).

    28개 기술적 지표를 Rolling Elastic Net으로 결합하여
    forward return을 예측하고 시그널을 생성합니다.

    Attributes:
        training_window: Rolling training window (캔들 수)
        prediction_horizon: Forward return prediction 기간
        alpha: Elastic Net L1 ratio (0=Ridge, 1=Lasso)
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = CTRENDConfig(training_window=252, alpha=0.5)
    """

    model_config = ConfigDict(frozen=True)

    # ML 파라미터
    training_window: int = Field(
        default=252,
        ge=60,
        le=504,
        description="Rolling training window (캔들 수)",
    )
    prediction_horizon: int = Field(
        default=5,
        ge=1,
        le=21,
        description="Forward return prediction horizon",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Elastic Net L1 ratio (0=Ridge, 1=Lasso)",
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
        description="연환산 계수 (일봉: 365)",
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

        training_window + 50 (indicator warmup용 추가 기간).

        Returns:
            필요한 캔들 수
        """
        # 50 extra for indicator warmup (longest indicator period)
        return self.training_window + 50
