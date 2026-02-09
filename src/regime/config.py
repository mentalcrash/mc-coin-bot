"""Regime Detector Configuration.

레짐 감지에 사용되는 설정 모델과 라벨 정의.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing, StrEnum
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegimeLabel(StrEnum):
    """시장 레짐 라벨.

    Attributes:
        TRENDING: 추세장 (ER 높음, 방향성 강함)
        RANGING: 횡보장 (ER 낮음, 변동성 낮음)
        VOLATILE: 고변동장 (RV 높음, ER 낮음)
    """

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


class RegimeDetectorConfig(BaseModel):
    """레짐 감지 설정.

    RV Ratio와 Efficiency Ratio를 기반으로 시장 레짐을 분류합니다.

    Attributes:
        rv_short_window: 단기 RV 윈도우 (bar 수)
        rv_long_window: 장기 RV 윈도우 (bar 수)
        er_window: Efficiency Ratio 윈도우 (bar 수)
        er_trending_threshold: ER 추세 판별 임계값
        rv_expansion_threshold: RV 확장 판별 임계값
        min_hold_bars: 최소 레짐 유지 bar 수 (hysteresis)

    Example:
        >>> config = RegimeDetectorConfig(
        ...     rv_short_window=5,
        ...     rv_long_window=20,
        ...     er_window=10,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    rv_short_window: int = Field(
        default=5,
        ge=2,
        le=50,
        description="단기 RV 윈도우 (bar 수)",
    )
    rv_long_window: int = Field(
        default=20,
        ge=5,
        le=200,
        description="장기 RV 윈도우 (bar 수)",
    )
    er_window: int = Field(
        default=10,
        ge=3,
        le=100,
        description="Efficiency Ratio 윈도우 (bar 수)",
    )
    er_trending_threshold: float = Field(
        default=0.6,
        ge=0.1,
        le=0.95,
        description="ER 추세 판별 임계값 (높을수록 엄격)",
    )
    rv_expansion_threshold: float = Field(
        default=1.2,
        ge=1.0,
        le=3.0,
        description="RV 확장 판별 임계값 (rv_ratio > 이 값이면 변동 확대)",
    )
    min_hold_bars: int = Field(
        default=3,
        ge=1,
        le=20,
        description="최소 레짐 유지 bar 수 (hysteresis)",
    )

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        """윈도우 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: rv_short_window >= rv_long_window일 경우
        """
        if self.rv_short_window >= self.rv_long_window:
            msg = (
                f"rv_short_window ({self.rv_short_window}) must be "
                f"< rv_long_window ({self.rv_long_window})"
            )
            raise ValueError(msg)
        return self
