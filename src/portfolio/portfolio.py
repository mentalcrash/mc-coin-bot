"""Portfolio Domain Object.

이 모듈은 포트폴리오 도메인 객체를 정의합니다.
initial_capital과 PortfolioManagerConfig를 결합하여
백테스트, Dry Run, Live 트레이딩에서 동일하게 사용할 수 있습니다.

Design Principles:
    - Immutable: 생성 후 변경 불가
    - initial_capital은 Portfolio의 책임
    - Config는 집행 규칙만 담당

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, ConfigDict
    - #10 Python Standards: Modern typing, Self
"""

from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.portfolio.config import PortfolioManagerConfig


class Portfolio(BaseModel):
    """포트폴리오 도메인 객체.

    초기 자본과 집행 설정을 결합한 불변 객체입니다.
    백테스트, Dry Run, Live 트레이딩에서 동일하게 사용할 수 있습니다.

    Attributes:
        initial_capital: 초기 자본 (USD)
        config: 포트폴리오 집행 설정

    Example:
        >>> portfolio = Portfolio.create(initial_capital=10000)
        >>> portfolio = Portfolio.conservative(initial_capital=50000)
    """

    model_config = ConfigDict(frozen=True)

    initial_capital: Decimal = Field(
        ...,
        gt=0,
        description="초기 자본 (USD)",
    )
    config: PortfolioManagerConfig = Field(
        default_factory=PortfolioManagerConfig,
        description="포트폴리오 집행 설정",
    )

    @field_validator("initial_capital", mode="before")
    @classmethod
    def ensure_decimal(cls, v: Decimal | float | int | str) -> Decimal:
        """초기 자본을 Decimal로 변환.

        Args:
            v: 초기 자본 값 (다양한 타입 허용)

        Returns:
            Decimal 객체
        """
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))

    # =========================================================================
    # Factory Methods (팩토리 메서드)
    # =========================================================================
    @classmethod
    def create(
        cls,
        initial_capital: Decimal | float | int = Decimal(10000),
        config: PortfolioManagerConfig | None = None,
    ) -> Self:
        """기본 포트폴리오 생성.

        Args:
            initial_capital: 초기 자본 (기본값: $10,000)
            config: 포트폴리오 설정 (None이면 기본값)

        Returns:
            Portfolio 인스턴스

        Example:
            >>> portfolio = Portfolio.create(initial_capital=10000)
            >>> portfolio = Portfolio.create(
            ...     initial_capital=50000,
            ...     config=PortfolioManagerConfig(max_leverage_cap=3.0),
            ... )
        """
        return cls(
            initial_capital=Decimal(str(initial_capital)),
            config=config or PortfolioManagerConfig(),
        )

    @classmethod
    def conservative(
        cls,
        initial_capital: Decimal | float | int = Decimal(10000),
    ) -> Self:
        """보수적 포트폴리오 생성.

        낮은 레버리지와 타이트한 손절로 리스크를 최소화합니다.

        Args:
            initial_capital: 초기 자본

        Returns:
            보수적 설정이 적용된 Portfolio
        """
        return cls(
            initial_capital=Decimal(str(initial_capital)),
            config=PortfolioManagerConfig.conservative(),
        )

    @classmethod
    def aggressive(
        cls,
        initial_capital: Decimal | float | int = Decimal(10000),
    ) -> Self:
        """공격적 포트폴리오 생성.

        높은 레버리지와 관대한 손절로 수익을 극대화합니다.

        Args:
            initial_capital: 초기 자본

        Returns:
            공격적 설정이 적용된 Portfolio
        """
        return cls(
            initial_capital=Decimal(str(initial_capital)),
            config=PortfolioManagerConfig.aggressive(),
        )

    @classmethod
    def paper_trading(
        cls,
        initial_capital: Decimal | float | int = Decimal(10000),
    ) -> Self:
        """페이퍼 트레이딩 포트폴리오 생성.

        비용 없음, 관대한 레버리지로 전략 로직만 테스트합니다.

        Warning:
            실제 성과와 크게 다를 수 있습니다. 연구 목적으로만 사용하세요.

        Args:
            initial_capital: 초기 자본

        Returns:
            페이퍼 트레이딩 설정이 적용된 Portfolio
        """
        return cls(
            initial_capital=Decimal(str(initial_capital)),
            config=PortfolioManagerConfig.paper_trading(),
        )

    @classmethod
    def binance_vip(
        cls,
        initial_capital: Decimal | float | int = Decimal(10000),
        vip_level: int = 1,
    ) -> Self:
        """바이낸스 VIP 등급별 포트폴리오 생성.

        Args:
            initial_capital: 초기 자본
            vip_level: VIP 등급 (0-9)

        Returns:
            VIP 등급에 맞는 Portfolio
        """
        return cls(
            initial_capital=Decimal(str(initial_capital)),
            config=PortfolioManagerConfig.binance_vip(vip_level),
        )

    # =========================================================================
    # Utility Methods (유틸리티 메서드)
    # =========================================================================
    @property
    def initial_capital_float(self) -> float:
        """초기 자본을 float로 반환 (VectorBT 호환).

        Returns:
            초기 자본 (float)
        """
        return float(self.initial_capital)

    def summary(self) -> dict[str, object]:
        """포트폴리오 요약 정보 반환.

        Returns:
            핵심 정보가 포함된 딕셔너리
        """
        return {
            "initial_capital": f"${self.initial_capital:,.2f}",
            **self.config.summary(),
        }

    def __repr__(self) -> str:
        """문자열 표현."""
        return (
            f"Portfolio("
            f"capital=${self.initial_capital:,.0f}, "
            f"leverage_cap={self.config.max_leverage_cap}x)"
        )
