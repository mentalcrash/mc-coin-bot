"""Cost Model for Portfolio Management.

이 모듈은 포트폴리오 관리에서 사용되는 거래 비용 모델을 정의합니다.
수수료, 슬리피지, 펀딩비 등을 포함한 현실적인 비용을 모델링합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, validation
    - #25 QuantStats Standards: Realistic cost assumptions
    - #01 Project Structure: CostModel은 Portfolio 레이어의 책임
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CostModel(BaseModel):
    """거래 비용 모델.

    백테스팅에서 적용할 모든 거래 비용을 정의합니다.
    현실적인 백테스트 결과를 위해 보수적인 기본값을 사용합니다.

    Attributes:
        maker_fee: 메이커 수수료 (지정가 주문)
        taker_fee: 테이커 수수료 (시장가 주문)
        slippage: 슬리피지 (시장 충격)
        funding_rate_8h: 8시간당 펀딩비 (선물)
        market_impact: 시장 충격 비용 (대량 주문)
        use_taker: 테이커 수수료 사용 여부 (기본 True)

    Example:
        >>> model = CostModel(
        ...     maker_fee=0.0002,
        ...     taker_fee=0.0004,
        ...     slippage=0.0005,
        ... )
        >>> total_cost = model.total_fee_rate
    """

    model_config = ConfigDict(frozen=True)

    # 수수료
    maker_fee: float = Field(
        default=0.0002,
        ge=0,
        le=0.01,
        description="메이커 수수료 (예: 0.0002 = 0.02%)",
    )
    taker_fee: float = Field(
        default=0.0004,
        ge=0,
        le=0.01,
        description="테이커 수수료 (예: 0.0004 = 0.04%)",
    )

    # 슬리피지
    slippage: float = Field(
        default=0.0005,
        ge=0,
        le=0.02,
        description="슬리피지 (예: 0.0005 = 0.05%)",
    )

    # 펀딩비 (선물)
    funding_rate_8h: float = Field(
        default=0.0001,
        ge=-0.01,
        le=0.01,
        description="8시간당 펀딩비 (예: 0.0001 = 0.01%)",
    )

    # 시장 충격 (대량 주문)
    market_impact: float = Field(
        default=0.0002,
        ge=0,
        le=0.01,
        description="시장 충격 비용",
    )

    # 옵션
    use_taker: bool = Field(
        default=True,
        description="테이커 수수료 사용 여부 (시장가 주문 가정)",
    )

    @model_validator(mode="after")
    def validate_fees(self) -> Self:
        """수수료 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 비용이 비합리적으로 높을 경우
        """
        max_reasonable_cost = 0.05  # 5% 이상이면 경고
        total = self.total_fee_rate
        if total > max_reasonable_cost:
            msg = f"Total cost rate ({total:.2%}) seems unreasonably high"
            raise ValueError(msg)
        return self

    @property
    def effective_fee(self) -> float:
        """실효 수수료율 (메이커 또는 테이커).

        Returns:
            적용될 수수료율
        """
        return self.taker_fee if self.use_taker else self.maker_fee

    @property
    def slip_rate(self) -> float:
        """슬리피지 + 시장충격 합산 (가격 악화용).

        Returns:
            편도 가격 악화율
        """
        return self.slippage + self.market_impact

    @property
    def total_fee_rate(self) -> float:
        """총 비용률 (수수료 + 슬리피지 + 시장 충격).

        진입과 청산 시 각각 적용되므로 실제 왕복 비용은 2배입니다.

        Returns:
            편도 총 비용률
        """
        return self.effective_fee + self.slippage + self.market_impact

    @property
    def round_trip_cost(self) -> float:
        """왕복 거래 비용률.

        진입 + 청산 비용을 합산합니다.

        Returns:
            왕복 비용률 (예: 0.002 = 0.2%)
        """
        return self.total_fee_rate * 2

    def daily_funding_cost(self) -> float:
        """일일 펀딩비.

        8시간 펀딩비 x 3 (하루 3번)

        Returns:
            일일 펀딩비
        """
        return self.funding_rate_8h * 3

    def annual_funding_cost(self) -> float:
        """연간 펀딩비 (단순 계산).

        Returns:
            연간 펀딩비
        """
        return self.daily_funding_cost() * 365

    @classmethod
    def zero(cls) -> "CostModel":
        """비용 없는 모델 (이상적 환경 테스트용).

        Warning:
            실제 성과와 크게 다를 수 있습니다. 연구 목적으로만 사용하세요.

        Returns:
            모든 비용이 0인 CostModel
        """
        return cls(
            maker_fee=0,
            taker_fee=0,
            slippage=0,
            funding_rate_8h=0,
            market_impact=0,
        )

    @classmethod
    def conservative(cls) -> "CostModel":
        """보수적 비용 모델.

        실제 거래에서 발생할 수 있는 높은 비용을 가정합니다.
        과대적합(Overfitting) 방지에 유용합니다.

        Returns:
            보수적 비용의 CostModel
        """
        return cls(
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage=0.001,  # 0.1%
            funding_rate_8h=0.0003,  # 0.03%
            market_impact=0.0005,
        )

    @classmethod
    def optimistic(cls) -> "CostModel":
        """낙관적 비용 모델.

        VIP 등급, 높은 유동성 환경을 가정합니다.

        Returns:
            낙관적 비용의 CostModel
        """
        return cls(
            maker_fee=0.0001,
            taker_fee=0.0003,
            slippage=0.0003,
            funding_rate_8h=0.00005,
            market_impact=0.0001,
        )

    @classmethod
    def binance_spot(cls) -> "CostModel":
        """바이낸스 현물 기본 비용.

        VIP 0 기준.

        Returns:
            바이낸스 현물 CostModel
        """
        return cls(
            maker_fee=0.001,  # 0.1%
            taker_fee=0.001,  # 0.1%
            slippage=0.0005,
            funding_rate_8h=0,  # 현물은 펀딩비 없음
            market_impact=0.0002,
        )

    @classmethod
    def binance_futures(cls) -> "CostModel":
        """바이낸스 선물 기본 비용.

        VIP 0 기준.

        Returns:
            바이낸스 선물 CostModel
        """
        return cls(
            maker_fee=0.0002,  # 0.02%
            taker_fee=0.0004,  # 0.04%
            slippage=0.0005,
            funding_rate_8h=0.0001,  # 평균 0.01%
            market_impact=0.0002,
        )

    @classmethod
    def binance_vip1(cls) -> "CostModel":
        """바이낸스 VIP 1 선물 비용.

        Returns:
            VIP 1 CostModel
        """
        return cls(
            maker_fee=0.00016,
            taker_fee=0.0004,
            slippage=0.0005,
            funding_rate_8h=0.0001,
            market_impact=0.0002,
        )

    def to_vbt_params(self) -> dict[str, float]:
        """VectorBT 파라미터로 변환.

        VectorBT Portfolio.from_signals()에 전달할 파라미터를 반환합니다.

        Returns:
            VectorBT 호환 파라미터 딕셔너리

        Example:
            >>> cost = CostModel.binance_futures()
            >>> pf = vbt.Portfolio.from_signals(..., **cost.to_vbt_params())
        """
        return {
            "fees": self.effective_fee,
            "slippage": self.slippage + self.market_impact,
        }

    def estimate_breakeven_return(self, trades_per_day: float = 1.0) -> float:
        """손익분기점 수익률 추정.

        주어진 거래 빈도에서 비용을 커버하기 위한 최소 수익률을 계산합니다.

        Args:
            trades_per_day: 일평균 거래 횟수

        Returns:
            일일 손익분기점 수익률

        Example:
            >>> cost = CostModel.binance_futures()
            >>> breakeven = cost.estimate_breakeven_return(trades_per_day=2)
            >>> print(f"Daily breakeven: {breakeven:.2%}")
        """
        daily_trading_cost = self.round_trip_cost * trades_per_day
        daily_funding = self.daily_funding_cost()
        return daily_trading_cost + daily_funding
