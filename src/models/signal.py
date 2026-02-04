"""Signal DTO (Data Transfer Object) for strategy outputs.

이 모듈은 전략이 생성하는 개별 시그널을 표현하는 Pydantic 모델을 정의합니다.
백테스팅과 라이브 트레이딩 모두에서 사용됩니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for prices
    - #10 Python Standards: Modern typing (X | None, list[])
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from src.models.types import Direction, SignalType


class Signal(BaseModel):
    """단일 매매 시그널.

    전략 엔진이 생성하는 개별 시그널을 표현합니다.
    이 모델은 라이브 트레이딩에서 주문 생성의 기반이 됩니다.

    Attributes:
        timestamp: 시그널 생성 시각 (UTC)
        symbol: 거래 심볼 (예: "BTC/USDT")
        signal_type: 시그널 유형 (ENTRY_LONG, EXIT_SHORT 등)
        direction: 포지션 방향 (-1, 0, 1)
        strength: 시그널 강도 (-2.0 ~ 2.0, 레버리지 포함)
        entry_price: 예상 진입가 (시장가면 현재가)
        stop_loss: 손절가 (선택적)
        take_profit: 익절가 (선택적)
        strategy_name: 시그널 생성 전략 이름
        metadata: 추가 메타데이터 (지표 값 등)

    Example:
        >>> signal = Signal(
        ...     timestamp=datetime.now(UTC),
        ...     symbol="BTC/USDT",
        ...     signal_type=SignalType.ENTRY_LONG,
        ...     direction=Direction.LONG,
        ...     strength=1.5,
        ...     entry_price=Decimal("42000.00"),
        ...     strategy_name="VW-TSMOM",
        ... )
    """

    model_config = ConfigDict(frozen=True)  # 불변 객체

    timestamp: datetime = Field(..., description="시그널 생성 시각 (UTC)")
    symbol: str = Field(..., description="거래 심볼 (예: BTC/USDT)")
    signal_type: SignalType = Field(..., description="시그널 유형")
    direction: Direction = Field(..., description="포지션 방향 (-1, 0, 1)")
    strength: float = Field(
        ...,
        ge=-3.0,
        le=3.0,
        description="시그널 강도 (레버리지 스케일)",
    )
    entry_price: Decimal | None = Field(
        default=None,
        gt=0,
        description="예상 진입가",
    )
    stop_loss: Decimal | None = Field(
        default=None,
        gt=0,
        description="손절가",
    )
    take_profit: Decimal | None = Field(
        default=None,
        gt=0,
        description="익절가",
    )
    strategy_name: str = Field(..., description="시그널 생성 전략 이름")
    metadata: dict[str, float] = Field(
        default_factory=dict,
        description="추가 메타데이터 (지표 값 등)",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """timestamp에 UTC timezone 적용.

        Args:
            v: datetime 객체

        Returns:
            UTC timezone이 적용된 datetime
        """
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_entry(self) -> bool:
        """진입 시그널 여부."""
        return self.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_exit(self) -> bool:
        """청산 시그널 여부."""
        return self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_long(self) -> bool:
        """롱 방향 시그널 여부."""
        return self.direction == Direction.LONG

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_short(self) -> bool:
        """숏 방향 시그널 여부."""
        return self.direction == Direction.SHORT

    def with_stop_loss(self, stop_loss: Decimal) -> Self:
        """손절가가 설정된 새 시그널 반환 (불변 패턴).

        Args:
            stop_loss: 손절가

        Returns:
            새로운 Signal 인스턴스
        """
        return self.model_copy(update={"stop_loss": stop_loss})

    def with_take_profit(self, take_profit: Decimal) -> Self:
        """익절가가 설정된 새 시그널 반환 (불변 패턴).

        Args:
            take_profit: 익절가

        Returns:
            새로운 Signal 인스턴스
        """
        return self.model_copy(update={"take_profit": take_profit})


class SignalBatch(BaseModel):
    """시그널 배치 (여러 시그널 묶음).

    여러 시그널을 한 번에 처리할 때 사용합니다.

    Attributes:
        signals: 시그널 튜플 (불변)
        generated_at: 배치 생성 시각
    """

    model_config = ConfigDict(frozen=True)

    signals: tuple[Signal, ...] = Field(
        default_factory=tuple,
        description="시그널 튜플 (불변)",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="배치 생성 시각 (UTC)",
    )

    @property
    def count(self) -> int:
        """시그널 개수."""
        return len(self.signals)

    @property
    def is_empty(self) -> bool:
        """배치가 비어있는지 확인."""
        return len(self.signals) == 0

    def filter_by_symbol(self, symbol: str) -> "SignalBatch":
        """특정 심볼의 시그널만 필터링.

        Args:
            symbol: 거래 심볼

        Returns:
            필터링된 새로운 SignalBatch
        """
        filtered = tuple(s for s in self.signals if s.symbol == symbol)
        return SignalBatch(signals=filtered, generated_at=self.generated_at)

    def filter_entries(self) -> "SignalBatch":
        """진입 시그널만 필터링.

        Returns:
            진입 시그널만 포함된 새로운 SignalBatch
        """
        filtered = tuple(s for s in self.signals if s.is_entry)
        return SignalBatch(signals=filtered, generated_at=self.generated_at)
