"""Donchian Channel Breakout Strategy Configuration.

터틀 트레이딩 기반 Donchian Channel 전략 설정.
Entry Channel (진입)과 Exit Channel (청산)을 분리합니다.

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
        DISABLED: Long-Only 모드 (숏 시그널 → 중립)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    FULL = 2


class DonchianConfig(BaseModel):
    """Donchian Channel Breakout 전략 설정.

    터틀 트레이딩 규칙을 따르며, Entry/Exit 채널을 분리합니다.

    Turtle Rules:
        - Entry: N일 최고가/최저가 돌파 시 진입
        - Exit: M일 반대 채널 터치 시 청산 (N > M)

    Position Sizing:
        - ATR 기반 변동성 스케일링
        - strength = vol_target / realized_vol

    Attributes:
        entry_period: 진입 채널 기간 (N일)
        exit_period: 청산 채널 기간 (M일, M < N)
        atr_period: ATR 계산 기간
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 모드 (DISABLED/FULL)

    Example:
        >>> config = DonchianConfig(entry_period=20, exit_period=10)
        >>> config.warmup_periods()
        21
    """

    model_config = ConfigDict(frozen=True)

    # Entry Channel (진입)
    entry_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description="진입 채널 기간 (N일 최고/최저)",
    )

    # Exit Channel (청산) - 반드시 entry_period보다 작아야 함
    exit_period: int = Field(
        default=10,
        ge=3,
        le=50,
        description="청산 채널 기간 (M일 최고/최저, M < N)",
    )

    # ATR for position sizing
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (포지션 사이징용)",
    )

    # Volatility Scaling
    vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.40 = 40%)",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
    )

    # Short Mode
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드",
    )

    @model_validator(mode="after")
    def validate_channel_periods(self) -> Self:
        """exit_period < entry_period 검증."""
        if self.exit_period >= self.entry_period:
            msg = f"exit_period ({self.exit_period}) must be < entry_period ({self.entry_period})"
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """워밍업 기간 (캔들 수)."""
        return max(self.entry_period, self.atr_period) + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> "DonchianConfig":
        """타임프레임별 기본 설정."""
        annualization_map: dict[str, float] = {
            "1h": 8760.0,
            "2h": 4380.0,
            "3h": 2920.0,
            "4h": 2190.0,
            "6h": 1460.0,
            "1d": 365.0,
        }
        # 터틀 규칙: 20일 Entry, 10일 Exit (일봉 기준)
        entry_map: dict[str, int] = {
            "1h": 480,  # 20일
            "2h": 240,  # 20일
            "3h": 160,  # 20일
            "4h": 120,  # 20일
            "6h": 80,  # 20일
            "1d": 20,
        }
        exit_map: dict[str, int] = {
            "1h": 240,  # 10일
            "2h": 120,  # 10일
            "3h": 80,  # 10일
            "4h": 60,  # 10일
            "6h": 40,  # 10일
            "1d": 10,
        }

        return cls(
            entry_period=entry_map.get(timeframe, 20),
            exit_period=exit_map.get(timeframe, 10),
            annualization_factor=annualization_map.get(timeframe, 365.0),
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> "DonchianConfig":
        """보수적 설정 (55/20 터틀 시스템2)."""
        return cls(
            entry_period=55,
            exit_period=20,
            vol_target=0.30,
        )

    @classmethod
    def aggressive(cls) -> "DonchianConfig":
        """공격적 설정 (20/10 터틀 시스템1)."""
        return cls(
            entry_period=20,
            exit_period=10,
            vol_target=0.50,
        )
