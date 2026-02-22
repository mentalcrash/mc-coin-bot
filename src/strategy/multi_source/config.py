"""Multi-Source 전략 설정.

다중 데이터 소스(On-chain, Macro, Options, Derivatives)의 서브시그널을
결합하여 복합 시그널을 생성하는 전략의 설정 모델.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 모드."""

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class SignalCombineMethod(StrEnum):
    """서브시그널 결합 방법."""

    ZSCORE_SUM = "zscore_sum"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_SUM = "weighted_sum"


class SubSignalTransform(StrEnum):
    """서브시그널 전처리 방법."""

    ZSCORE = "zscore"
    PERCENTILE = "percentile"
    MA_CROSS = "ma_cross"
    MOMENTUM = "momentum"


class SubSignalSpec(BaseModel):
    """개별 서브시그널 사양.

    Attributes:
        column: enriched DataFrame 컬럼명 (예: "oc_fear_greed")
        transform: 전처리 방법
        window: rolling window 크기
        weight: 결합 시 가중치
        invert: True면 contrarian 시그널 (부호 반전)
    """

    model_config = ConfigDict(frozen=True)

    column: str
    transform: SubSignalTransform = SubSignalTransform.ZSCORE
    window: int = Field(default=30, ge=5, le=200)
    weight: float = Field(default=1.0, gt=0.0)
    invert: bool = False


class MultiSourceConfig(BaseModel):
    """Multi-Source 전략 설정.

    Attributes:
        signals: 서브시그널 사양 리스트 (최소 2개, 최대 5개)
        combine_method: 서브시그널 결합 방법
        entry_threshold: 진입 임계값 (결합 시그널 절대값)
        exit_threshold: 청산 임계값
        min_agreement: majority_vote 최소 합의 비율
        vol_target: 연환산 변동성 타겟
        vol_window: 변동성 계산 rolling window
        short_mode: 숏 포지션 허용 모드
    """

    model_config = ConfigDict(frozen=True)

    signals: tuple[SubSignalSpec, ...] = Field(min_length=2, max_length=5)
    combine_method: SignalCombineMethod = SignalCombineMethod.ZSCORE_SUM
    entry_threshold: float = Field(default=1.5, gt=0.0)
    exit_threshold: float = Field(default=0.5, ge=0.0)
    min_agreement: float = Field(default=0.6, ge=0.0, le=1.0)
    vol_target: float = Field(default=0.35, gt=0.0, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=200)
    min_volatility: float = Field(default=0.05, gt=0.0)
    annualization_factor: float = Field(default=365.0, gt=0.0)
    short_mode: ShortMode = ShortMode.HEDGE_ONLY

    @model_validator(mode="after")
    def _validate_thresholds(self) -> MultiSourceConfig:
        if self.exit_threshold >= self.entry_threshold:
            msg = f"exit_threshold ({self.exit_threshold}) must be < entry_threshold ({self.entry_threshold})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성 전 필요한 최소 bar 수."""
        max_window = max(s.window for s in self.signals)
        return max(max_window, self.vol_window) + 10
