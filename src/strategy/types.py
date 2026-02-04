"""전략 관련 타입 정의.

이 모듈은 전략 레이어에서 사용되는 공통 타입(Enum, TypeAlias)을 정의합니다.
모든 전략 구현체는 이 타입들을 사용하여 일관성을 유지해야 합니다.

Rules Applied:
    - #10 Python Standards: Modern typing (X | None, list[])
    - #16 Basedpyright Typing: type keyword for aliases
    - #01 Project Structure: Re-export from models to maintain backward compatibility
"""

from typing import NamedTuple

import pandas as pd

# Re-export from models (canonical location) for backward compatibility
# NOTE: Direction, SignalType are now defined in src/models/types.py
# This re-export allows existing code to continue using src.strategy.types imports
from src.models.types import Direction, SignalType

__all__ = ["Direction", "SignalType", "StrategySignals"]


class StrategySignals(NamedTuple):
    """전략 시그널 결과 (벡터화된 출력).

    모든 전략의 `generate_signals()` 메서드는 이 타입을 반환해야 합니다.
    VectorBT 및 QuantStats와 호환되는 표준 출력 형식입니다.

    Note:
        strength는 전략이 계산한 순수 시그널 강도입니다.
        레버리지 클램핑(max_leverage_cap)과 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 처리됩니다.

    Attributes:
        entries: 진입 시그널 (True = 진입, False = 대기)
        exits: 청산 시그널 (True = 청산, False = 유지)
        direction: 방향 시리즈 (-1, 0, 1)
        strength: 시그널 강도 (레버리지 무제한, PM에서 클램핑)

    Example:
        >>> signals = strategy.generate_signals(df)
        >>> signals.entries  # pd.Series[bool]
        >>> signals.strength  # pd.Series[float] (unbounded)
    """

    entries: pd.Series  # bool Series - 진입 시그널
    exits: pd.Series  # bool Series - 청산 시그널
    direction: pd.Series  # int Series (-1, 0, 1) - 방향
    strength: pd.Series  # float Series - 시그널 강도 (포지션 사이징용)


# Type Aliases (Python 3.12+)
type Price = float  # 가격 (DataFrame 내부에서는 float 사용)
type Volume = float  # 거래량
type Returns = float  # 수익률
type Volatility = float  # 변동성

# DataFrame 컬럼 관련 타입
type OHLCVColumns = tuple[str, str, str, str, str]  # (open, high, low, close, volume)

# 기본 OHLCV 컬럼 이름
DEFAULT_OHLCV_COLUMNS: OHLCVColumns = ("open", "high", "low", "close", "volume")
