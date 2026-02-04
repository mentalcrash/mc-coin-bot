"""공용 타입 정의.

이 모듈은 여러 레이어에서 공통으로 사용되는 타입(Enum, TypeAlias)을 정의합니다.
Strategy, Models, Backtest 등 다양한 모듈에서 순환 참조 없이 사용할 수 있습니다.

Rules Applied:
    - #10 Python Standards: Modern typing (X | None, list[])
    - #16 Basedpyright Typing: type keyword for aliases
    - #01 Project Structure: Dependency flow (Models can be imported by all layers)
"""

from enum import Enum, IntEnum


class Direction(IntEnum):
    """매매 방향 (포지션 방향).

    VectorBT와 호환되는 정수 값을 사용합니다.
    - SHORT (-1): 숏 포지션 / 매도
    - NEUTRAL (0): 중립 / 홀드
    - LONG (1): 롱 포지션 / 매수
    """

    SHORT = -1
    NEUTRAL = 0
    LONG = 1


class SignalType(str, Enum):
    """시그널 유형.

    전략이 생성하는 시그널의 유형을 정의합니다.
    """

    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
