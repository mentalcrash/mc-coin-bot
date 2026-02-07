"""EDA 설정 모델.

Event-Driven Architecture 백테스트 및 실행 설정을 정의합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Field validators
    - #10 Python Standards: Modern typing (StrEnum)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(StrEnum):
    """실행 모드.

    EDA 시스템의 실행 모드를 정의합니다.
    각 모드는 데이터 소스와 체결 방식이 다릅니다.
    """

    BACKTEST = "backtest"  # 히스토리컬 리플레이 + 시뮬레이션 체결
    SHADOW = "shadow"  # 실시간 데이터 + 시그널 로깅만 (주문 없음)
    PAPER = "paper"  # 실시간 데이터 + 시뮬레이션 체결
    CANARY = "canary"  # 실시간 데이터 + 실주문 (소액)
    LIVE = "live"  # 실시간 데이터 + 실주문 (전체 자본)


class EDAConfig(BaseModel):
    """EDA 시스템 설정.

    EventBus, 이벤트 로깅, 백테스트 체결 지연 등 EDA 고유 설정입니다.

    Attributes:
        execution_mode: 실행 모드
        event_queue_size: EventBus 큐 최대 크기
        event_log_path: JSONL 이벤트 로그 파일 경로 (None=비활성)
        enable_heartbeat: 헬스체크 이벤트 활성화
        heartbeat_interval_bars: 헬스체크 주기 (bar 단위)
    """

    model_config = ConfigDict(frozen=True)

    execution_mode: ExecutionMode = ExecutionMode.BACKTEST
    event_queue_size: int = Field(default=10000, ge=100)
    event_log_path: str | None = None
    enable_heartbeat: bool = True
    heartbeat_interval_bars: int = Field(default=100, ge=1)
