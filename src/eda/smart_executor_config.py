"""SmartExecutor 설정.

Limit order 우선 실행기의 설정을 정의합니다.
PortfolioManagerConfig에서 Pydantic 필드로 사용되므로 독립 모듈로 분리합니다.
"""

from pydantic import BaseModel, ConfigDict, Field


class SmartExecutorConfig(BaseModel):
    """SmartExecutor 설정 — Limit order 우선 실행.

    Attributes:
        enabled: SmartExecutor 활성화 (False=기존 market-only 동작)
        limit_timeout_seconds: Limit 주문 대기 시간 (초)
        price_offset_bps: bid/ask 대비 가격 offset (basis points)
        max_price_deviation_pct: 가격 이탈 시 조기 취소 임계값 (%)
        poll_interval_seconds: 주문 상태 확인 주기 (초)
        max_concurrent_limit_orders: 동시 limit 주문 제한
        fallback_to_market: timeout 시 market 주문 전환 여부
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="SmartExecutor 활성화 (opt-in)",
    )
    limit_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Limit 주문 대기 시간 (초)",
    )
    price_offset_bps: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="bid/ask 대비 가격 offset (basis points)",
    )
    max_price_deviation_pct: float = Field(
        default=0.3,
        ge=0.05,
        le=2.0,
        description="가격 이탈 시 조기 취소 임계값 (%)",
    )
    poll_interval_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=30.0,
        description="주문 상태 확인 주기 (초)",
    )
    max_concurrent_limit_orders: int = Field(
        default=4,
        ge=1,
        le=20,
        description="동시 limit 주문 제한",
    )
    fallback_to_market: bool = Field(
        default=True,
        description="timeout 시 market 주문 전환 여부",
    )
