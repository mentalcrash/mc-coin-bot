"""Netting offset Discord 알림 테스트.

netting_offset_warning_threshold 초과 시 RiskAlertEvent가 publish되는지 검증합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, RiskAlertEvent
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


class _OpposingStrategy(BaseStrategy):
    """항상 LONG(+1, strength=1.0) 시그널을 생성하는 전략."""

    def __init__(self, *, direction: int = 1) -> None:
        self._dir = direction

    @property
    def name(self) -> str:
        return "opposing"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        n = len(df)
        return StrategySignals(
            entries=pd.Series([True] * n, index=df.index),
            exits=pd.Series([False] * n, index=df.index),
            direction=pd.Series([self._dir] * n, index=df.index),
            strength=pd.Series([1.0] * n, index=df.index),
        )


def _make_config(
    threshold: float = 0.5,
) -> OrchestratorConfig:
    """2개의 반대 방향 Pod을 가진 OrchestratorConfig 생성."""
    return OrchestratorConfig(
        pods=(
            PodConfig(
                pod_id="long-pod",
                strategy_name="test",
                symbols=("BTC/USDT",),
                initial_fraction=0.5,
                max_fraction=0.5,
                min_fraction=0.01,
            ),
            PodConfig(
                pod_id="short-pod",
                strategy_name="test",
                symbols=("BTC/USDT",),
                initial_fraction=0.5,
                max_fraction=0.5,
                min_fraction=0.01,
            ),
        ),
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        netting_offset_warning_threshold=threshold,
    )


class TestNettingOffsetAlert:
    """Netting offset RiskAlertEvent 발행 검증."""

    @pytest.mark.asyncio
    async def test_risk_alert_published_on_high_offset(self) -> None:
        """offset_ratio > threshold 시 RiskAlertEvent가 publish된다."""
        config = _make_config(threshold=0.3)
        allocator = CapitalAllocator(config)

        # 반대 방향 Pod 구성 → 높은 offset_ratio
        long_strategy = _OpposingStrategy(direction=1)
        short_strategy = _OpposingStrategy(direction=-1)

        pod_long = StrategyPod(
            config=config.pods[0],
            strategy=long_strategy,
            capital_fraction=config.pods[0].initial_fraction,
        )
        pod_short = StrategyPod(
            config=config.pods[1],
            strategy=short_strategy,
            capital_fraction=config.pods[1].initial_fraction,
        )

        orchestrator = StrategyOrchestrator(
            config=config,
            pods=[pod_long, pod_short],
            allocator=allocator,
        )

        # EventBus mock
        bus = AsyncMock(spec=EventBus)
        published_events: list[AnyEvent] = []

        async def capture_publish(event: AnyEvent) -> None:
            published_events.append(event)

        bus.publish = capture_publish
        orchestrator._bus = bus

        # 반대 방향 시그널을 직접 _last_pod_targets에 설정
        orchestrator._last_pod_targets = {
            "long-pod": {"BTC/USDT": 0.5},
            "short-pod": {"BTC/USDT": -0.5},
        }
        orchestrator._pending_net_weights = {"BTC/USDT": 0.0}
        orchestrator._pending_bar_ts = datetime(2025, 6, 1, tzinfo=UTC)

        await orchestrator._flush_net_signals()

        # RiskAlertEvent 확인
        risk_alerts = [e for e in published_events if isinstance(e, RiskAlertEvent)]
        assert len(risk_alerts) >= 1
        alert = risk_alerts[0]
        assert alert.alert_level == "WARNING"
        assert "netting offset" in alert.message.lower()
        assert alert.source == "StrategyOrchestrator"

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self) -> None:
        """offset_ratio <= threshold 시 RiskAlertEvent가 발행되지 않는다."""
        config = _make_config(threshold=0.9)
        allocator = CapitalAllocator(config)

        strategy = _OpposingStrategy(direction=1)
        pod = StrategyPod(
            config=config.pods[0],
            strategy=strategy,
            capital_fraction=config.pods[0].initial_fraction,
        )

        orchestrator = StrategyOrchestrator(
            config=config,
            pods=[
                pod,
                StrategyPod(
                    config=config.pods[1],
                    strategy=strategy,
                    capital_fraction=config.pods[1].initial_fraction,
                ),
            ],
            allocator=allocator,
        )

        bus = AsyncMock(spec=EventBus)
        published_events: list[AnyEvent] = []

        async def capture_publish(event: AnyEvent) -> None:
            published_events.append(event)

        bus.publish = capture_publish
        orchestrator._bus = bus

        # 같은 방향 → offset 낮음
        orchestrator._last_pod_targets = {
            "long-pod": {"BTC/USDT": 0.5},
            "short-pod": {"BTC/USDT": 0.3},
        }
        orchestrator._pending_net_weights = {"BTC/USDT": 0.8}
        orchestrator._pending_bar_ts = datetime(2025, 6, 1, tzinfo=UTC)

        await orchestrator._flush_net_signals()

        risk_alerts = [e for e in published_events if isinstance(e, RiskAlertEvent)]
        assert len(risk_alerts) == 0

    def test_config_default_threshold(self) -> None:
        """기본 netting_offset_warning_threshold는 0.5."""
        config = OrchestratorConfig(
            pods=(
                PodConfig(
                    pod_id="pod-a",
                    strategy_name="test",
                    symbols=("BTC/USDT",),
                    initial_fraction=1.0,
                    max_fraction=1.0,
                ),
            ),
        )
        assert config.netting_offset_warning_threshold == 0.5

    def test_config_custom_threshold(self) -> None:
        """netting_offset_warning_threshold 커스텀 설정."""
        config = _make_config(threshold=0.7)
        assert config.netting_offset_warning_threshold == 0.7
