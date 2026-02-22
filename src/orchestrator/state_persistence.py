"""OrchestratorStatePersistence — Orchestrator/Pod/Lifecycle 상태 영속화.

StateManager의 bot_state key-value 패턴을 재사용하여
Orchestrator 재시작 시 Pod 상태·성과·daily_returns를 복구합니다.

Keys:
    orchestrator_state        — Pod/Lifecycle/Orchestrator 핵심 상태 (version 포함)
    orchestrator_daily_returns — Pod별 일간 수익률 (크기 분리, trim 270일)

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - #23 Exception Handling: JSONDecodeError/version 불일치 graceful fallback
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.eda.persistence.database import Database
    from src.orchestrator.orchestrator import StrategyOrchestrator
    from src.orchestrator.surveillance import MarketSurveillanceService

# ── Constants ─────────────────────────────────────────────────────

_KEY_STATE = "orchestrator_state"
_KEY_DAILY_RETURNS = "orchestrator_daily_returns"
_KEY_HISTORIES = "orchestrator_histories"
_KEY_SURVEILLANCE = "surveillance_state"
_STATE_VERSION = 1
_MAX_DAILY_RETURNS = 270
_MAX_ALLOCATION_HISTORY = 500
_MAX_LIFECYCLE_EVENTS = 100
_MAX_RISK_HISTORY = 500


class OrchestratorStatePersistence:
    """Orchestrator/Pod/Lifecycle 상태를 bot_state에 영속화.

    Args:
        state_manager: StateManager 인스턴스 (DB 연결 완료 상태).
            Database를 직접 받아 bot_state를 읽고 씁니다.
    """

    def __init__(self, state_manager: Any) -> None:
        # StateManager의 _db (Database) 참조
        self._db: Database = state_manager._db

    # ── Save ──────────────────────────────────────────────────────

    async def save(self, orchestrator: StrategyOrchestrator) -> None:
        """Orchestrator 전체 상태를 bot_state에 저장.

        1. orchestrator_state: orchestrator + pods + lifecycle
        2. orchestrator_daily_returns: pod별 daily_returns (trim 270)
        """
        # 1. Core state
        pods_data: dict[str, object] = {}
        daily_returns_data: dict[str, list[float]] = {}

        for pod in orchestrator.pods:
            pods_data[pod.pod_id] = pod.to_dict()
            # daily_returns: trim to 270
            returns = list(pod.daily_returns)
            if len(returns) > _MAX_DAILY_RETURNS:
                returns = returns[-_MAX_DAILY_RETURNS:]
            daily_returns_data[pod.pod_id] = returns

        lifecycle_data: dict[str, dict[str, object]] | None = None
        lifecycle = orchestrator.lifecycle
        if lifecycle is not None:
            lifecycle_data = lifecycle.to_dict()

        state_payload: dict[str, object] = {
            "version": _STATE_VERSION,
            "orchestrator": orchestrator.to_dict(),
            "pods": pods_data,
            "lifecycle": lifecycle_data,
        }

        # 3. Histories (allocation, lifecycle, risk contributions)
        histories_payload = self._build_histories_payload(orchestrator)

        await self._save_keys_atomic(
            [
                (_KEY_STATE, json.dumps(state_payload)),
                (_KEY_DAILY_RETURNS, json.dumps(daily_returns_data)),
                (_KEY_HISTORIES, json.dumps(histories_payload, default=str)),
            ]
        )

        logger.debug("Orchestrator state saved: {} pods", len(pods_data))

    def _build_histories_payload(
        self, orchestrator: StrategyOrchestrator
    ) -> dict[str, list[dict[str, object]]]:
        """Build trimmed histories payload for persistence."""
        alloc = list(orchestrator.allocation_history)
        if len(alloc) > _MAX_ALLOCATION_HISTORY:
            alloc = alloc[-_MAX_ALLOCATION_HISTORY:]

        lifecycle = list(orchestrator.lifecycle_events)
        if len(lifecycle) > _MAX_LIFECYCLE_EVENTS:
            lifecycle = lifecycle[-_MAX_LIFECYCLE_EVENTS:]

        risk = list(orchestrator.risk_contributions_history)
        if len(risk) > _MAX_RISK_HISTORY:
            risk = risk[-_MAX_RISK_HISTORY:]

        return {
            "allocation_history": alloc,
            "lifecycle_events": lifecycle,
            "risk_contributions_history": risk,
        }

    # ── Restore ───────────────────────────────────────────────────

    async def restore(self, orchestrator: StrategyOrchestrator) -> bool:
        """Orchestrator 상태를 bot_state에서 복구.

        Returns:
            True if state was restored, False if no saved state or error.
        """
        raw = await self._load_key(_KEY_STATE)
        if raw is None:
            logger.info("No saved orchestrator state found — starting fresh")
            return False

        state = self._parse_state(raw)
        if state is None:
            return False

        # Restore orchestrator-level state
        orch_data = state.get("orchestrator")
        if isinstance(orch_data, dict):
            orchestrator.restore_from_dict(orch_data)

        # Restore pods (match by pod_id)
        self._restore_pods(orchestrator, state)

        # Restore lifecycle
        self._restore_lifecycle(orchestrator, state)

        # Restore daily_returns
        await self._restore_daily_returns(orchestrator)

        # Restore histories
        await self._restore_histories(orchestrator)

        logger.info("Orchestrator state restored successfully")
        return True

    # ── Parse / Validate ─────────────────────────────────────────

    def _parse_state(self, raw: str) -> dict[str, Any] | None:
        """Parse JSON + validate version."""
        try:
            state: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Orchestrator state: corrupted JSON — starting fresh")
            return None

        if not isinstance(state, dict):
            logger.warning("Orchestrator state: invalid format — starting fresh")
            return None

        version = state.get("version")
        if not isinstance(version, int) or version > _STATE_VERSION:
            logger.warning(
                "Orchestrator state: version {} > {} — starting fresh",
                version,
                _STATE_VERSION,
            )
            return None

        return state  # type: ignore[no-any-return]

    def _restore_pods(
        self,
        orchestrator: StrategyOrchestrator,
        state: dict[str, Any],
    ) -> None:
        """Match saved pod data to current config pods."""
        pods_data = state.get("pods")
        if not isinstance(pods_data, dict):
            return

        current_pod_ids = {pod.pod_id for pod in orchestrator.pods}
        saved_pod_ids = set(pods_data.keys())

        # Warn about added/removed pods
        added = current_pod_ids - saved_pod_ids
        removed = saved_pod_ids - current_pod_ids
        if added:
            logger.info("New pods (no saved state): {}", added)
        if removed:
            logger.info("Removed pods (saved state ignored): {}", removed)

        # Restore matching pods
        for pod in orchestrator.pods:
            pod_data = pods_data.get(pod.pod_id)
            if isinstance(pod_data, dict):
                pod.restore_from_dict(pod_data)

    def _restore_lifecycle(
        self,
        orchestrator: StrategyOrchestrator,
        state: dict[str, Any],
    ) -> None:
        """Restore lifecycle manager state."""
        lifecycle = orchestrator.lifecycle
        if lifecycle is None:
            return

        lifecycle_data = state.get("lifecycle")
        if isinstance(lifecycle_data, dict):
            lifecycle.restore_from_dict(lifecycle_data)

    async def _restore_daily_returns(
        self,
        orchestrator: StrategyOrchestrator,
    ) -> None:
        """Restore daily_returns from separate key + sync live_days."""
        raw = await self._load_key(_KEY_DAILY_RETURNS)
        if raw is None:
            return

        try:
            all_returns: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Orchestrator daily_returns: corrupted JSON — skipped")
            return

        if not isinstance(all_returns, dict):
            return

        for pod in orchestrator.pods:
            pod_returns = all_returns.get(pod.pod_id)
            if not isinstance(pod_returns, list):
                continue

            # Trim to max + restore
            if len(pod_returns) > _MAX_DAILY_RETURNS:
                pod_returns = pod_returns[-_MAX_DAILY_RETURNS:]

            pod.restore_daily_returns([float(r) for r in pod_returns])

    async def _restore_histories(
        self,
        orchestrator: StrategyOrchestrator,
    ) -> None:
        """Restore allocation/lifecycle/risk histories from separate key."""
        raw = await self._load_key(_KEY_HISTORIES)
        if raw is None:
            return

        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Orchestrator histories: corrupted JSON — skipped")
            return

        if not isinstance(data, dict):
            return

        alloc = data.get("allocation_history")
        lifecycle = data.get("lifecycle_events")
        risk = data.get("risk_contributions_history")

        orchestrator.restore_histories(
            allocation_history=alloc if isinstance(alloc, list) else [],
            lifecycle_events=lifecycle if isinstance(lifecycle, list) else [],
            risk_contributions_history=risk if isinstance(risk, list) else [],
        )

    # ── Surveillance ────────────────────────────────────────────

    async def save_surveillance(self, service: MarketSurveillanceService) -> None:
        """Surveillance 상태를 bot_state에 저장.

        Args:
            service: MarketSurveillanceService 인스턴스
        """
        payload = json.dumps(service.to_dict())
        await self._save_key(_KEY_SURVEILLANCE, payload)
        logger.debug("Surveillance state saved")

    async def restore_surveillance(self, service: MarketSurveillanceService) -> bool:
        """Surveillance 상태를 bot_state에서 복구.

        Args:
            service: MarketSurveillanceService 인스턴스

        Returns:
            True if restored, False if no saved state.
        """
        raw = await self._load_key(_KEY_SURVEILLANCE)
        if raw is None:
            logger.info("No saved surveillance state found — starting fresh")
            return False

        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Surveillance state: corrupted JSON — starting fresh")
            return False

        if isinstance(data, dict):
            service.restore_from_dict(data)
            logger.info("Surveillance state restored")
            return True

        return False

    # ── Internal ─────────────────────────────────────────────────

    async def _save_keys_atomic(self, entries: list[tuple[str, str]]) -> None:
        """Insert or replace multiple key-values in a single transaction."""
        conn = self._db.connection
        now = datetime.now(UTC).isoformat()
        for key, value in entries:
            await conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )
        await conn.commit()

    async def _save_key(self, key: str, value: str) -> None:
        """Insert or replace key-value in bot_state table."""
        conn = self._db.connection
        now = datetime.now(UTC).isoformat()
        await conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        await conn.commit()

    async def _load_key(self, key: str) -> str | None:
        """Load value from bot_state by key."""
        conn = self._db.connection
        cursor = await conn.execute(
            "SELECT value FROM bot_state WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return row[0]  # type: ignore[no-any-return]
