# Strategy Orchestrator — State Persistence Plan

> **Version**: 1.0
> **Date**: 2026-02-14
> **Status**: DRAFT — Pending Approval
> **Scope**: Orchestrator 재시작 복구 (State Persistence & Recovery)
> **Prerequisites**: Orchestrator Phase 1~8 완료

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [현재 상태 분석](#2-현재-상태-분석)
3. [Phase 9: Orchestrator State Persistence](#3-phase-9-orchestrator-state-persistence)
4. [Phase 10: OMS 복원 버그 수정](#4-phase-10-oms-복원-버그-수정)
5. [Phase 11: History Persistence (Append-Only)](#5-phase-11-history-persistence-append-only)
6. [Phase 12: Recovery Integration & Validation](#6-phase-12-recovery-integration--validation)
7. [SQLite Schema Extension](#7-sqlite-schema-extension)
8. [Risk & Constraints](#8-risk--constraints)
9. [File Map](#9-file-map)
10. [Test Strategy](#10-test-strategy)

---

## 1. Executive Summary

### 문제

Orchestrator 모드에서 프로그램이 종료 후 재시작하면 **모든 Pod 상태가 초기화**된다.
PRODUCTION까지 승격된 Pod이 INCUBATION으로 리셋되고, daily_returns 이력 소실로
Risk Parity/Kelly 계산이 불가능하며, 열화 감지(PageHinkley)도 무력화된다.

### 현재 영속성 커버리지

```
┌──────────────────────────────────────────────────────┐
│              State Persistence Coverage               │
│                                                       │
│  EDA Layer (기존):                                    │
│    ✅ PM: positions, cash, weights → bot_state JSON  │
│    ✅ RM: peak_equity, circuit_breaker → bot_state   │
│    ⚠️  OMS: save만 구현, restore 누락 (버그)          │
│    ❌ AnalyticsEngine: in-memory only                │
│    ❌ CandleAggregator: in-memory only               │
│                                                       │
│  Orchestrator Layer (신규 대상):                       │
│    ❌ StrategyPod: lifecycle, fraction, positions     │
│    ❌ LifecycleManager: state machine, PH detector    │
│    ❌ Orchestrator: rebalance timestamp, targets      │
│    ❌ History: allocation, lifecycle events, PRC      │
│    ✅ CapitalAllocator: Stateless (영속 불필요)        │
│    ✅ RiskAggregator: Stateless (영속 불필요)          │
│    ✅ PositionNetter: Pure function (영속 불필요)      │
└──────────────────────────────────────────────────────┘
```

### 설계 원칙

| 원칙 | 설명 |
|------|------|
| **기존 패턴 재사용** | `StateManager` + `bot_state` 테이블 (key-value JSON) 패턴 유지 |
| **History는 전용 테이블** | 시계열 이력은 append-only 테이블로 분리 |
| **Stateless 유지** | Allocator/RiskAgg/Netter는 입력 기반 계산 — 영속 불필요 |
| **Graceful Degradation** | 저장 상태가 없으면 config 기본값으로 시작 (기존 동작) |
| **원자적 저장** | 전체 Orchestrator 상태를 단일 트랜잭션으로 저장 |

---

## 2. 현재 상태 분석

### 2.1 컴포넌트별 상태 분류

#### CRITICAL — 복구 없으면 운용 불가

| 컴포넌트 | 손실 상태 | 영향 |
|----------|----------|------|
| `StrategyPod._state` | LifecycleState (enum) | 모든 Pod이 INCUBATION으로 리셋 |
| `StrategyPod._capital_fraction` | 현재 배분 비율 | initial_fraction으로 복귀 — 동적 배분 무효화 |
| `StrategyPod._daily_returns` | 일간 수익률 이력 | Risk Parity/Kelly 계산 불가 |
| `StrategyPod._performance` | PodPerformance 13개 필드 | 졸업/퇴출 판정 불가 |
| `StrategyPod._positions` | PodPosition per symbol | Fill 귀속 불가, Pod P&L 왜곡 |
| `LifecycleManager._pod_states` | state_entered_at, loss_months | 시간 기반 전이 판정 오류 |

#### HIGH — 감지/방어 무력화

| 컴포넌트 | 손실 상태 | 영향 |
|----------|----------|------|
| `PageHinkleyDetector` | n, x_mean, m_t, m_min | 열화 감지 누적 통계 리셋 |
| `OMS._processed_orders` | 처리 완료 주문 ID set | 멱등성 갭 — 중복 주문 위험 |

#### MEDIUM — 운영 연속성 저하

| 컴포넌트 | 손실 상태 | 영향 |
|----------|----------|------|
| `Orchestrator._last_rebalance_ts` | 마지막 리밸런스 시각 | 재시작 즉시 불필요 리밸런스 |
| `Orchestrator._last_pod_targets` | Pod별 목표 비중 | Fill 귀속 비율 오류 (첫 bar까지) |
| `Orchestrator._allocation_history` | 배분 이력 | 리포트/모니터링 공백 |
| `Orchestrator._lifecycle_events` | 생애주기 전이 이력 | 감사 추적 불가 |

#### LOW / 불필요

| 컴포넌트 | 이유 |
|----------|------|
| `CapitalAllocator` | Stateless — config + 입력 기반 계산 |
| `RiskAggregator` | Stateless — config + 입력 기반 계산 |
| `PositionNetter` | Pure function |
| `CandleAggregator._partials` | 재시작 후 첫 캔들 1개만 불완전 (허용 가능) |
| `StrategyPod._buffers` | REST warmup으로 재구성 (`inject_warmup()`) |

### 2.2 기존 영속 메커니즘

```
StateManager (bot_state key-value)
├── save_pm_state()  → "pm_state" key → JSON
├── save_rm_state()  → "rm_state" key → JSON
├── save_oms_state() → "oms_processed_orders" key → JSON
└── save_all(pm, rm, oms?)

LiveRunner
├── _restore_state()        → load PM + RM (OMS 누락 ⚠️)
├── _periodic_state_save()  → save PM + RM every 300s (OMS 누락 ⚠️)
└── shutdown                → save PM + RM (OMS 누락 ⚠️)
```

---

## 3. Phase 9: Orchestrator State Persistence

> **목표**: Pod/Lifecycle/Orchestrator 핵심 상태를 `bot_state`에 저장/복구

### 3.1 직렬화 대상

```python
# bot_state key: "orchestrator_state"
{
    "version": 1,
    "saved_at": "2026-02-14T12:00:00+00:00",

    # Orchestrator 레벨
    "last_rebalance_ts": "2026-02-14T00:00:00+00:00",  # nullable
    "last_pod_targets": {
        "pod-tsmom-major": {"BTC/USDT": 0.15, "ETH/USDT": 0.10},
        ...
    },

    # Pod 레벨
    "pods": {
        "pod-tsmom-major": {
            "lifecycle_state": "production",
            "capital_fraction": 0.35,
            "target_weights": {"BTC/USDT": 0.30, "ETH/USDT": 0.20},
            "positions": {
                "BTC/USDT": {
                    "target_weight": 0.30,
                    "global_weight": 0.105,
                    "notional_usd": 10500.0,
                    "unrealized_pnl": 230.0,
                    "realized_pnl": 1500.0
                }
            },
            "performance": {
                "total_return": 0.12,
                "sharpe_ratio": 1.82,
                "max_drawdown": 0.08,
                "calmar_ratio": 1.50,
                "win_rate": 0.55,
                "trade_count": 45,
                "live_days": 120,
                "rolling_volatility": 0.15,
                "peak_equity": 35000.0,
                "current_equity": 34500.0,
                "current_drawdown": 0.014,
                "last_updated": "2026-02-14T00:00:00+00:00"
            }
        },
        ...
    },

    # Lifecycle 레벨
    "lifecycle_states": {
        "pod-tsmom-major": {
            "state_entered_at": "2026-01-01T00:00:00+00:00",
            "consecutive_loss_months": 0,
            "last_monthly_check_day": 1,
            "ph_detector": {
                "n": 120,
                "x_mean": 0.001,
                "m_t": 0.45,
                "m_min": 0.02
            }
        },
        ...
    }
}
```

### 3.2 별도 key: Pod Daily Returns

daily_returns 이력은 크기가 클 수 있으므로 별도 key로 분리:

```python
# bot_state key: "orchestrator_daily_returns"
{
    "pod-tsmom-major": [0.01, -0.005, 0.003, ...],  # 최근 N일
    "pod-donchian-alt": [0.008, -0.002, ...],
    ...
}
```

> **최대 보관**: `correlation_lookback * 3` (기본 270일) — 이후 오래된 데이터 trim

### 3.3 새 파일: `src/orchestrator/state_persistence.py`

```python
"""Orchestrator 상태 영속성 — 저장/복구."""

from __future__ import annotations

_KEY_ORCHESTRATOR_STATE = "orchestrator_state"
_KEY_DAILY_RETURNS = "orchestrator_daily_returns"
_STATE_VERSION = 1
_MAX_DAILY_RETURNS = 270  # correlation_lookback * 3


class OrchestratorStatePersistence:
    """Orchestrator 상태 저장/복구.

    기존 StateManager의 bot_state 테이블을 활용하여
    Orchestrator/Pod/Lifecycle 상태를 JSON으로 직렬화.
    """

    def __init__(self, state_manager: StateManager) -> None:
        self._state_mgr = state_manager

    # === 저장 ===

    async def save(
        self,
        orchestrator: StrategyOrchestrator,
    ) -> None:
        """Orchestrator 전체 상태를 원자적으로 저장.

        단일 JSON blob으로 직렬화 → bot_state 테이블 INSERT OR REPLACE.
        """

    def _serialize_pod(self, pod: StrategyPod) -> dict[str, object]:
        """Pod 상태 직렬화."""

    def _serialize_lifecycle(
        self, lifecycle: LifecycleManager
    ) -> dict[str, dict[str, object]]:
        """LifecycleManager 내부 상태 직렬화."""

    def _serialize_ph_detector(
        self, detector: PageHinkleyDetector
    ) -> dict[str, float]:
        """PageHinkley detector 4-tuple 직렬화."""

    async def _save_daily_returns(
        self, pods: list[StrategyPod]
    ) -> None:
        """Pod별 daily_returns를 별도 key로 저장 (trim 적용)."""

    # === 복구 ===

    async def restore(
        self,
        orchestrator: StrategyOrchestrator,
    ) -> bool:
        """저장된 상태로 Orchestrator 복구.

        Returns:
            True if state was restored, False if no saved state exists.
        """

    def _restore_pod(
        self,
        pod: StrategyPod,
        data: dict[str, object],
    ) -> None:
        """Pod 상태 복구."""

    def _restore_lifecycle(
        self,
        lifecycle: LifecycleManager,
        data: dict[str, dict[str, object]],
    ) -> None:
        """LifecycleManager 상태 복구."""

    def _restore_ph_detector(
        self,
        detector: PageHinkleyDetector,
        data: dict[str, float],
    ) -> None:
        """PageHinkley detector 복구."""

    async def _restore_daily_returns(
        self, pods: list[StrategyPod]
    ) -> None:
        """Pod별 daily_returns 복구."""
```

### 3.4 컴포넌트 확장

#### `PageHinkleyDetector` — 직렬화 메서드 추가

```python
class PageHinkleyDetector:
    ...

    def to_dict(self) -> dict[str, float | int]:
        """내부 상태 직렬화."""
        return {
            "n": self._n,
            "x_mean": self._x_mean,
            "m_t": self._m_t,
            "m_min": self._m_min,
        }

    def restore_from_dict(self, data: dict[str, float | int]) -> None:
        """저장된 상태에서 복구."""
        self._n = int(data["n"])
        self._x_mean = float(data["x_mean"])
        self._m_t = float(data["m_t"])
        self._m_min = float(data["m_min"])
```

#### `LifecycleManager` — 직렬화/복구 메서드 추가

```python
class LifecycleManager:
    ...

    def to_dict(self) -> dict[str, dict[str, object]]:
        """전체 pod 상태 직렬화."""

    def restore_from_dict(
        self, data: dict[str, dict[str, object]]
    ) -> None:
        """저장된 상태에서 복구."""
```

#### `StrategyPod` — 직렬화/복구 메서드 추가

```python
class StrategyPod:
    ...

    def to_dict(self) -> dict[str, object]:
        """Pod 상태 직렬화 (buffers 제외)."""

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """저장된 상태에서 복구."""
```

#### `StrategyOrchestrator` — 직렬화/복구 메서드 추가

```python
class StrategyOrchestrator:
    ...

    def to_dict(self) -> dict[str, object]:
        """Orchestrator 레벨 상태 직렬화."""

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Orchestrator 레벨 상태 복구."""
```

### 3.5 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/state_persistence.py` (신규) | OrchestratorStatePersistence 클래스 |
| `src/orchestrator/degradation.py` (수정) | `to_dict()`, `restore_from_dict()` 추가 |
| `src/orchestrator/lifecycle.py` (수정) | `to_dict()`, `restore_from_dict()` 추가 |
| `src/orchestrator/pod.py` (수정) | `to_dict()`, `restore_from_dict()` 추가 |
| `src/orchestrator/orchestrator.py` (수정) | `to_dict()`, `restore_from_dict()` 추가 |
| `tests/orchestrator/test_state_persistence.py` | 직렬화/복구 단위 테스트 |

### 3.6 Estimated Tests: ~30

- 직렬화 round-trip: Pod 상태 save → restore → 동일 검증
- 직렬화 round-trip: Lifecycle 상태 save → restore
- 직렬화 round-trip: PH detector save → restore → score 동일
- 직렬화 round-trip: Orchestrator 전체 save → restore
- daily_returns trim: 270일 초과 시 잘림 검증
- 빈 상태: 저장 상태 없으면 False 반환, config 기본값 유지
- version mismatch: 미래 version → graceful skip
- pod_id mismatch: config에 없는 pod → 무시
- pod_id 추가: 새 pod은 기본값으로 시작
- 부분 복구: 일부 필드 누락 → 기본값 fallback
- JSON 직렬화 크기: 10 pod × 270일 daily_returns → 합리적 크기 검증

---

## 4. Phase 10: OMS 복원 버그 수정

> **목표**: `LiveRunner._restore_state()`에서 OMS 상태 복원 누락 수정

### 4.1 문제

현재 `StateManager`는 OMS 저장 메서드(`save_oms_state`)를 갖고 있지만:

1. `_restore_state()`에서 `load_oms_state()` **미호출**
2. `_periodic_state_save()`에서 OMS **미저장** (`save_all(pm, rm)` — oms=None)
3. shutdown 시 `save_all(pm, rm)` — OMS **미저장**

이로 인해 재시작 시 `_processed_orders`가 비어 있어 중복 주문 위험이 있다.

### 4.2 수정: `src/eda/live_runner.py`

```python
# _restore_state() 수정 — OMS 복원 추가
@staticmethod
async def _restore_state(
    db: Database | None,
    pm: EDAPortfolioManager,
    rm: EDARiskManager,
    oms: OMS | None = None,          # 추가
) -> StateManager | None:
    if db is None:
        return None
    state_mgr = StateManager(db)
    pm_state = await state_mgr.load_pm_state()
    if pm_state:
        pm.restore_state(pm_state)
    rm_state = await state_mgr.load_rm_state()
    if rm_state:
        rm.restore_state(rm_state)
    # OMS 복원 추가
    if oms is not None:
        oms_state = await state_mgr.load_oms_state()
        if oms_state:
            oms.restore_processed_orders(oms_state)
            logger.info("OMS state restored ({} orders)", len(oms_state))
    return state_mgr

# _periodic_state_save() 수정 — OMS 포함
@staticmethod
async def _periodic_state_save(
    state_mgr: StateManager,
    pm: EDAPortfolioManager,
    rm: EDARiskManager,
    oms: OMS | None = None,          # 추가
    interval: float = 300.0,
) -> None:
    while True:
        await asyncio.sleep(interval)
        await state_mgr.save_all(pm, rm, oms=oms)

# shutdown 수정 — OMS 포함
if state_mgr:
    await state_mgr.save_all(pm, rm, oms=oms)
```

### 4.3 호출 위치 수정

`run()` 메서드 내에서 OMS 생성 후 `_restore_state()` 호출 순서 조정:

```
기존: PM 생성 → RM 생성 → _restore_state(pm, rm) → OMS 생성
수정: PM 생성 → RM 생성 → OMS 생성 → _restore_state(pm, rm, oms)
```

### 4.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/eda/live_runner.py` (수정) | `_restore_state()`, `_periodic_state_save()`, shutdown에 OMS 추가 |
| `tests/eda/test_live_runner_oms_restore.py` | OMS 복원 테스트 |

### 4.5 Estimated Tests: ~8

- OMS 복원: 저장된 order ID set이 복원 후 동일
- OMS 없는 경우: oms=None → 기존 동작 유지 (하위 호환)
- 주기 저장: _periodic_state_save에 OMS 포함 검증
- shutdown 저장: 종료 시 OMS 상태 포함 검증
- 빈 상태: OMS 저장 없을 때 빈 set으로 시작

---

## 5. Phase 11: History Persistence (Append-Only)

> **목표**: 배분 이력, 생애주기 이벤트, 리스크 기여도를 시계열 테이블에 저장

### 5.1 새 SQLite 테이블

```sql
-- 자본 배분 이력
CREATE TABLE IF NOT EXISTS orchestrator_allocation_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    pod_id          TEXT NOT NULL,
    capital_fraction REAL NOT NULL,
    lifecycle_state TEXT NOT NULL,
    method          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alloc_hist_ts
    ON orchestrator_allocation_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_alloc_hist_pod
    ON orchestrator_allocation_history(pod_id);

-- 생애주기 전이 이벤트
CREATE TABLE IF NOT EXISTS orchestrator_lifecycle_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    pod_id          TEXT NOT NULL,
    from_state      TEXT NOT NULL,
    to_state        TEXT NOT NULL,
    reason          TEXT,
    performance_snapshot TEXT    -- JSON: PodPerformance at transition time
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_ts
    ON orchestrator_lifecycle_events(timestamp);

-- 리스크 기여도 이력
CREATE TABLE IF NOT EXISTS orchestrator_risk_contributions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    pod_id          TEXT NOT NULL,
    prc             REAL NOT NULL,
    effective_n     REAL NOT NULL,
    portfolio_volatility REAL
);

CREATE INDEX IF NOT EXISTS idx_risk_contrib_ts
    ON orchestrator_risk_contributions(timestamp);
```

### 5.2 새 파일: `src/orchestrator/history_persistence.py`

```python
"""Orchestrator 이력 영속성 — Append-Only 테이블."""


class OrchestratorHistoryPersistence:
    """Orchestrator 이벤트 이력을 SQLite에 기록.

    StateManager의 key-value 저장과 달리,
    시계열 데이터를 append-only 테이블에 기록합니다.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    async def initialize(self) -> None:
        """테이블 생성 (IF NOT EXISTS)."""

    # === 배분 이력 ===

    async def record_allocation(
        self,
        timestamp: datetime,
        allocations: dict[str, float],
        pod_states: dict[str, LifecycleState],
        method: AllocationMethod,
    ) -> None:
        """리밸런스 시 배분 결과 기록."""

    async def get_allocation_history(
        self,
        pod_id: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        """배분 이력 조회."""

    # === 생애주기 이벤트 ===

    async def record_lifecycle_event(
        self,
        timestamp: datetime,
        pod_id: str,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
        performance: PodPerformance | None = None,
    ) -> None:
        """상태 전이 이벤트 기록."""

    async def get_lifecycle_events(
        self,
        pod_id: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, object]]:
        """생애주기 이벤트 조회."""

    # === 리스크 기여도 ===

    async def record_risk_contributions(
        self,
        timestamp: datetime,
        contributions: dict[str, float],
        effective_n: float,
        portfolio_vol: float | None = None,
    ) -> None:
        """리스크 기여도 기록."""

    async def get_risk_contributions(
        self,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        """리스크 기여도 이력 조회."""
```

### 5.3 Orchestrator 연동

`StrategyOrchestrator._periodic_rebalance()`에서 리밸런스 수행 시 자동 기록:

```python
async def _periodic_rebalance(self) -> None:
    ...
    # 기존: in-memory history 추가
    self._allocation_history.append(...)

    # 추가: DB 기록
    if self._history_persistence:
        await self._history_persistence.record_allocation(...)
```

생애주기 전이 시 자동 기록:

```python
# LifecycleManager.evaluate()에서 상태 전이 발생 시
if new_state != old_state:
    self._lifecycle_events.append(...)

    # 추가: DB 기록
    if self._history_persistence:
        await self._history_persistence.record_lifecycle_event(...)
```

### 5.4 재시작 시 in-memory 복구

```python
# OrchestratorStatePersistence.restore()에서
# History는 DB에 이미 있으므로 in-memory list를 DB에서 복구
orchestrator._allocation_history = (
    await history.get_allocation_history(limit=1000)
)
orchestrator._lifecycle_events = (
    await history.get_lifecycle_events()
)
```

### 5.5 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/history_persistence.py` (신규) | OrchestratorHistoryPersistence |
| `src/eda/persistence/schema.py` (수정) | 3개 테이블 DDL 추가 |
| `src/orchestrator/orchestrator.py` (수정) | history_persistence 연동 |
| `tests/orchestrator/test_history_persistence.py` | 이력 저장/조회 테스트 |

### 5.6 Estimated Tests: ~15

- 배분 이력: record → get round-trip
- 배분 이력: pod_id 필터링
- 배분 이력: since 필터링
- 생애주기 이벤트: 전이 기록 + 조회
- 생애주기 이벤트: performance snapshot JSON 검증
- 리스크 기여도: record → get round-trip
- 빈 테이블: 이력 없을 때 빈 리스트 반환
- 대량 데이터: 1000건 이상에서 limit 동작 검증

---

## 6. Phase 12: Recovery Integration & Validation

> **목표**: LiveRunner에 Orchestrator 영속성 통합 + E2E 검증

### 6.1 수정: `src/eda/live_runner.py`

```python
class LiveRunner:

    async def run(self) -> None:
        ...
        # === Orchestrator 상태 복구 (추가) ===
        if self._orchestrator and db:
            from src.orchestrator.state_persistence import (
                OrchestratorStatePersistence,
            )
            orch_persistence = OrchestratorStatePersistence(state_mgr)
            restored = await orch_persistence.restore(self._orchestrator)
            if restored:
                logger.info("Orchestrator state restored")

        ...

        # === 주기적 저장 (Orchestrator 포함) ===
        if self._orchestrator and orch_persistence:
            save_tasks.append(
                asyncio.create_task(
                    self._periodic_orchestrator_save(
                        orch_persistence, self._orchestrator
                    )
                )
            )

        ...

        # === 종료 시 최종 저장 (추가) ===
        if self._orchestrator and orch_persistence:
            await orch_persistence.save(self._orchestrator)
            logger.info("Orchestrator final state saved")

    @staticmethod
    async def _periodic_orchestrator_save(
        persistence: OrchestratorStatePersistence,
        orchestrator: StrategyOrchestrator,
        interval: float = 300.0,
    ) -> None:
        """Orchestrator 상태 주기적 저장."""
        while True:
            await asyncio.sleep(interval)
            try:
                await persistence.save(orchestrator)
            except Exception:
                logger.exception("Orchestrator state save failed")
```

### 6.2 복구 흐름 (전체)

```
프로그램 재시작
    │
    ├─ 1. SQLite 연결 (기존)
    │
    ├─ 2. PM 생성 → RM 생성 → OMS 생성
    │
    ├─ 3. _restore_state(pm, rm, oms)     ← OMS 복원 추가 (Phase 10)
    │      ├─ PM 포지션/현금/비중 복구
    │      ├─ RM peak_equity/circuit_breaker 복구
    │      └─ OMS processed_orders 복구
    │
    ├─ 4. Orchestrator 생성 (Pods, Allocator, Lifecycle 등)
    │
    ├─ 5. OrchestratorStatePersistence.restore()  ← 신규 (Phase 9)
    │      ├─ Pod별: lifecycle_state, capital_fraction,
    │      │         positions, performance 복구
    │      ├─ Pod별: daily_returns 복구 (별도 key)
    │      ├─ Lifecycle: state_entered_at,
    │      │            consecutive_loss_months, PH detector 복구
    │      ├─ Orchestrator: last_rebalance_ts,
    │      │               last_pod_targets 복구
    │      └─ History: allocation/lifecycle in-memory list 복구
    │
    ├─ 6. REST warmup (기존) — OHLCV buffers 재구성
    │
    ├─ 7. EventBus 등록 + WebSocket 시작
    │
    └─ 8. 운용 재개 (이전 상태에서 연속)
```

### 6.3 Invariant 검증

재시작 복구 후 반드시 검증할 invariant:

```python
def validate_restored_state(orchestrator: StrategyOrchestrator) -> None:
    """복구 후 상태 정합성 검증."""

    # 1. Pod fraction 합계 <= 1.0
    total = sum(p.capital_fraction for p in orchestrator.pods)
    assert total <= 1.0 + 1e-9

    # 2. RETIRED Pod은 fraction == 0
    for pod in orchestrator.pods:
        if pod.state == LifecycleState.RETIRED:
            assert pod.capital_fraction == 0.0

    # 3. Pod ID는 config와 1:1 매칭
    config_ids = {p.pod_id for p in orchestrator.config.pods}
    state_ids = {p.pod_id for p in orchestrator.pods}
    assert config_ids == state_ids

    # 4. daily_returns 길이 <= max_daily_returns
    for pod in orchestrator.pods:
        assert len(pod.daily_returns) <= _MAX_DAILY_RETURNS
```

### 6.4 Edge Cases

| 시나리오 | 동작 |
|----------|------|
| 저장 상태 없음 (첫 실행) | 모든 Pod → config 기본값으로 시작 |
| Config Pod 목록 변경 | 새 Pod → 기본값, 제거된 Pod → 무시 |
| Config 파라미터 변경 | Config 값 우선 (strategy_params 등), 상태는 유지 |
| 저장 중 crash | SQLite WAL → 직전 완료 트랜잭션까지 복구 |
| 상태 파일 손상 | JSON 파싱 실패 → 경고 로그 + 기본값 시작 |
| Version mismatch | 현재 version > 저장 version → 마이그레이션 또는 기본값 |

### 6.5 Deliverables

| 파일 | 내용 |
|------|------|
| `src/eda/live_runner.py` (수정) | Orchestrator 영속성 통합 |
| `tests/orchestrator/test_recovery_integration.py` | E2E 복구 테스트 |

### 6.6 Estimated Tests: ~20

- E2E: Orchestrator 실행 → 저장 → 재생성 → 복구 → 동일 상태
- E2E: 2 Pod 운용 → 종료 → 재시작 → 배분 비율 유지
- E2E: PRODUCTION Pod → 재시작 → 여전히 PRODUCTION
- E2E: WARNING Pod → 재시작 → 여전히 WARNING + PH detector 연속
- Edge: Config에 Pod 추가 → 새 Pod만 INCUBATION
- Edge: Config에서 Pod 제거 → 저장 상태 무시
- Edge: 저장 상태 없음 → 기존 동작 (config 기본값)
- Edge: JSON 손상 → 경고 + 기본값 fallback
- Invariant: 복구 후 fraction 합계 <= 1.0
- Invariant: 복구 후 RETIRED Pod fraction == 0
- OMS: 복구 후 중복 주문 방지 확인
- Periodic save: 300초 간격 저장 동작 확인

---

## 7. SQLite Schema Extension

### 7.1 기존 테이블 (변경 없음)

```
bot_state           ← key-value, orchestrator_state/daily_returns 추가 사용
trades              ← 변경 없음
equity_snapshots    ← 변경 없음
positions_history   ← 변경 없음
risk_events         ← 변경 없음
```

### 7.2 신규 테이블 (Phase 11)

```
orchestrator_allocation_history    ← 배분 이력 (append-only)
orchestrator_lifecycle_events      ← 생애주기 전이 (append-only)
orchestrator_risk_contributions    ← 리스크 기여도 (append-only)
```

### 7.3 bot_state 사용 현황

| Key | 용도 | Phase |
|-----|------|-------|
| `pm_state` | PM 포지션/현금 | 기존 |
| `rm_state` | RM peak_equity | 기존 |
| `oms_processed_orders` | OMS 멱등성 set | 기존 (복원 수정) |
| `last_save_timestamp` | 마지막 저장 시각 | 기존 |
| `orchestrator_state` | Orchestrator/Pod/Lifecycle 상태 | **Phase 9** |
| `orchestrator_daily_returns` | Pod별 일간 수익률 이력 | **Phase 9** |

---

## 8. Risk & Constraints

### 8.1 기술적 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| JSON blob 크기 증가 | 10+ Pod, 270일 daily_returns → 수 MB | 별도 key 분리 + trim |
| SQLite 동시 쓰기 | 주기적 저장 vs 이벤트 기록 충돌 | WAL 모드 (기존 설정) |
| 직렬화/역직렬화 비용 | 5분마다 전체 상태 직렬화 | JSON 기준 < 10ms (무시 가능) |
| Version 마이그레이션 | 상태 구조 변경 시 호환성 | version 필드 + graceful fallback |

### 8.2 운영 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 저장 후 5분 내 crash | 최대 5분 상태 손실 | SIGTERM 시 즉시 저장 (기존) |
| 부분 복구 | 일부 Pod만 복구 성공 | Pod별 독립 복구, 실패 시 기본값 |
| Config-State 불일치 | Config 변경 후 재시작 | Config 값 우선, State는 보조 |

### 8.3 제약 사항

- OHLCV buffers는 영속하지 않음 → REST warmup 의존 (기존 동작)
- CandleAggregator partial 캔들은 영속하지 않음 → 첫 캔들 불완전 (허용)
- AnalyticsEngine in-memory 이력은 영속하지 않음 → TradePersistence로 대체 가능

---

## 9. File Map

### 9.1 새 파일 (2개)

```
src/orchestrator/
├── state_persistence.py       # Phase 9: 상태 저장/복구
└── history_persistence.py     # Phase 11: 이력 기록/조회
```

### 9.2 수정 파일 (7개)

```
src/orchestrator/degradation.py    # Phase 9: to_dict/restore_from_dict
src/orchestrator/lifecycle.py      # Phase 9: to_dict/restore_from_dict
src/orchestrator/pod.py            # Phase 9: to_dict/restore_from_dict
src/orchestrator/orchestrator.py   # Phase 9+11: 직렬화 + history 연동
src/eda/live_runner.py             # Phase 10+12: OMS 복원 + Orchestrator 통합
src/eda/persistence/schema.py      # Phase 11: 3 테이블 DDL 추가
```

### 9.3 테스트 파일 (4개)

```
tests/orchestrator/
├── test_state_persistence.py      # Phase 9
├── test_history_persistence.py    # Phase 11
└── test_recovery_integration.py   # Phase 12

tests/eda/
└── test_live_runner_oms_restore.py # Phase 10
```

---

## 10. Test Strategy

### 10.1 단위 테스트 (~73 예상)

| Phase | 테스트 수 | 핵심 검증 |
|-------|----------|----------|
| 9. State Persistence | ~30 | 직렬화 round-trip, 부분 복구, version 호환 |
| 10. OMS 복원 수정 | ~8 | OMS restore 동작, 하위 호환 |
| 11. History Persistence | ~15 | append-only 기록, 필터 조회 |
| 12. Recovery Integration | ~20 | E2E 복구, invariant, edge cases |

### 10.2 Quality Gate

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/orchestrator/ src/eda/persistence/
uv run pytest tests/orchestrator/test_state_persistence.py \
              tests/orchestrator/test_history_persistence.py \
              tests/orchestrator/test_recovery_integration.py \
              tests/eda/test_live_runner_oms_restore.py \
              --cov=src/orchestrator --cov-report=term
# Coverage >= 90% 필수
```

---

## Implementation Order (Summary)

| Phase | 이름 | 의존성 | 예상 테스트 |
|-------|------|--------|-----------|
| **9** | Orchestrator State Persistence | Phase 1~8 완료 | ~30 |
| **10** | OMS 복원 버그 수정 | 없음 (독립) | ~8 |
| **11** | History Persistence | Phase 9 | ~15 |
| **12** | Recovery Integration | Phase 9, 10, 11 | ~20 |

**총 예상**: ~73 신규 테스트, 2 신규 파일, 7 수정 파일

> **Phase 10은 독립적**이므로 Phase 9와 병렬 진행 가능
