# Strategy Orchestrator Layer - Implementation Plan

> **Version**: 1.0
> **Date**: 2026-02-14
> **Status**: DRAFT - Pending Approval
> **Scope**: Multi-Strategy Portfolio Orchestration System

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Core Models & Config](#3-phase-1-core-models--config)
4. [Phase 2: Capital Allocator Engine](#4-phase-2-capital-allocator-engine)
5. [Phase 3: Strategy Pod & Orchestrator](#5-phase-3-strategy-pod--orchestrator)
6. [Phase 4: Lifecycle Manager & Degradation Detection](#6-phase-4-lifecycle-manager--degradation-detection)
7. [Phase 5: Position Netting & Risk Aggregation](#7-phase-5-position-netting--risk-aggregation)
8. [Phase 6: Runner Integration (Backtest + Live)](#8-phase-6-runner-integration-backtest--live)
9. [Phase 7: CLI & Config YAML](#9-phase-7-cli--config-yaml)
10. [Phase 8: Monitoring & Notification](#10-phase-8-monitoring--notification)
11. [Migration & Backward Compatibility](#11-migration--backward-compatibility)
12. [Risk & Constraints](#12-risk--constraints)
13. [File Map](#13-file-map)
14. [Test Strategy](#14-test-strategy)

---

## 1. Executive Summary

### 목적

현재 시스템은 **단일 전략 중심**(EnsembleStrategy로 부분적 멀티 지원)으로 설계되어 있다.
실제 라이브 운용에서는 **여러 독립 전략을 동시에 실행**하며, 각 전략에 **성과 기반으로
자본을 동적 배분**하고, **열화된 전략을 자동 축소/퇴출**하는 시스템이 필요하다.

### 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Pod 독립성** | 각 전략은 독립된 "Pod"로 운영 — 독립 P&L, 독립 리스크 |
| **Net Execution** | 실제 거래소 주문은 심볼별로 넷팅하여 마진 효율 극대화 |
| **기존 EDA 재사용** | EventBus, PM, RM, OMS 패턴을 최대한 재활용 |
| **점진적 배분** | 신규 전략은 소규모(5~10%)에서 시작, 성과에 따라 증가 |
| **자동 방어** | Degradation 감지 → 자동 축소 → Probation → Retirement |

### 기존 Ensemble과의 차이

| 비교 항목 | Ensemble (현재) | Orchestrator (신규) |
|----------|----------------|-------------------|
| 시그널 결합 방식 | 동일 심볼의 여러 전략 시그널을 **단일 값으로 합산** | 각 전략이 **독립 포지션** 보유, 심볼별로 넷팅 |
| 자본 배분 | 전략별 배분 없음 (시그널 가중치만) | 전략별 **독립 자본 슬롯** (capital_fraction) |
| P&L 추적 | 전체 포트폴리오 단위만 | **전략별 독립 P&L** + 전체 합산 |
| 전략 생애주기 | 없음 (수동 on/off) | INCUBATION → PRODUCTION → PROBATION → RETIRED |
| 리스크 관리 | 전체 포트폴리오 단일 SL/TS | **전략별 독립 리스크** + 전체 포트폴리오 리스크 |
| 동적 배분 | 정적 weight | Risk Parity + Adaptive Kelly + Degradation Guard |
| 사용 사례 | 동일 심볼에 여러 시그널 앙상블 | **서로 다른 심볼 세트**를 가진 독립 전략 동시 운용 |

> **Ensemble은 "같은 데이터에서 다른 관점을 합치는" 도구이고,
> Orchestrator는 "다른 전략들을 독립 사업부처럼 운영하는" 프레임워크**이다.

---

## 2. Architecture Overview

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Strategy Orchestrator                        │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │   Lifecycle    │  │   Capital      │  │     Risk          │  │
│  │   Manager      │  │   Allocator    │  │     Aggregator    │  │
│  │                │  │                │  │                   │  │
│  │ - state machine│  │ - Risk Parity  │  │ - Position Netting│  │
│  │ - graduation   │  │ - Adaptive     │  │ - Aggregate limits│  │
│  │ - degradation  │  │   Kelly        │  │ - Circuit breaker │  │
│  │ - retirement   │  │ - Rebalancing  │  │ - Correlation     │  │
│  └───────┬────────┘  └───────┬────────┘  └────────┬──────────┘  │
│          │                   │                     │             │
│  ┌───────▼───────────────────▼─────────────────────▼──────────┐  │
│  │                    Pod Manager                              │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │  │
│  │  │  Pod A  │  │  Pod B  │  │  Pod C  │  │  Pod D  │      │  │
│  │  │ TSMOM   │  │ Donchian│  │ VolAdapt│  │ VW-TSMOM│      │  │
│  │  │ BTC,ETH │  │ SOL,BNB │  │ BTC,SOL │  │ ALL     │      │  │
│  │  │ cap:30% │  │ cap:25% │  │ cap:20% │  │ cap:25% │      │  │
│  │  │ PnL:+5% │  │ PnL:+3% │  │ PnL:-1% │  │ PnL:+8% │      │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │  │
│  └───────┼─────────────┼───────────┼─────────────┼────────────┘  │
│          │             │           │             │               │
│  ┌───────▼─────────────▼───────────▼─────────────▼────────────┐  │
│  │                Position Netting Layer                       │  │
│  │                                                             │  │
│  │  Pod A: BTC +0.3, ETH +0.2                                 │  │
│  │  Pod C: BTC -0.1, SOL +0.15                                │  │
│  │  ─────────────────────────────                              │  │
│  │  Net:  BTC +0.2, ETH +0.2, SOL +0.15 ← 실제 거래소 주문    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼───────────────┐
              │     Existing EDA Layer      │
              │  EventBus → PM → RM → OMS  │
              │         → Executor          │
              └────────────────────────────┘
```

### 2.2 Data Flow (Event Sequence)

```
1. DataFeed emits BarEvent(symbol=BTC, tf=1D)
        │
2. Orchestrator routes to relevant Pods
        │
        ├── Pod A (TSMOM):  receives BTC bar → computes signal
        ├── Pod C (VolAdapt): receives BTC bar → computes signal
        │   (Pod B, D: BTC not in their symbol set → skip)
        │
3. Pod signals collected:
        │  Pod A: BTC target_weight = +0.30 (of Pod A's capital)
        │  Pod C: BTC target_weight = -0.10 (of Pod C's capital)
        │
4. Capital Allocator converts to global weights:
        │  Pod A capital = 30% of total → BTC global = +0.30 × 0.30 = +0.090
        │  Pod C capital = 20% of total → BTC global = -0.10 × 0.20 = -0.020
        │  Net BTC global weight = +0.070
        │
5. Position Netting:
        │  Current BTC position: +0.050
        │  Target: +0.070
        │  Delta: +0.020 → OrderRequest(BTC, BUY, +0.020)
        │
6. Standard EDA Pipeline:
        OrderRequest → RM validation → OMS → Executor → Fill
        │
7. Fill Attribution:
        Fill(BTC, +0.020) → allocate back to Pod A (+0.015) & Pod C (+0.005)
```

### 2.3 Integration with Existing EDA

**변경하지 않는 컴포넌트** (그대로 재사용):
- `EventBus` — 이벤트 라우팅
- `OMS` — 주문 관리 (idempotent)
- `ExecutorPort` / `BacktestExecutor` / `LiveExecutor` — 체결
- `AnalyticsEngine` — 성과 측정
- `DataFeedPort` / `HistoricalDataFeed` / `LiveDataFeed` — 데이터
- 모든 `BaseStrategy` 구현체 — 전략 로직

**새로 만드는 컴포넌트**:
- `StrategyOrchestrator` — 최상위 오케스트레이터
- `StrategyPod` — 전략별 독립 실행 단위
- `CapitalAllocator` — 자본 배분 엔진
- `LifecycleManager` — 전략 생애주기 관리
- `PositionNetter` — 포지션 넷팅
- `RiskAggregator` — 전략 간 리스크 통합
- `OrchestratorPM` — Orchestrator 전용 PM (기존 PM 래핑)

**수정하는 컴포넌트** (확장):
- `EDARunner` — `run_orchestrated()` 메서드 추가
- `LiveRunner` — Orchestrator 모드 지원
- `config_loader.py` — 멀티 전략 YAML 파싱
- CLI (`eda.py`) — `--orchestrator` 플래그

---

## 3. Phase 1: Core Models & Config

> **목표**: Orchestrator의 데이터 모델과 설정 구조 정의

### 3.1 새 파일: `src/orchestrator/models.py`

```python
"""Strategy Orchestrator 핵심 모델."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class LifecycleState(StrEnum):
    """전략 생애주기 상태."""
    INCUBATION = "incubation"    # 소규모 시범 운영 (5~10%)
    PRODUCTION = "production"    # 정상 운용 (동적 배분)
    WARNING = "warning"          # 열화 감지, 배분 50% 감축
    PROBATION = "probation"      # 최종 관찰기, 배분 25%로 고정
    RETIRED = "retired"          # 운용 중단


class AllocationMethod(StrEnum):
    """자본 배분 알고리즘."""
    EQUAL_WEIGHT = "equal_weight"           # 균등 배분
    RISK_PARITY = "risk_parity"             # ERC (Equal Risk Contribution)
    ADAPTIVE_KELLY = "adaptive_kelly"       # Risk Parity + Kelly overlay
    INVERSE_VOLATILITY = "inverse_volatility"  # 변동성 역비례


class RebalanceTrigger(StrEnum):
    """리밸런싱 트리거 방식."""
    CALENDAR = "calendar"        # 고정 주기 (weekly/daily)
    THRESHOLD = "threshold"      # PRC drift 초과 시
    HYBRID = "hybrid"            # calendar + threshold


@dataclass
class PodPerformance:
    """Pod별 성과 추적 (rolling window)."""
    pod_id: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    live_days: int = 0
    rolling_volatility: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    last_updated: str = ""


@dataclass
class PodPosition:
    """Pod별 포지션 (심볼 단위)."""
    pod_id: str
    symbol: str
    target_weight: float = 0.0       # Pod 내부 비중
    global_weight: float = 0.0       # 전체 포트폴리오 비중
    notional_usd: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
```

### 3.2 새 파일: `src/orchestrator/config.py`

```python
"""Orchestrator 설정 모델."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.orchestrator.models import AllocationMethod, RebalanceTrigger


class PodConfig(BaseModel):
    """개별 전략 Pod 설정."""
    model_config = ConfigDict(frozen=True)

    pod_id: str                              # 고유 식별자 (e.g., "pod-tsmom-btc")
    strategy_name: str                       # Registry 전략명 (e.g., "tsmom")
    strategy_params: dict[str, object] = {}  # 전략 파라미터
    symbols: list[str]                       # 거래 심볼 목록
    timeframe: str = "1D"                    # 타겟 타임프레임

    # 자본 배분
    initial_fraction: float = 0.10           # 초기 자본 비율 (10%)
    max_fraction: float = 0.40               # 최대 자본 비율 (40%)
    min_fraction: float = 0.02               # 최소 자본 비율 (2%)

    # Pod 레벨 리스크
    max_drawdown: float = 0.15               # Pod 최대 MDD (15%)
    drawdown_warning: float = 0.10           # 경고 임계 (10%)
    max_leverage: float = 2.0                # Pod 내 최대 레버리지

    # PM 설정 (Pod별 독립)
    system_stop_loss: float | None = 0.10
    use_trailing_stop: bool = False
    trailing_stop_atr_multiplier: float = 3.0
    rebalance_threshold: float = 0.05


class GraduationCriteria(BaseModel):
    """INCUBATION → PRODUCTION 승격 기준."""
    model_config = ConfigDict(frozen=True)

    min_live_days: int = 90              # 최소 90일 live
    min_sharpe: float = 1.0              # Annualized Sharpe >= 1.0
    max_drawdown: float = 0.15           # MDD <= 15%
    min_trade_count: int = 30            # 최소 30회 거래
    min_calmar: float = 0.8              # CAGR/MDD >= 0.8
    max_backtest_live_gap: float = 0.30  # Backtest-Live Sharpe 괴리 <= 30%
    max_portfolio_correlation: float = 0.50  # 기존 포트폴리오 상관 <= 0.5


class RetirementCriteria(BaseModel):
    """전략 퇴출 기준."""
    model_config = ConfigDict(frozen=True)

    # Hard stops (즉시)
    max_drawdown_breach: float = 0.25    # MDD > 25% → 즉시 RETIRED
    consecutive_loss_months: int = 6     # 6개월 연속 손실

    # Soft signals (WARNING → PROBATION → RETIRED)
    rolling_sharpe_floor: float = 0.3    # 6M Sharpe < 0.3 → WARNING
    probation_days: int = 30             # PROBATION 관찰 기간


class OrchestratorConfig(BaseModel):
    """Strategy Orchestrator 최상위 설정."""
    model_config = ConfigDict(frozen=True)

    # Pod 목록
    pods: list[PodConfig]

    # 배분 알고리즘
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY
    kelly_fraction: float = 0.25         # Fractional Kelly 계수
    kelly_confidence_ramp: int = 180     # Kelly 신뢰도 ramp-up 기간 (일)

    # 리밸런싱
    rebalance_trigger: RebalanceTrigger = RebalanceTrigger.HYBRID
    rebalance_calendar_days: int = 7     # Calendar: 7일마다
    rebalance_drift_threshold: float = 0.10  # Threshold: PRC 10% 초과

    # 전체 포트폴리오 리스크
    max_portfolio_volatility: float = 0.20   # 20% ann. vol
    max_portfolio_drawdown: float = 0.15     # 15% MDD
    max_gross_leverage: float = 3.0          # 총 gross exposure
    max_single_pod_risk_pct: float = 0.40    # 단일 Pod 리스크 기여 40% 이하
    daily_loss_limit: float = 0.03           # -3% 일간 손실 → 전체 중단

    # 생애주기 기준
    graduation: GraduationCriteria = Field(
        default_factory=GraduationCriteria
    )
    retirement: RetirementCriteria = Field(
        default_factory=RetirementCriteria
    )

    # 상관관계
    correlation_lookback: int = 90       # 상관관계 계산 기간 (일)
    correlation_stress_threshold: float = 0.70  # 평균 상관 > 0.7 → 경고

    # 비용 모델 (글로벌)
    cost_bps: float = 4.0                # 거래 비용 (bps)
```

### 3.3 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/__init__.py` | 패키지 초기화 |
| `src/orchestrator/models.py` | 데이터 모델 (LifecycleState, PodPerformance 등) |
| `src/orchestrator/config.py` | 설정 모델 (OrchestratorConfig, PodConfig 등) |
| `tests/orchestrator/test_models.py` | 모델 단위 테스트 |
| `tests/orchestrator/test_config.py` | 설정 유효성 검증 테스트 |

### 3.4 Estimated Tests: ~20

---

## 4. Phase 2: Capital Allocator Engine

> **목표**: Risk Parity + Adaptive Kelly 기반 자본 배분 알고리즘 구현

### 4.1 새 파일: `src/orchestrator/allocator.py`

핵심 클래스:

```python
class CapitalAllocator:
    """멀티 전략 자본 배분 엔진.

    3-Layer 배분:
    1. Base: Risk Parity (ERC) — 상관관계 기반
    2. Overlay: Adaptive Kelly — 성과 기반 조정
    3. Guard: Lifecycle state — 상태별 clamp
    """

    def __init__(self, config: OrchestratorConfig) -> None: ...

    def compute_weights(
        self,
        pod_returns: dict[str, pd.Series],  # pod_id → daily returns
        pod_states: dict[str, LifecycleState],
        lookback: int = 90,
    ) -> dict[str, float]:
        """전략별 자본 배분 비율 계산.

        Returns:
            pod_id → capital_fraction (합계 <= 1.0)
        """

    def _risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        risk_budgets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Equal Risk Contribution via Spinu convex optimization.

        Formula:
            minimize  Σ[-b_i * log(w_i)] + 0.5 * w^T * Σ * w
            s.t.      w_i >= 0, Σw_i = 1
        """

    def _adaptive_kelly_overlay(
        self,
        base_weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        confidence: float,  # 0.0 ~ 0.5
    ) -> np.ndarray:
        """Risk Parity 위에 Kelly overlay 적용.

        w_final = (1 - alpha) * w_rp + alpha * w_kelly
        alpha = min(confidence, kelly_fraction)
        """

    def _apply_lifecycle_clamps(
        self,
        weights: dict[str, float],
        pod_states: dict[str, LifecycleState],
        pod_configs: dict[str, PodConfig],
    ) -> dict[str, float]:
        """Lifecycle 상태에 따른 weight clamp.

        INCUBATION: max(initial_fraction)
        PRODUCTION: 동적 (min_fraction ~ max_fraction)
        WARNING:    현재의 50%
        PROBATION:  min_fraction
        RETIRED:    0.0
        """

    def _compute_confidence(
        self,
        live_days: int,
        ramp_days: int = 180,
    ) -> float:
        """Track record 기반 Kelly 신뢰도.

        0일 → 0.0 (순수 Risk Parity)
        180일 → kelly_fraction (최대)

        Formula: min(live_days / ramp_days, 1.0) * kelly_fraction
        """
```

### 4.2 Risk Parity 구현 세부

```python
def _risk_parity_weights(self, cov_matrix, risk_budgets=None):
    """
    Spinu (2013) convex formulation:

    f(w) = 0.5 * w^T * Σ * w - Σ[b_i * log(w_i)]

    이 함수는 unique global minimum을 가짐 (strictly convex).
    scipy.optimize.minimize(method="SLSQP") 사용.

    Fallback: 수렴 실패 시 inverse_volatility로 fallback.
    """
```

**Naive Risk Parity (Fallback)**:
```python
def _inverse_vol_weights(self, volatilities: np.ndarray) -> np.ndarray:
    """inv_vol_i / Σ(inv_vol_j) — 상관관계 무시 버전."""
```

### 4.3 Adaptive Kelly 세부

```python
def _adaptive_kelly_overlay(self, base_weights, mu, cov, confidence):
    """
    Step 1: Full Kelly — f* = Σ^{-1} × μ
    Step 2: Fractional — f_frac = fraction × f*
    Step 3: Risk constraint — if σ_p > max_vol, scale down
    Step 4: Long-only clamp — clip(0, None)
    Step 5: Blend — w = (1 - alpha) * w_rp + alpha * f_frac

    alpha = confidence × kelly_fraction
    confidence = min(live_days / ramp_days, 1.0)
    """
```

### 4.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/allocator.py` | CapitalAllocator 클래스 |
| `tests/orchestrator/test_allocator.py` | 배분 알고리즘 단위 테스트 |

### 4.5 Test Cases (~25)

- ERC: 2개 전략, 동일 변동성 → 5:5
- ERC: 변동성 2:1 → 저변동 전략에 더 많은 배분
- ERC: 상관관계 0.9 두 전략 → 큰 차이 없이 나눔 (diversification 낮음)
- ERC: 상관관계 -0.5 두 전략 → 역상관 전략에 더 많이
- Kelly: 양의 기대수익 → 비중 증가
- Kelly: 음의 기대수익 → 비중 0
- Adaptive: live_days=0 → 순수 Risk Parity
- Adaptive: live_days=180 → Kelly 최대 반영
- Lifecycle clamp: INCUBATION → initial_fraction 이하
- Lifecycle clamp: RETIRED → 0.0
- Lifecycle clamp: WARNING → 현재의 50%
- sum(weights) <= 1.0 항상 보장
- cov_matrix 특이 행렬 → fallback to inverse_vol

---

## 5. Phase 3: Strategy Pod & Orchestrator

> **목표**: 전략별 독립 실행 단위(Pod)와 최상위 오케스트레이터 구현

### 5.1 새 파일: `src/orchestrator/pod.py`

```python
class StrategyPod:
    """전략별 독립 실행 단위.

    각 Pod는:
    - 독립 BaseStrategy 인스턴스
    - 독립 심볼 세트
    - 독립 자본 슬롯 (capital_fraction)
    - 독립 P&L 추적
    - 독립 StrategyEngine (bar → signal 변환)
    """

    def __init__(
        self,
        config: PodConfig,
        strategy: BaseStrategy,
        capital_fraction: float,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.capital_fraction = capital_fraction
        self.state = LifecycleState.INCUBATION
        self.performance = PodPerformance(pod_id=config.pod_id)

        # Pod별 내부 포지션 추적
        self._positions: dict[str, PodPosition] = {}
        self._daily_returns: list[float] = []
        self._equity_curve: list[float] = []

    @property
    def pod_id(self) -> str: ...

    @property
    def symbols(self) -> list[str]: ...

    def accepts_symbol(self, symbol: str) -> bool:
        """이 Pod이 해당 심볼의 시그널을 처리하는지."""

    def compute_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> tuple[float, float]:
        """전략 실행 → (direction, strength) 반환.

        Pod 내부의 target_weight를 계산 (capital_fraction 미적용).
        """

    def update_position(
        self,
        symbol: str,
        fill_qty: float,
        fill_price: float,
        fee: float,
    ) -> None:
        """Fill 귀속 처리 → Pod P&L 업데이트."""

    def record_daily_return(self, daily_return: float) -> None:
        """일간 수익률 기록 (allocator용)."""

    def get_target_weights(self) -> dict[str, float]:
        """Pod 내부 심볼별 target weight 반환."""

    def get_global_weights(self) -> dict[str, float]:
        """capital_fraction 적용된 글로벌 weight 반환.

        global_weight[symbol] = internal_weight[symbol] * capital_fraction
        """
```

### 5.2 새 파일: `src/orchestrator/orchestrator.py`

```python
class StrategyOrchestrator:
    """멀티 전략 오케스트레이터.

    EventBus에 등록되어:
    1. BarEvent → 관련 Pod들에 라우팅
    2. Pod 시그널 수집 → Position Netting
    3. Net 포지션 → OrderRequest 생성
    4. FillEvent → Pod별 귀속 처리
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        pods: list[StrategyPod],
        allocator: CapitalAllocator,
        lifecycle_manager: LifecycleManager,
        netter: PositionNetter,
        risk_aggregator: RiskAggregator,
    ) -> None: ...

    async def register(self, bus: EventBus) -> None:
        """EventBus 구독.

        - BAR → _on_bar
        - FILL → _on_fill
        - BALANCE_UPDATE → _on_balance (daily rebalance check)
        """

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent 처리.

        Flow:
        1. bar.symbol → 관련 Pod 필터
        2. 각 Pod.compute_signal(symbol, df) 호출
        3. Pod별 global_weights 수집
        4. Position Netter → net weights 계산
        5. Net weight → SignalEvent 생성 (PM으로 전달)
        """

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent 처리 → Pod별 귀속.

        Fill attribution:
        각 Pod의 target_weight 비율로 fill을 분배.
        """

    async def _periodic_rebalance(self) -> None:
        """주기적 자본 재배분.

        1. Pod별 일간 수익률 수집
        2. CapitalAllocator.compute_weights()
        3. Pod.capital_fraction 업데이트
        4. LifecycleManager.evaluate() — 상태 전이 체크
        """

    def get_pod_summary(self) -> list[dict[str, object]]:
        """각 Pod 상태 요약 (모니터링/알림용)."""
```

### 5.3 핵심 설계: Signal → Order 변환

기존 PM의 `_on_signal`을 활용하되, Orchestrator가 **넷팅된 시그널**을 발행:

```
Pod A: BTC target = +0.30 (Pod A 자본의 30%)
Pod C: BTC target = -0.10 (Pod C 자본의 10%)

Orchestrator:
  Pod A global = +0.30 × 0.30 (30% allocation) = +0.090
  Pod C global = -0.10 × 0.20 (20% allocation) = -0.020
  Net BTC = +0.070

→ SignalEvent(symbol=BTC, strength=0.070, direction=LONG)
→ 기존 PM._on_signal() 처리
```

### 5.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/pod.py` | StrategyPod 클래스 |
| `src/orchestrator/orchestrator.py` | StrategyOrchestrator 메인 클래스 |
| `tests/orchestrator/test_pod.py` | Pod 단위 테스트 |
| `tests/orchestrator/test_orchestrator.py` | Orchestrator 통합 테스트 |

### 5.5 Estimated Tests: ~35

---

## 6. Phase 4: Lifecycle Manager & Degradation Detection

> **목표**: 전략 생애주기 자동 관리 + 실시간 열화 감지

### 6.1 새 파일: `src/orchestrator/lifecycle.py`

```python
class LifecycleManager:
    """전략 생애주기 상태 머신.

    State transitions:
        INCUBATION → PRODUCTION  (graduation criteria met)
        PRODUCTION → WARNING     (degradation detected)
        WARNING    → PRODUCTION  (recovery within 30 days)
        WARNING    → PROBATION   (no recovery)
        PROBATION  → PRODUCTION  (strong recovery)
        PROBATION  → RETIRED     (still degrading after 30 days)
        ANY        → RETIRED     (hard stop: MDD > 25%)
    """

    def __init__(
        self,
        graduation: GraduationCriteria,
        retirement: RetirementCriteria,
    ) -> None: ...

    def evaluate(
        self,
        pod: StrategyPod,
        portfolio_returns: pd.Series | None = None,
    ) -> LifecycleState:
        """현재 성과 기반 상태 전이 평가.

        Returns:
            새로운 LifecycleState (변경 없으면 현재 상태)
        """

    def _check_hard_stops(self, perf: PodPerformance) -> bool:
        """즉시 퇴출 조건 체크.

        - MDD > max_drawdown_breach (25%)
        - consecutive_loss_months >= 6
        """

    def _check_graduation(
        self,
        perf: PodPerformance,
        portfolio_returns: pd.Series | None,
    ) -> bool:
        """승격 조건 체크 (INCUBATION → PRODUCTION).

        All criteria must be met:
        - live_days >= 90
        - sharpe >= 1.0
        - max_drawdown <= 15%
        - trade_count >= 30
        - calmar >= 0.8
        - portfolio_correlation <= 0.5
        """

    def _check_degradation(self, perf: PodPerformance) -> bool:
        """열화 신호 체크 (PRODUCTION → WARNING).

        Uses Page-Hinkley test on rolling Sharpe.
        """
```

### 6.2 새 파일: `src/orchestrator/degradation.py`

```python
class PageHinkleyDetector:
    """Page-Hinkley 검정 기반 전략 열화 감지기.

    CUSUM variant로, 수익률의 평균 이동(mean shift)을 감지한다.
    누적 편차가 임계값(lambda)을 초과하면 열화 경보.

    Parameters:
        delta: 최소 감지 가능 변화량 (default: 0.005)
        lambda_: 감지 임계값 (default: 50.0)
        alpha: 망각 계수 (default: 0.9999)
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.9999,
    ) -> None: ...

    def update(self, value: float) -> bool:
        """새 관측값 업데이트 → 열화 감지 여부.

        Returns:
            True if degradation detected (m_t - M_t > lambda)
        """

    def reset(self) -> None:
        """상태 초기화 (WARNING → PRODUCTION 복귀 시)."""

    @property
    def score(self) -> float:
        """현재 PH score (m_t - M_t). 모니터링용."""
```

### 6.3 State Transition 상세

```
┌──────────────────────────────────────────────────────┐
│                 Lifecycle State Machine               │
│                                                       │
│   INCUBATION ──────graduation──────► PRODUCTION      │
│       │                                  │  ▲        │
│       │ hard_stop                        │  │        │
│       ▼                          degrade │  │recover │
│   RETIRED ◄──probation_expire──── PROBATION │        │
│       ▲                              ▲   │  │        │
│       │                              │   ▼  │        │
│       └────────hard_stop─────────  WARNING ──┘        │
│                                                       │
│  Capital fraction at each state:                      │
│    INCUBATION: initial_fraction (5~10%, 고정)          │
│    PRODUCTION: 동적 (min_fraction ~ max_fraction)      │
│    WARNING:    현재의 50% (즉시 감축)                    │
│    PROBATION:  min_fraction (최소 유지, 30일 관찰)       │
│    RETIRED:    0% (포지션 청산, Pod 비활성화)             │
└──────────────────────────────────────────────────────┘
```

### 6.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/lifecycle.py` | LifecycleManager 클래스 |
| `src/orchestrator/degradation.py` | PageHinkleyDetector 클래스 |
| `tests/orchestrator/test_lifecycle.py` | 상태 전이 테스트 |
| `tests/orchestrator/test_degradation.py` | PH 검정 테스트 |

### 6.5 Estimated Tests: ~30

---

## 7. Phase 5: Position Netting & Risk Aggregation

> **목표**: 여러 Pod의 포지션을 넷팅하고 전체 리스크를 통합 관리

### 7.1 새 파일: `src/orchestrator/netting.py`

```python
class PositionNetter:
    """Pod간 포지션 넷팅.

    여러 Pod이 동일 심볼에 대해 서로 다른 방향의 포지션을 가질 수 있다.
    실제 거래소에는 넷팅된 단일 포지션만 유지하여 마진 효율을 극대화한다.

    Example:
        Pod A: BTC +0.30, ETH +0.20
        Pod B: BTC -0.10, SOL +0.15
        Pod C: BTC +0.05, ETH -0.10
        ────────────────────────
        Net:   BTC +0.25, ETH +0.10, SOL +0.15
    """

    def compute_net_weights(
        self,
        pod_global_weights: dict[str, dict[str, float]],
        # pod_id → {symbol → global_weight}
    ) -> dict[str, float]:
        """심볼별 넷 글로벌 weight 계산.

        Returns:
            symbol → net_global_weight
        """

    def compute_deltas(
        self,
        net_targets: dict[str, float],
        current_positions: dict[str, float],
    ) -> dict[str, float]:
        """현재 → 목표 포지션 delta 계산.

        Returns:
            symbol → weight_delta (양수=매수, 음수=매도)
        """

    def attribute_fill(
        self,
        symbol: str,
        fill_qty: float,
        fill_price: float,
        fee: float,
        pod_targets: dict[str, float],
        # pod_id → target_weight for this symbol
    ) -> dict[str, tuple[float, float, float]]:
        """Fill을 Pod별로 귀속.

        각 Pod의 target_weight 비율로 fill을 분배.

        Returns:
            pod_id → (attributed_qty, attributed_price, attributed_fee)
        """
```

### 7.2 새 파일: `src/orchestrator/risk_aggregator.py`

```python
class RiskAggregator:
    """전략 간 리스크 통합 관리.

    Pod별 독립 리스크 + 포트폴리오 전체 리스크 이중 체크.
    """

    def __init__(self, config: OrchestratorConfig) -> None: ...

    def check_portfolio_limits(
        self,
        pod_performances: dict[str, PodPerformance],
        net_positions: dict[str, float],
        total_equity: float,
    ) -> list[RiskAlert]:
        """전체 포트폴리오 리스크 체크.

        Checks:
        1. Gross leverage <= max_gross_leverage (3.0x)
        2. Portfolio drawdown <= max_portfolio_drawdown (15%)
        3. Daily loss <= daily_loss_limit (3%)
        4. Single Pod PRC <= max_single_pod_risk_pct (40%)
        5. Effective N strategies >= 2 (HHI-based)
        """

    def compute_risk_contributions(
        self,
        pod_returns: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Pod별 Percentage Risk Contribution (PRC) 계산.

        PRC_i = w_i × (Σ × w)_i / σ_p²
        sum(PRC_i) = 1.0
        """

    def compute_effective_n(
        self,
        prc: dict[str, float],
    ) -> float:
        """Effective Number of Strategies (HHI 역수).

        HHI = Σ(PRC_i²)
        Effective_N = 1 / HHI
        """

    def check_correlation_stress(
        self,
        pod_returns: dict[str, pd.Series],
        threshold: float = 0.70,
    ) -> bool:
        """평균 전략 간 상관관계가 stress 수준인지.

        Returns:
            True if avg_correlation > threshold
        """
```

### 7.3 RiskAlert 모델

```python
@dataclass(frozen=True)
class RiskAlert:
    """리스크 경고 이벤트."""
    alert_type: str          # "gross_leverage", "drawdown", "daily_loss" 등
    severity: str            # "warning", "critical"
    message: str
    current_value: float
    threshold: float
    pod_id: str | None = None  # None이면 포트폴리오 전체
```

### 7.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/netting.py` | PositionNetter 클래스 |
| `src/orchestrator/risk_aggregator.py` | RiskAggregator 클래스 |
| `tests/orchestrator/test_netting.py` | 넷팅 로직 테스트 |
| `tests/orchestrator/test_risk_aggregator.py` | 리스크 통합 테스트 |

### 7.5 Estimated Tests: ~30

---

## 8. Phase 6: Runner Integration (Backtest + Live)

> **목표**: 기존 EDARunner/LiveRunner와 Orchestrator 통합

### 8.1 수정: `src/eda/runner.py`

```python
class EDARunner:
    # 기존 메서드 유지 (backtest, shadow, run)

    @classmethod
    def orchestrated(
        cls,
        config: OrchestratorConfig,
        data: MultiSymbolData,
        initial_capital: float = 100_000.0,
        fast_mode: bool = False,
    ) -> EDARunner:
        """Orchestrator 모드 백테스트.

        기존 backtest()와 동일한 EDA 파이프라인을 사용하되,
        StrategyEngine 대신 StrategyOrchestrator가 BAR→SIGNAL을 처리.
        """

    async def run_orchestrated(self) -> OrchestratedResult:
        """Orchestrator 백테스트 실행.

        Returns:
            OrchestratedResult: 전체 + Pod별 성과
        """
```

### 8.2 수정: `src/eda/live_runner.py`

```python
class LiveRunner:
    # 기존 classmethods 유지

    @classmethod
    def orchestrated_paper(
        cls,
        config: OrchestratorConfig,
        client: BinanceClient,
        initial_capital: float = 100_000.0,
        **kwargs,
    ) -> LiveRunner: ...

    @classmethod
    def orchestrated_live(
        cls,
        config: OrchestratorConfig,
        client: BinanceClient,
        futures_client: BinanceFuturesClient,
        initial_capital: float = 100_000.0,
        **kwargs,
    ) -> LiveRunner: ...
```

### 8.3 결과 모델

```python
@dataclass
class OrchestratedResult:
    """Orchestrator 백테스트 결과."""

    # 전체 포트폴리오
    portfolio_metrics: PerformanceMetrics
    portfolio_equity_curve: pd.Series

    # Pod별 성과
    pod_metrics: dict[str, PerformanceMetrics]
    pod_equity_curves: dict[str, pd.Series]

    # 배분 이력
    allocation_history: pd.DataFrame  # (time, pod_id, fraction)

    # 생애주기 이벤트
    lifecycle_events: list[dict[str, object]]

    # 리스크 기여도
    risk_contributions: pd.DataFrame  # (time, pod_id, PRC)
```

### 8.4 Backtest 실행 흐름

```
1. OrchestratorConfig → build all StrategyPod instances
2. Create single EventBus
3. Create single DataFeed (all symbols from all Pods)
4. Create StrategyOrchestrator (replaces StrategyEngine)
5. Create single PM (net positions only)
6. Create single RM, OMS, Executor
7. Register all to EventBus
8. Run DataFeed → EventBus loop
9. Orchestrator._on_bar():
   a. Route bar to relevant Pods
   b. Collect Pod signals
   c. Apply capital_fraction → global weights
   d. Net positions
   e. Publish net SignalEvent
10. Standard PM → RM → OMS → Fill flow
11. Orchestrator._on_fill(): attribute back to Pods
12. Periodic: CapitalAllocator rebalance
13. Periodic: LifecycleManager evaluate
14. Return OrchestratedResult
```

### 8.5 Deliverables

| 파일 | 내용 |
|------|------|
| `src/orchestrator/result.py` | OrchestratedResult 모델 |
| `src/eda/runner.py` (수정) | `orchestrated()`, `run_orchestrated()` 추가 |
| `src/eda/live_runner.py` (수정) | `orchestrated_paper()`, `orchestrated_live()` 추가 |
| `tests/orchestrator/test_backtest_integration.py` | E2E 백테스트 테스트 |
| `tests/orchestrator/test_live_integration.py` | Live 모드 mock 테스트 |

### 8.6 Estimated Tests: ~25

---

## 9. Phase 7: CLI & Config YAML

> **목표**: CLI 명령어와 YAML 설정 포맷 확장

### 9.1 YAML 설정 포맷

```yaml
# config/orchestrator-example.yaml

orchestrator:
  allocation_method: risk_parity    # equal_weight | risk_parity | adaptive_kelly
  kelly_fraction: 0.25
  kelly_confidence_ramp: 180

  rebalance:
    trigger: hybrid                 # calendar | threshold | hybrid
    calendar_days: 7
    drift_threshold: 0.10

  risk:
    max_portfolio_volatility: 0.20
    max_portfolio_drawdown: 0.15
    max_gross_leverage: 3.0
    max_single_pod_risk_pct: 0.40
    daily_loss_limit: 0.03

  graduation:
    min_live_days: 90
    min_sharpe: 1.0
    max_drawdown: 0.15
    min_trade_count: 30

  retirement:
    max_drawdown_breach: 0.25
    consecutive_loss_months: 6
    rolling_sharpe_floor: 0.3
    probation_days: 30

  correlation:
    lookback: 90
    stress_threshold: 0.70

pods:
  - pod_id: pod-tsmom-major
    strategy: tsmom
    params:
      lookback: 30
      vol_target: 0.35
    symbols: [BTC/USDT, ETH/USDT]
    timeframe: "1D"
    initial_fraction: 0.15
    max_fraction: 0.40
    min_fraction: 0.05
    risk:
      max_drawdown: 0.15
      max_leverage: 2.0
      system_stop_loss: 0.10
      use_trailing_stop: true
      trailing_stop_atr_multiplier: 3.0

  - pod_id: pod-donchian-alt
    strategy: donchian-ensemble
    params:
      lookbacks: [20, 60, 150]
    symbols: [SOL/USDT, BNB/USDT, AVAX/USDT]
    timeframe: "1D"
    initial_fraction: 0.10
    max_fraction: 0.30
    min_fraction: 0.05
    risk:
      max_drawdown: 0.15
      max_leverage: 1.5

  - pod_id: pod-voladapt-cross
    strategy: vol-adaptive
    params: {}
    symbols: [BTC/USDT, SOL/USDT]
    timeframe: "1D"
    initial_fraction: 0.10
    max_fraction: 0.35
    min_fraction: 0.05

backtest:
  start: "2024-01-01"
  end: "2025-12-31"
  capital: 100000

portfolio:
  cost_bps: 4.0
```

### 9.2 CLI 명령어

```bash
# Orchestrator 백테스트
uv run mcbot orchestrate backtest config/orchestrator-example.yaml
uv run mcbot orchestrate backtest config/orchestrator-example.yaml --report

# Orchestrator Paper Trading
uv run mcbot orchestrate paper config/orchestrator-example.yaml

# Orchestrator Live
uv run mcbot orchestrate live config/orchestrator-example.yaml

# Pod 상태 조회
uv run mcbot orchestrate status

# Pod별 성과 리포트
uv run mcbot orchestrate report --pod pod-tsmom-major
```

### 9.3 수정: `src/cli/eda.py` 또는 새 `src/cli/orchestrate.py`

```python
@app.command("orchestrate")
def orchestrate_group():
    """Multi-strategy orchestration commands."""

@orchestrate_group.command("backtest")
def orchestrate_backtest(
    config_path: Path,
    report: bool = False,
    fast: bool = False,
):
    """Run multi-strategy orchestrated backtest."""

@orchestrate_group.command("paper")
def orchestrate_paper(config_path: Path):
    """Run orchestrated paper trading."""

@orchestrate_group.command("live")
def orchestrate_live(config_path: Path):
    """Run orchestrated live trading."""
```

### 9.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/cli/orchestrate.py` | CLI 명령어 |
| `src/config/config_loader.py` (수정) | OrchestratorConfig YAML 파싱 |
| `config/orchestrator-example.yaml` | 예시 설정 |
| `tests/cli/test_orchestrate.py` | CLI 테스트 |
| `tests/config/test_orchestrator_config.py` | Config 파싱 테스트 |

### 9.5 Estimated Tests: ~15

---

## 10. Phase 8: Monitoring & Notification

> **목표**: Pod별 모니터링 메트릭 + Discord 알림 확장

### 10.1 Prometheus 메트릭 확장

```python
# src/monitoring/metrics.py (확장)

# Pod-level metrics
pod_equity = Gauge(
    "mcbot_pod_equity_usdt",
    "Pod equity in USD",
    ["pod_id", "strategy"],
)
pod_allocation = Gauge(
    "mcbot_pod_allocation_fraction",
    "Pod capital allocation fraction",
    ["pod_id"],
)
pod_sharpe = Gauge(
    "mcbot_pod_rolling_sharpe",
    "Pod rolling Sharpe ratio",
    ["pod_id"],
)
pod_drawdown = Gauge(
    "mcbot_pod_drawdown",
    "Pod current drawdown",
    ["pod_id"],
)
pod_lifecycle_state = Info(
    "mcbot_pod_lifecycle",
    "Pod lifecycle state",
    ["pod_id"],
)
pod_prc = Gauge(
    "mcbot_pod_risk_contribution",
    "Pod percentage risk contribution",
    ["pod_id"],
)

# Portfolio-level metrics
portfolio_effective_n = Gauge(
    "mcbot_portfolio_effective_n",
    "Effective number of strategies (1/HHI)",
)
portfolio_avg_correlation = Gauge(
    "mcbot_portfolio_avg_correlation",
    "Average inter-strategy correlation",
)
```

### 10.2 Discord 알림 확장

| 이벤트 | 채널 | Severity |
|--------|------|----------|
| Pod 승격 (INCUBATION → PRODUCTION) | alerts | INFO |
| Pod 열화 경고 (→ WARNING) | alerts | WARNING |
| Pod 관찰기 진입 (→ PROBATION) | alerts | WARNING |
| Pod 퇴출 (→ RETIRED) | alerts | CRITICAL |
| 자본 재배분 실행 | trade_log | INFO |
| 포트폴리오 리스크 초과 | alerts | CRITICAL |
| Daily Orchestrator Report | daily_report | INFO |

### 10.3 Daily Report 포맷

```
📊 Orchestrator Daily Report (2026-02-14)

Portfolio:
  Equity: $105,230  (+2.1%)
  Gross Leverage: 1.8x
  Effective Strategies: 3.2

Pod Performance:
  ┌─────────────────┬────────┬────────┬──────┬────────┐
  │ Pod             │ State  │ Alloc  │ PnL  │ Sharpe │
  ├─────────────────┼────────┼────────┼──────┼────────┤
  │ pod-tsmom-major │ PROD   │ 35.2%  │ +3.1%│  1.82  │
  │ pod-donchian-alt│ PROD   │ 28.1%  │ +1.5%│  1.21  │
  │ pod-voladapt    │ INCUB  │ 10.0%  │ +0.8%│  0.95  │
  │ pod-vw-tsmom    │ WARN   │ 12.5%  │ -1.2%│  0.42  │
  │ (unallocated)   │   -    │ 14.2%  │    - │    -   │
  └─────────────────┴────────┴────────┴──────┴────────┘

Risk:
  Portfolio DD: 3.2% / 15.0% limit
  Avg Correlation: 0.31
  Top PRC: pod-tsmom-major (38.5%)
```

### 10.4 Deliverables

| 파일 | 내용 |
|------|------|
| `src/monitoring/metrics.py` (수정) | Pod 메트릭 추가 |
| `src/notification/engine.py` (수정) | Orchestrator 이벤트 핸들러 추가 |
| `src/notification/formatters.py` (수정) | Daily report 포맷 |
| `tests/monitoring/test_orchestrator_metrics.py` | 메트릭 테스트 |
| `tests/notification/test_orchestrator_alerts.py` | 알림 테스트 |

### 10.5 Estimated Tests: ~15

---

## 11. Migration & Backward Compatibility

### 11.1 기존 기능 100% 호환

| 기존 사용법 | 변경 여부 |
|------------|----------|
| `mcbot eda run config.yaml` | 변경 없음 |
| `mcbot eda run-live config.yaml --mode paper` | 변경 없음 |
| `mcbot backtest run tsmom BTC/USDT` | 변경 없음 |
| EnsembleStrategy | 변경 없음 (독립 유지) |
| 단일 전략 + 멀티 심볼 | 변경 없음 |

### 11.2 신규 기능 추가 방식

- **새 CLI 그룹**: `mcbot orchestrate` (기존 `eda`와 분리)
- **새 Config 키**: `orchestrator:` 섹션 (기존 `strategy:` 키와 공존)
- **새 패키지**: `src/orchestrator/` (기존 코드 수정 최소화)
- **Runner 확장**: 기존 메서드 유지 + `orchestrated_*` 추가

### 11.3 Ensemble과의 관계

Ensemble은 **동일 심볼에 여러 시그널을 합산**하는 전략이고,
Orchestrator는 **독립 전략을 병렬 운영**하는 프레임워크이다.

**공존 가능**: Pod 내부의 전략이 EnsembleStrategy일 수 있다.

```yaml
pods:
  - pod_id: pod-ensemble-1
    strategy: ensemble          # Ensemble을 Pod으로 감쌈
    params:
      aggregation: inverse_volatility
    sub_strategies:
      - name: tsmom
      - name: donchian-ensemble
    symbols: [BTC/USDT, ETH/USDT]
    initial_fraction: 0.30
```

---

## 12. Risk & Constraints

### 12.1 기술적 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 넷팅 로직 오류 | 과다/과소 주문 | 넷팅 전후 invariant 검증 테스트 |
| Fill 귀속 오류 | Pod P&L 왜곡 | 합산 검증 (Pod 합계 = 실제 Fill) |
| 상관관계 추정 오류 | Risk Parity 편향 | Inverse Vol fallback |
| scipy 최적화 수렴 실패 | weight 미생성 | Equal Weight fallback |
| 동시 SL/TS 충돌 | 여러 Pod이 동시 청산 | 넷팅 후 단일 주문 |

### 12.2 운영 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| Pod 간 high correlation | 실질 분산 부족 | correlation_stress 경고 |
| 모든 Pod 동시 MDD | 포트폴리오 폭락 | daily_loss_limit 회로차단 |
| 전략 과다 (>10 Pods) | 자본 분산 과다 | max_pods 제한 |

### 12.3 제약 사항

- **Binance Hedge Mode 필수**: 동일 심볼 Long/Short 동시 보유를 위해
- **최소 자본**: Pod당 최소 $1,000 권장 (주문 최소 단위 고려)
- **연산 비용**: Risk Parity 최적화는 매 리밸런스마다 실행 (주 1회 → 무시 가능)

---

## 13. File Map

### 13.1 새 파일 (12개)

```
src/orchestrator/
├── __init__.py                 # 패키지 초기화
├── models.py                   # Phase 1: 데이터 모델
├── config.py                   # Phase 1: 설정 모델
├── allocator.py                # Phase 2: 자본 배분 엔진
├── pod.py                      # Phase 3: StrategyPod
├── orchestrator.py             # Phase 3: StrategyOrchestrator
├── lifecycle.py                # Phase 4: 생애주기 관리
├── degradation.py              # Phase 4: 열화 감지
├── netting.py                  # Phase 5: 포지션 넷팅
├── risk_aggregator.py          # Phase 5: 리스크 통합
└── result.py                   # Phase 6: 결과 모델

src/cli/
└── orchestrate.py              # Phase 7: CLI 명령어

config/
└── orchestrator-example.yaml   # Phase 7: 예시 설정
```

### 13.2 수정 파일 (5개)

```
src/eda/runner.py               # Phase 6: orchestrated() 추가
src/eda/live_runner.py          # Phase 6: orchestrated_paper/live() 추가
src/config/config_loader.py     # Phase 7: OrchestratorConfig 파싱
src/monitoring/metrics.py       # Phase 8: Pod 메트릭
src/notification/engine.py      # Phase 8: Orchestrator 알림
```

### 13.3 테스트 파일 (12개)

```
tests/orchestrator/
├── __init__.py
├── test_models.py              # Phase 1
├── test_config.py              # Phase 1
├── test_allocator.py           # Phase 2
├── test_pod.py                 # Phase 3
├── test_orchestrator.py        # Phase 3
├── test_lifecycle.py           # Phase 4
├── test_degradation.py         # Phase 4
├── test_netting.py             # Phase 5
├── test_risk_aggregator.py     # Phase 5
├── test_backtest_integration.py # Phase 6
└── test_live_integration.py    # Phase 6

tests/cli/
└── test_orchestrate.py         # Phase 7

tests/monitoring/
└── test_orchestrator_metrics.py # Phase 8

tests/notification/
└── test_orchestrator_alerts.py  # Phase 8
```

---

## 14. Test Strategy

### 14.1 단위 테스트 (~200 예상)

| Phase | 테스트 수 | 핵심 검증 |
|-------|----------|----------|
| 1. Models & Config | ~20 | 모델 직렬화, 유효성 검증, 기본값 |
| 2. Capital Allocator | ~25 | ERC, Kelly, fallback, clamp |
| 3. Pod & Orchestrator | ~35 | 시그널 라우팅, 넷팅, Fill 귀속 |
| 4. Lifecycle & Degradation | ~30 | 상태 전이, PH 검정 정확도 |
| 5. Netting & Risk | ~30 | 넷팅 정합성, PRC, HHI |
| 6. Runner Integration | ~25 | E2E 백테스트, Live mock |
| 7. CLI & Config | ~15 | YAML 파싱, 명령어 |
| 8. Monitoring & Notification | ~15 | 메트릭, Discord 알림 |

### 14.2 통합 테스트 핵심

```python
def test_orchestrated_backtest_two_pods():
    """2개 Pod (TSMOM + Donchian), 3개 심볼, 1년 백테스트.

    Invariants:
    1. sum(pod_equity) ≈ portfolio_equity (± 비용)
    2. net_positions = sum(pod_positions) per symbol
    3. all pod_fractions <= max_fraction
    4. sum(pod_fractions) <= 1.0
    """

def test_lifecycle_graduation_flow():
    """90일 이상 운용 후 INCUBATION → PRODUCTION 전환 확인."""

def test_degradation_retirement_flow():
    """의도적 열화 데이터 → WARNING → PROBATION → RETIRED 전환 확인."""

def test_netting_opposite_positions():
    """Pod A: BTC LONG, Pod B: BTC SHORT → 넷 포지션 정확 계산."""

def test_fill_attribution_proportional():
    """Fill 합계 = Pod별 귀속 합계 (원자성 검증)."""
```

### 14.3 Quality Gate

```bash
# 모든 Phase 완료 후 실행
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/orchestrator/
uv run pytest tests/orchestrator/ --cov=src/orchestrator --cov-report=term
# Coverage >= 90% 필수
```

---

## Implementation Order (Summary)

| Phase | 이름 | 의존성 | 예상 테스트 |
|-------|------|--------|-----------|
| **1** | Core Models & Config | 없음 | ~20 |
| **2** | Capital Allocator | Phase 1 | ~25 |
| **3** | Pod & Orchestrator | Phase 1, 2 | ~35 |
| **4** | Lifecycle & Degradation | Phase 1, 3 | ~30 |
| **5** | Position Netting & Risk | Phase 1, 3 | ~30 |
| **6** | Runner Integration | Phase 3, 4, 5 | ~25 |
| **7** | CLI & Config YAML | Phase 6 | ~15 |
| **8** | Monitoring & Notification | Phase 6 | ~15 |

**총 예상**: ~195 신규 테스트, 12 신규 파일, 5 수정 파일
