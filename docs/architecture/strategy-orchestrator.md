# Strategy Orchestrator Architecture

> **Multi-Strategy Portfolio Orchestration System**
>
> 여러 독립 전략을 동시에 실행하며, 성과 기반으로 자본을 동적 배분하고,
> 열화된 전략을 자동 축소/퇴출하는 프레임워크.

---

## 1. 핵심 개념

### Ensemble vs Orchestrator

```
┌─────────────────────────────────────────────────────────┐
│  Ensemble (기존)                                         │
│  "같은 데이터에서 다른 관점을 합치는 도구"                     │
│                                                          │
│  TSMOM ─┐                                                │
│  Donch  ─┼─→ 시그널 합산 → 단일 포지션                     │
│  VolAdp ─┘                                               │
│  (동일 심볼, 가중 평균, 단일 P&L)                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Orchestrator (신규)                                     │
│  "독립 전략들을 사업부처럼 운영하는 프레임워크"                  │
│                                                          │
│  Pod A ─→ BTC,ETH   (30% 자본, 독립 P&L)                 │
│  Pod B ─→ SOL,BNB   (25% 자본, 독립 P&L)                 │
│  Pod C ─→ BTC,SOL   (20% 자본, 독립 P&L)                 │
│           ────────────────────                           │
│           넷팅 → 실제 거래소 주문                           │
└─────────────────────────────────────────────────────────┘
```

| 비교 | Ensemble | Orchestrator |
|------|----------|-------------|
| 시그널 결합 | 단일 값으로 합산 | **독립 포지션**, 심볼별 넷팅 |
| 자본 배분 | 시그널 가중치만 | **전략별 독립 자본 슬롯** |
| P&L 추적 | 포트폴리오 단위 | **전략별 독립** + 전체 합산 |
| 생애주기 | 수동 on/off | 자동 승격/경고/퇴출 |
| 리스크 | 포트폴리오 단일 SL/TS | **전략별** + 포트폴리오 이중 관리 |
| 동적 배분 | 정적 weight | Risk Parity + Adaptive Kelly |

> **공존 가능** — Pod 내부의 전략이 EnsembleStrategy일 수 있다.

---

## 2. System Architecture

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

### 2.2 Module Dependency

```
orchestrator.py ─── 메인 오케스트레이터
  │
  ├── pod.py ─── 전략 래퍼 (BaseStrategy + 포지션/P&L 추적)
  │    ├── strategy.base.BaseStrategy
  │    ├── models.py (PodPerformance, PodPosition, LifecycleState)
  │    └── config.py (PodConfig)
  │
  ├── allocator.py ─── 4가지 자본 배분 알고리즘
  │    └── config.py (OrchestratorConfig)
  │
  ├── lifecycle.py ─── 5-state 자동 생애주기
  │    ├── degradation.py (PageHinkleyDetector)
  │    └── config.py (GraduationCriteria, RetirementCriteria)
  │
  ├── risk_aggregator.py ─── 포트폴리오 리스크 5-check
  │    └── netting.py (gross_leverage, scale_weights)
  │
  ├── netting.py ─── 포지션 넷팅 (Pure functions)
  │
  ├── state_persistence.py ─── 재시작 복구
  │    └── eda.persistence.database (Database)
  │
  ├── metrics.py ─── Prometheus 메트릭
  │
  └── result.py ─── 백테스트 결과 모델

                 ▲ Stateless (영속 불필요)
                 │ CapitalAllocator, RiskAggregator, PositionNetter
```

---

## 3. Data Flow

### 3.1 Bar → Order → Fill

```
1. DataFeed emits BarEvent(symbol=BTC, tf=1D)
        │
2. Orchestrator routes to relevant Pods
        │
        ├── Pod A (TSMOM):  receives BTC bar → signal
        ├── Pod C (VolAdapt): receives BTC bar → signal
        │   (Pod B, D: BTC not in their symbol set → skip)
        │
3. Pod signals collected:
        │  Pod A: BTC target_weight = +0.30 (of Pod A's capital)
        │  Pod C: BTC target_weight = -0.10 (of Pod C's capital)
        │
4. Capital Allocator converts to global weights:
        │  Pod A capital = 30% → BTC global = +0.30 × 0.30 = +0.090
        │  Pod C capital = 20% → BTC global = -0.10 × 0.20 = -0.020
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
        Fill(BTC, +0.020) → Pod A (+0.015) & Pod C (+0.005)
```

### 3.2 Event Sequence (EventBus)

```
                  ┌─────────────────────────────┐
                  │          EventBus            │
                  └──┬───┬───┬───┬───┬───┬──────┘
                     │   │   │   │   │   │
BarEvent ──────────→ │   │   │   │   │   │
                     │   │   │   │   │   │
              ┌──────▼┐  │   │   │   │   │
              │ Orch. │  │   │   │   │   │
              │_on_bar│  │   │   │   │   │
              └──┬────┘  │   │   │   │   │
                 │       │   │   │   │   │
SignalEvent ─────┼──────→│   │   │   │   │  (넷팅된 시그널)
                 │  ┌────▼┐  │   │   │   │
                 │  │ PM  │  │   │   │   │
                 │  └──┬──┘  │   │   │   │
                 │     │     │   │   │   │
OrderRequest ────┼─────┼────→│   │   │   │
                 │     │  ┌──▼─┐ │   │   │
                 │     │  │ RM │ │   │   │
                 │     │  └──┬─┘ │   │   │
                 │     │     │   │   │   │
ValidatedOrder ──┼─────┼─────┼──→│   │   │
                 │     │     │ ┌─▼─┐ │   │
                 │     │     │ │OMS│ │   │
                 │     │     │ └─┬─┘ │   │
                 │     │     │   │   │   │
FillEvent ───────┼─────┼─────┼───┼──→│   │
              ┌──▼────┐│     │   │ ┌─▼─┐ │
              │ Orch. ││     │   │ │PM │ │
              │_on_fill│     │   │ │upd│ │
              └───────┘│     │   │ └───┘ │
                       │     │   │       │
```

---

## 4. Strategy Pod

### 4.1 Pod = 독립 실행 단위

각 Pod는 하나의 `BaseStrategy`를 래핑하고, **독립적인 심볼 세트, 자본 슬롯, P&L**을 관리한다.

```
┌─────────────────────────────────┐
│          StrategyPod            │
│                                 │
│  ┌───────────┐  ┌────────────┐ │
│  │ BaseStrat │  │ Config     │ │
│  │ (TSMOM)   │  │ pod_id     │ │
│  │           │  │ symbols    │ │
│  │ run_incr()│  │ fractions  │ │
│  └─────┬─────┘  └────────────┘ │
│        │                       │
│  ┌─────▼─────────────────────┐ │
│  │ Internal State            │ │
│  │                           │ │
│  │ _state: LifecycleState    │ │
│  │ _capital_fraction: 0.30   │ │
│  │ _buffers: {sym: [OHLCV]} │ │
│  │ _target_weights: {sym: w} │ │
│  │ _positions: {sym: PodPos} │ │
│  │ _daily_returns: [r1,r2..] │ │
│  │ _performance: PodPerf     │ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

### 4.2 Signal Flow within a Pod

```
bar_data(OHLCV) ──→ _buffers 누적
                         │
                    warmup 충족?
                    │No        │Yes
                    ▼          ▼
                   skip    strategy.run_incremental()
                                │
                           target_weight 추출
                                │
                           _target_weights[symbol] = weight
```

### 4.3 Pod Configuration

```yaml
pods:
  - pod_id: pod-tsmom-major
    strategy: tsmom
    params:
      lookback: 30
      vol_target: 0.35
    symbols: [BTC/USDT, ETH/USDT]
    timeframe: "1D"
    initial_fraction: 0.15     # 초기 자본 비율
    max_fraction: 0.40         # 최대 허용
    min_fraction: 0.05         # 최소 유지
    risk:
      max_drawdown: 0.15
      max_leverage: 2.0
      system_stop_loss: 0.10
      use_trailing_stop: true
      trailing_stop_atr_multiplier: 3.0
```

---

## 5. Capital Allocator

### 5.1 3-Layer 배분 구조

```
Layer 1: Base Allocation          Layer 2: Kelly Overlay        Layer 3: Lifecycle Clamp
──────────────────────           ─────────────────────         ────────────────────────

 Risk Parity (ERC)               Blend with Kelly              State-based clamp
 ┌──────────────────┐           ┌──────────────────┐          ┌──────────────────┐
 │ min 0.5w'Σw      │           │ f* = Σ⁻¹μ       │          │ INCUBATION: ≤ini │
 │   - Σ bᵢlog(wᵢ) │    →      │ α = conf × frac  │    →     │ PRODUCTION: dyn  │
 │ s.t. w≥0, Σw=1  │           │ w=(1-α)rp+α·kelly│          │ WARNING:    ×0.5 │
 └──────────────────┘           └──────────────────┘          │ PROBATION:  =min │
                                                              │ RETIRED:    =0   │
      OR fallbacks:              confidence ramp:              └──────────────────┘
      - Inverse Vol              0일→0.0 (pure RP)
      - Equal Weight             180일→kelly_fraction
```

### 5.2 4가지 배분 알고리즘

| 알고리즘 | 방법 | 사용 시점 |
|----------|------|----------|
| **Equal Weight** | `1/N` | 초기 또는 데이터 부족 시 |
| **Inverse Volatility** | `(1/σᵢ) / Σ(1/σⱼ)` | Risk Parity fallback |
| **Risk Parity (ERC)** | Spinu convex optimization | 기본값 (상관관계 반영) |
| **Adaptive Kelly** | RP + Kelly blend | 충분한 track record 있을 때 |

### 5.3 Lifecycle Clamp

```
State         Capital Fraction Rule
─────────     ─────────────────────────────────────
INCUBATION    min(allocated, initial_fraction)       고정 상한
PRODUCTION    clip(allocated, min_frac, max_frac)    동적 범위
WARNING       allocated × 0.5, then clip             즉시 50% 감축
PROBATION     min_fraction                           최소 고정
RETIRED       0.0                                    청산
```

---

## 6. Lifecycle Manager

### 6.1 State Machine

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
│  Hard Stops (즉시 RETIRED):                            │
│    - MDD ≥ 25%                                        │
│    - 6개월 연속 손실                                     │
└──────────────────────────────────────────────────────┘
```

### 6.2 Graduation Criteria (INCUBATION → PRODUCTION)

**모든 조건을 동시에 만족**해야 승격:

| 기준 | 기본값 | 설명 |
|------|--------|------|
| `min_live_days` | 90 | 최소 운용 기간 |
| `min_sharpe` | 1.0 | 연환산 Sharpe ≥ 1.0 |
| `max_drawdown` | 0.15 | MDD ≤ 15% |
| `min_trade_count` | 30 | 최소 거래 횟수 |
| `min_calmar` | 0.8 | CAGR/MDD ≥ 0.8 |
| `max_portfolio_correlation` | 0.50 | 기존 포트폴리오 상관 ≤ 0.5 |

### 6.3 Degradation Detection (Page-Hinkley Test)

```
수익률 시계열 관찰
        │
        ▼
┌───────────────────────────────┐
│    Page-Hinkley Detector      │
│                               │
│  x̄ₜ = α·x̄ₜ₋₁ + (1-α)·xₜ     │  ← EWMA 평활 (α=0.99)
│  mₜ += x̄ₜ - xₜ - δ           │  ← 누적 편차
│  Mₜ = min(Mₜ, mₜ)            │  ← 최소값 추적
│                               │
│  if mₜ - Mₜ > λ → 열화 감지!  │  ← λ=50.0 임계값
└───────────────────────────────┘
        │
        ▼
  PRODUCTION → WARNING
```

- **δ (delta)**: 최소 감지 가능 변화량 (0.005)
- **λ (lambda)**: 감지 임계값 (50.0)
- **α (alpha)**: EWMA 평활 계수 (0.99, ~69일 반감기)

---

## 7. Position Netting & Risk

### 7.1 Netting

여러 Pod이 동일 심볼에 반대 포지션을 가질 수 있다.
실제 거래소에는 **넷팅된 단일 포지션**만 유지하여 마진 효율을 극대화한다.

```
Pod A: BTC +0.30, ETH +0.20
Pod B: BTC -0.10, SOL +0.15
Pod C: BTC +0.05, ETH -0.10
────────────────────────────
Net:   BTC +0.25, ETH +0.10, SOL +0.15  ← 실제 거래소 주문
```

### 7.2 Fill Attribution

Fill은 각 Pod의 target_weight 비율에 따라 **비례 귀속**된다:

```
Fill: BTC +0.020

Pod targets: Pod A = +0.30, Pod C = -0.10
BUY fill → long pods만 귀속
Pod A target = +0.30 (100%)
→ Pod A gets +0.020
```

### 7.3 Portfolio Risk 5-Check

| # | 검사 | 기본 임계값 | Severity |
|---|------|-----------|----------|
| 1 | Gross Leverage | ≤ 3.0x | Critical |
| 2 | Portfolio Drawdown | ≤ 15% | Critical |
| 3 | Daily Loss | ≤ 3% | Critical |
| 4 | Single Pod PRC | ≤ 40% | Warning/Critical |
| 5 | Correlation Stress | avg corr ≤ 0.70 | Warning/Critical |

- **Warning**: 임계값의 80% 도달
- **Critical**: 임계값 100% 초과 → `_risk_breached` 활성화 → 모든 weight 0 (방어 모드)

### 7.4 Risk Contribution (PRC)

```
PRC_i = w_i × (Σw)_i / σ²_p

Σ PRC_i = 1.0

Effective N = 1 / Σ(PRC_i²)    ← HHI 역수
```

---

## 8. State Persistence & Recovery

### 8.1 Persistence Coverage

```
┌──────────────────────────────────────────────────────┐
│              State Persistence Coverage               │
│                                                       │
│  EDA Layer:                                           │
│    ✅ PM: positions, cash, weights → bot_state JSON  │
│    ✅ RM: peak_equity, circuit_breaker → bot_state   │
│    ✅ OMS: processed_orders → bot_state              │
│                                                       │
│  Orchestrator Layer:                                  │
│    ✅ Pod: lifecycle_state, fraction, positions,     │
│           performance, target_weights                 │
│    ✅ Pod: daily_returns (별도 key, 270일 trim)       │
│    ✅ Lifecycle: PH detector, state_entered_at,      │
│                 consecutive_loss_months               │
│    ✅ Orchestrator: rebalance_ts, pod_targets        │
│    ── Stateless (영속 불필요) ──                      │
│    ○  CapitalAllocator (입력 기반 계산)               │
│    ○  RiskAggregator (입력 기반 계산)                 │
│    ○  PositionNetter (Pure function)                 │
└──────────────────────────────────────────────────────┘
```

### 8.2 Storage Layout

```
SQLite (bot_state key-value table)
├── "pm_state"                     ← PM 포지션/현금
├── "rm_state"                     ← RM peak_equity
├── "oms_processed_orders"         ← OMS 멱등성 set
├── "orchestrator_state"           ← Pod/Lifecycle/Orchestrator 전체
└── "orchestrator_daily_returns"   ← Pod별 수익률 이력 (270일)
```

### 8.3 Recovery Flow

```
프로그램 재시작
    │
    ├─ 1. SQLite 연결
    │
    ├─ 2. PM → RM → OMS 생성
    │
    ├─ 3. _restore_state(pm, rm, oms)
    │      ├─ PM: 포지션/현금/비중 복구
    │      ├─ RM: peak_equity/circuit_breaker 복구
    │      └─ OMS: processed_orders 복구
    │
    ├─ 4. Orchestrator 생성 (Pods, Allocator, Lifecycle 등)
    │
    ├─ 5. OrchestratorStatePersistence.restore()
    │      ├─ Pod별: lifecycle_state, capital_fraction,
    │      │         positions, performance 복구
    │      ├─ Pod별: daily_returns 복구
    │      ├─ Lifecycle: PH detector, state_entered_at 복구
    │      └─ Orchestrator: last_rebalance_ts, pod_targets 복구
    │
    ├─ 6. REST warmup (OHLCV buffers 재구성)
    │
    ├─ 7. EventBus 등록 + WebSocket 시작
    │
    └─ 8. 운용 재개 (이전 상태에서 연속)
```

### 8.4 Graceful Degradation

| 시나리오 | 동작 |
|----------|------|
| 저장 상태 없음 (첫 실행) | Config 기본값으로 시작 |
| Config에 Pod 추가 | 새 Pod → INCUBATION |
| Config에서 Pod 제거 | 저장 상태 무시 |
| JSON 파싱 실패 | 경고 로그 + 기본값 |
| Version mismatch | Graceful skip + 기본값 |

---

## 9. Strategy Discovery Pipeline

### 9.1 Gate Pipeline

```
G0A(주관) → IC Check → 구현 → G0B(코드검증) → G1(백테스트) → G2(IS/OOS) → G3(파라미터) → G4(통계) → G5(EDA)
```

### 9.2 전략 발굴 개선 체계

**문제**: 85개 전략 중 2개만 ACTIVE (성공률 2.4%).
G1 이후 검증은 견고하나, **발굴 초기 필터링이 느슨**하여 구현 비용이 낭비됨.

**5가지 개선**:

| # | 개선 | 효과 |
|---|------|------|
| 1 | **G0A 점수 검증** — 18점 미만 자동 FAIL | 첫 방어선 정상화 |
| 2 | **IC Quick Check** — 구현 전 지표 예측력 사전 검증 | G1 FAIL 60%의 절반 사전 차단 |
| 3 | **학술 근거 기록** — rationale_references YAML 필드 | 논거 중복 방지 |
| 4 | **RETIRED 패턴 분석** — 실패 사유 카테고리화 | 동일 실패 반복 방지 |
| 5 | **G0A v2 채점** — 데이터 기반 항목으로 교체 | 변별력 향상 (통과 기준 18→21점) |

**실패 패턴 분석 결과**:

| 패턴 | 비율 | 근본 원인 |
|------|:----:|----------|
| 신호 Alpha 부재 | 40% | 지표 예측력 없음 (IC ≈ 0) |
| 과적합 (OOS 붕괴) | 25% | 파라미터 IS에 과적합 |
| Timeframe 부적합 | 10% | 고주파 전략의 1D 왜곡 |
| 단일 레짐 의존 | 10% | 특정 시장 환경에서만 작동 |
| CAGR 기준 미달 | 10% | 거래 빈도 부족 |

**성공 전략 공통점**: 복합 신호 + 1D + HEDGE_ONLY + OOS Decay < 40%

---

## 10. Monitoring & Notification

### 10.1 Prometheus Metrics

**Pod-level:**

| Metric | Type | Description |
|--------|------|-------------|
| `mcbot_pod_equity_usdt` | Gauge | Pod 자본 (USD) |
| `mcbot_pod_allocation_fraction` | Gauge | 할당 비중 |
| `mcbot_pod_rolling_sharpe` | Gauge | Rolling Sharpe |
| `mcbot_pod_drawdown_pct` | Gauge | 현재 낙폭 |
| `mcbot_pod_risk_contribution` | Gauge | PRC |
| `mcbot_pod_lifecycle_state` | Enum | 생애주기 상태 |

**Portfolio-level:**

| Metric | Type | Description |
|--------|------|-------------|
| `mcbot_portfolio_effective_n` | Gauge | 유효 분산 수 (1/HHI) |
| `mcbot_portfolio_avg_correlation` | Gauge | 평균 전략 간 상관 |
| `mcbot_active_pods` | Gauge | 활성 Pod 수 |

### 10.2 Discord Alerts

| Event | Severity |
|-------|----------|
| Pod 승격 (INCUBATION → PRODUCTION) | INFO |
| Pod 열화 경고 (→ WARNING) | WARNING |
| Pod 관찰기 진입 (→ PROBATION) | WARNING |
| Pod 퇴출 (→ RETIRED) | CRITICAL |
| 자본 재배분 실행 | INFO |
| 포트폴리오 리스크 초과 | CRITICAL |

---

## 11. Configuration Reference

### 11.1 Full YAML Schema

```yaml
orchestrator:
  allocation_method: risk_parity    # equal_weight | risk_parity | adaptive_kelly | inverse_volatility
  kelly_fraction: 0.25              # Fractional Kelly 계수
  kelly_confidence_ramp: 180        # Kelly 신뢰도 ramp-up (일)

  rebalance:
    trigger: hybrid                 # calendar | threshold | hybrid
    calendar_days: 7                # Calendar 주기 (일)
    drift_threshold: 0.10           # Threshold: PRC drift 10% 초과 시

  risk:
    max_portfolio_volatility: 0.20  # 20% ann. vol
    max_portfolio_drawdown: 0.15    # 15% MDD
    max_gross_leverage: 3.0         # 총 gross exposure
    max_single_pod_risk_pct: 0.40   # 단일 Pod 리스크 기여 40% 이하
    daily_loss_limit: 0.03          # -3% 일간 손실 → 전체 중단

  graduation:
    min_live_days: 90
    min_sharpe: 1.0
    max_drawdown: 0.15
    min_trade_count: 30
    min_calmar: 0.8
    max_portfolio_correlation: 0.50

  retirement:
    max_drawdown_breach: 0.25       # MDD > 25% → 즉시 RETIRED
    consecutive_loss_months: 6      # 6개월 연속 손실 → 즉시 RETIRED
    rolling_sharpe_floor: 0.3       # 6M Sharpe < 0.3 → WARNING
    probation_days: 30              # PROBATION 관찰 기간

  correlation:
    lookback: 90                    # 상관관계 계산 기간 (일)
    stress_threshold: 0.70          # 평균 상관 > 0.7 → 경고

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

backtest:
  start: "2024-01-01"
  end: "2025-12-31"
  capital: 100000

portfolio:
  cost_bps: 4.0
```

---

## 12. File Map

```
src/orchestrator/
├── __init__.py              # 패키지 초기화
├── models.py                # LifecycleState, AllocationMethod, PodPerformance, PodPosition, RiskAlert
├── config.py                # OrchestratorConfig, PodConfig, GraduationCriteria, RetirementCriteria
├── allocator.py             # CapitalAllocator (EW, InvVol, Risk Parity, Adaptive Kelly)
├── pod.py                   # StrategyPod (전략 래퍼 + 독립 P&L)
├── orchestrator.py          # StrategyOrchestrator (EventBus 통합, 넷팅, 배치 처리)
├── lifecycle.py             # LifecycleManager (5-state machine, graduation, degradation)
├── degradation.py           # PageHinkleyDetector (CUSUM variant 열화 감지)
├── netting.py               # compute_net_weights, attribute_fill (Pure functions)
├── risk_aggregator.py       # RiskAggregator (PRC, Effective N, 5-check)
├── result.py                # OrchestratedResult (백테스트 결과 모델)
├── state_persistence.py     # OrchestratorStatePersistence (JSON save/restore)
└── metrics.py               # OrchestratorMetrics (Prometheus gauges)
```
