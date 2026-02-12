# EDA System Architecture

Phase 4: Event-Driven Architecture 시스템 설계 문서.
벡터화 백테스트(Phase 1-3)와 실거래(Phase 5-6) 사이의 브릿지 역할.

> **핵심 목표:** Backtest-Live Parity — 백테스트와 라이브에서 **동일한 전략/PM/RM/OMS 코드** 실행

---

## 1. 왜 EDA인가

### 벡터화 vs EDA 비교

| 항목 | Vectorized (Phase 1-3) | EDA (Phase 4+) |
|------|---|---|
| 데이터 처리 | 전체 DataFrame 한번에 | Bar 단위 이벤트 스트림 |
| 시그널 생성 | 전체 Series 한번에 | 각 Bar마다 개별 생성 |
| 리스크 체크 | 사후 분석 (weights에 규칙 적용) | **사전** 체크 (주문 전 검증) |
| 주문 체결 | VectorBT 내부 시뮬레이션 | OMS → Executor (모드별) |
| 코드 재사용 | 라이브 별도 구현 필요 | **라이브와 동일 코드** |
| 속도 | 매우 빠름 (Numba) | 느림 (이벤트 오버헤드) |
| 용도 | 파라미터 탐색, 대량 스윕 | 최종 검증, Shadow/Paper/Live |

**둘 다 유지.** Vectorized는 스윕/검증용, EDA는 라이브 전환용.

---

## 2. 아키텍처 개요

### 2.1 이벤트 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                        EventBus                              │
│  publish(event) → handlers[event_type] → [h1, h2, ...]      │
│  bounded queue (backpressure) + event log (audit trail)      │
└───┬──────┬──────┬──────┬──────┬──────┬──────────────────────┘
    │      │      │      │      │      │
┌───▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼───┐ ┌▼───┐ ┌▼────────┐
│Data  │ │Strat│ │ PM  │ │ RM │ │OMS │ │Analytics│
│Feed  │ │Eng  │ │     │ │    │ │    │ │Engine   │
└──────┘ └─────┘ └─────┘ └────┘ └────┘ └─────────┘
```

### 2.2 이벤트 체인

```
DataFeed ─→ BarEvent ─→ StrategyEngine ─→ SignalEvent ─→ PortfolioManager
                                                              │
                                                     OrderRequestEvent
                                                              │
                                                         RiskManager
                                                         (검증/거부)
                                                              │
                                                     OrderRequestEvent (validated)
                                                              │
                                                       OMS ──→ Executor
                                                              │
                                                         FillEvent
                                                              │
                                                    PositionUpdateEvent
                                                              │
                                                    BalanceUpdateEvent
```

### 2.3 핵심 설계 원칙

1. **Stateless Strategy / Stateful Execution** — 전략은 시그널만 생성, PM/RM/OMS가 상태 관리
2. **correlation_id** — MarketData → Signal → Order → Fill 인과 관계 추적
3. **멱등성** — `client_order_id`로 중복 주문 방지
4. **Fail-Safe** — PM → RM → OMS 3단계 방어, Kill Switch
5. **Look-Ahead Bias 차단** — Signal at Close → Execute at Next Open

---

## 3. 이벤트 타입 계층

### 3.1 Base Event

```python
class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None  # 인과 관계 추적
    source: str = ""                     # 발행 컴포넌트
```

### 3.2 이벤트 분류

| 도메인 | 이벤트 | 발행자 | 구독자 |
|--------|--------|--------|--------|
| **Market** | `BarEvent` | DataFeed | StrategyEngine, AnalyticsEngine |
| **Strategy** | `SignalEvent` | StrategyEngine | PortfolioManager, AnalyticsEngine |
| **Execution** | `OrderRequestEvent` | PortfolioManager | RiskManager |
| **Execution** | `OrderAckEvent` | OMS | AnalyticsEngine |
| **Execution** | `OrderRejectedEvent` | RiskManager / OMS | AnalyticsEngine |
| **Execution** | `FillEvent` | OMS (Executor) | PortfolioManager, AnalyticsEngine |
| **Portfolio** | `PositionUpdateEvent` | PortfolioManager | RiskManager, AnalyticsEngine |
| **Risk** | `RiskAlertEvent` | RiskManager | AnalyticsEngine |
| **Risk** | `CircuitBreakerEvent` | RiskManager | OMS (전량 청산) |
| **System** | `HeartbeatEvent` | 각 컴포넌트 | Monitor |

### 3.3 Crypto 특화 이벤트 (Phase 5+)

| 이벤트 | 용도 | 도입 시점 |
|--------|------|----------|
| `FundingRateEvent` | Perpetual 펀딩 비율 (8h 주기) | Phase 5 |
| `LiquidationEvent` | 대규모 청산 감지 (risk signal) | Phase 6 |
| `ReconnectEvent` | WebSocket 재연결 알림 | Phase 5 |

---

## 4. EventBus 설계

### 4.1 In-process Async EventBus

Phase 4-6은 단일 프로세스로 운용. In-process EventBus가 latency와 복잡도 면에서 최적.

```
선택: In-process async EventBus (자체 구현)
이유: Pydantic V2 네이티브, asyncio.Queue 기반, bounded backpressure
추후: 멀티 서비스 확장 시 Redis Streams로 교체 가능 (인터페이스 추상화)
```

### 4.2 핵심 기능

| 기능 | 설명 |
|------|------|
| **Type-safe subscribe** | `subscribe(EventType, handler)` — 타입별 핸들러 등록 |
| **Bounded queue** | `asyncio.Queue(maxsize=N)` — backpressure 보호 |
| **Event log** | 모든 이벤트 JSONL 기록 — audit trail + replay |
| **Backpressure** | timeout 후 드롭 + 메트릭 기록 |
| **Load shedding** | MarketData: stale 드롭 / Signal: 절대 드롭 금지 |

### 4.3 Backpressure 전략

| 이벤트 유형 | 드롭 정책 | 이유 |
|-------------|---------|------|
| `BarEvent` (market data) | 최신 것만 유지 (stale 드롭) | 오래된 시세 무의미 |
| `SignalEvent` | **절대 드롭 금지** (큐 넉넉히) | 돈이 걸린 결정 |
| `FillEvent` | **절대 드롭 금지** | 포지션 상태 정확성 |
| `HeartbeatEvent` | 드롭 가능 | 다음 heartbeat가 보충 |

---

## 5. 컴포넌트 상세

### 5.1 DataFeed

**역할:** 데이터 소스 추상화 — 히스토리컬/실시간 동일 인터페이스

| 구현 | 데이터 소스 | 사용 단계 |
|------|-----------|----------|
| `HistoricalDataFeed` | Silver Parquet 데이터 bar-by-bar 리플레이 | EDA Backtest |
| `LiveDataFeed` | WebSocket (CCXT Pro) 실시간 스트림 | Shadow/Paper/Live |

```
DataSource (Protocol)
  ├── HistoricalDataSource  # Silver 데이터 → BarEvent 순차 발행
  └── LiveDataSource        # WebSocket → BarEvent 실시간 발행
```

### 5.2 StrategyEngine

**역할:** BaseStrategy를 이벤트 기반으로 래핑 (bar-by-bar 실행)

- `BarEvent` 수신 → 내부 버퍼에 OHLCV 누적
- lookback 기간만큼 데이터가 쌓이면 `strategy.run()` 호출
- 시그널 변화 시 `SignalEvent` 발행
- **기존 BaseStrategy 코드 변경 없음** — 래퍼 패턴

### 5.3 PortfolioManager (Event-based)

**역할:** `SignalEvent` → `OrderRequestEvent` 변환 + Position Risk Management

- 현재 포지션 상태 관리 (FillEvent 구독)
- Signal strength × asset_weight → 목표 비중 계산
- 현재 vs 목표 비중 차이가 `rebalance_threshold` 이상이면 주문 생성
- `client_order_id` 생성 (멱등성)

#### Equity 계산

```python
total_equity = cash + long_notional - short_notional
```

- `long_notional = size * current_price` (시가평가 기준, unrealized PnL 이미 포함)
- SHORT 포지션은 반환 비용이므로 차감

#### Position Stop-Loss

- **Intrabar (`use_intrabar_stop=True`)**: LONG은 `bar.low`, SHORT는 `bar.high` 기준
- **Close 기준 (`use_intrabar_stop=False`)**: `bar.close` 기준
- 트리거: `check_price < entry * (1 - stop_loss_pct)` (LONG) / `> entry * (1 + stop_loss_pct)` (SHORT)

#### Trailing Stop (ATR 기반)

- **ATR(14)**: True Range의 14기간 SMA (incremental 계산)
- **Peak/Trough 추적**: LONG은 진입 후 `bar.high` 최고가, SHORT는 `bar.low` 최저가
- 트리거: `close < peak - atr * multiplier` (LONG) / `close > trough + atr * multiplier` (SHORT)
- Warmup: ATR 14봉 미달 시 trailing stop 비활성

#### 매 Bar BalanceUpdateEvent 발행

- 포지션 보유 중 매 bar마다 `BalanceUpdateEvent` 발행
- RM의 system stop-loss가 실시간 mark-to-market drawdown 추적 가능

### 5.4 RiskManager

**역할:** `OrderRequestEvent` 사전 검증 (NEW)

| 체크 항목 | 규칙 |
|----------|------|
| Max Leverage | aggregate leverage ≤ `max_leverage_cap` (2.0x) |
| System Stop-Loss | 전체 포트폴리오 손실 ≥ `system_stop_loss` (10%) → CircuitBreaker |
| Trailing Stop | ATR 기반 trailing stop 계산 → 포지션 청산 |
| Position Limit | 최대 동시 포지션 ≤ `max_open_positions` (8) |
| Order Size | 단일 주문 금액 ≤ `max_order_size_usd` |

통과 시 OMS로 전달, 거부 시 `OrderRejectedEvent` 발행.

### 5.5 OMS (Order Management System)

**역할:** `ExecutionMode`별 Executor 주입 (Strategy Pattern)

```python
class ExecutionMode(StrEnum):
    BACKTEST = "backtest"   # 히스토리컬 리플레이 + 시뮬레이션 체결
    SHADOW = "shadow"       # 실시간 데이터 + 시그널 로깅만 (주문 없음)
    PAPER = "paper"         # 실시간 데이터 + 시뮬레이션 체결
    CANARY = "canary"       # 실시간 데이터 + 실주문 (소액)
    LIVE = "live"           # 실시간 데이터 + 실주문 (전체 자본)
```

| Executor | 체결 방식 | Phase |
|----------|---------|-------|
| `BacktestExecutor` | 다음 Bar open 가격으로 즉시 체결 | Phase 4 |
| `ShadowExecutor` | 주문 로깅만 (체결 없음) | Phase 5-A |
| `PaperExecutor` | 시뮬레이션 체결 (슬리피지/수수료 모델링) | Phase 5-B |
| `LiveExecutor` | Binance Futures API 실주문 | Phase 5-C / 6 |

### 5.6 AnalyticsEngine

**역할:** 모든 이벤트 구독 → 실시간 메트릭 수집 → `BacktestResult` 생성

- 기존 `PerformanceMetrics` 스키마 재사용
- Trade 기록, PnL 시계열, 포지션 변화 추적
- EDA 백테스트 종료 시 `generate_result()` → 벡터화 결과와 비교 가능

---

## 6. 테스트 전략

### 6.1 Deterministic Event Replay

모든 이벤트를 JSONL로 기록 → 동일 시퀀스 재생 시 동일 결과 보장.

```
EventRecorder: 이벤트 → JSONL 파일 기록
EventReplayer: JSONL 파일 → EventBus 순차 발행 → 결과 비교
```

### 6.2 테스트 계층

| 레벨 | 대상 | 예시 |
|------|------|------|
| **Unit** | 개별 핸들러 격리 | SignalEvent → OrderRequestEvent 변환 검증 |
| **Integration** | 전체 이벤트 체인 | BarEvent → ... → FillEvent 흐름 |
| **Parity** | VBT vs EDA 결과 비교 | Sharpe/CAGR/MDD 오차 < 1% |
| **Property** | 불변량 검증 | 어떤 시그널이든 leverage ≤ max_leverage_cap |
| **Replay** | 기록된 이벤트 재생 | 장애 시나리오 재현 |

### 6.3 Parity Test (핵심)

EDA 백테스트 결과가 벡터화 백테스트와 일치하는지 자동 검증:

```
VBT run_multi() 결과 vs EDA run_eda() 결과
├── 양쪽 모두 결과 생성
├── 수익률 부호(양/음) 일치
├── 거래 수 유사 (±20%)
└── Warmup 미달 시 거래 0
```

#### 실제 Parity 결과 (BTC/USDT, 2024-01 ~ 2025-12)

| 지표 | VBT | EDA | 비고 |
|------|:---:|:---:|------|
| **Total Return** | 2.77% | 2.72% | 차이 0.05pp |
| **Sharpe Ratio** | 0.37 | 0.49 | 체결 방식 차이 |
| **MDD** | 12.32% | 12.34% | 거의 동일 |
| **Total Trades** | 21 | 20 | 1건 차이 |

> 체결 방식 차이: VBT는 시그널 시점의 close, EDA는 다음 bar의 open으로 체결.
> 수치 일치보다 **방향성과 규모**의 일치가 핵심.

---

## 7. Shadow / Paper / Canary 검증 단계

### 7.1 단계 정의

```
EDA Backtest ──→ Shadow Mode ──→ Paper Trading ──→ Canary ──→ Full Live
(Phase 4)        (Phase 5-A)     (Phase 5-B)      (Phase 5-C)  (Phase 6)
```

| 단계 | 데이터 소스 | 체결 방식 | 자본 | 위험도 |
|------|-----------|---------|------|--------|
| **EDA Backtest** | Silver Parquet (bar-by-bar) | 시뮬레이션 (next open) | 가상 | 0 |
| **Shadow** | **실시간 WebSocket** | **없음** (시그널 로깅만) | 없음 | 0 |
| **Paper** | **실시간 WebSocket** | **시뮬레이션** (슬리피지 모델) | 가상 | 0 |
| **Canary** | **실시간 WebSocket** | **실제 Binance 주문** | **소액** ($100-500) | 소액 |
| **Full Live** | 실시간 WebSocket | 실제 Binance 주문 | 전체 | 전체 |

### 7.2 Phase 4: EDA Backtest 검증 체크리스트

> **목표:** 벡터화 백테스트와 동일한 결과 재현

| # | 검증 항목 | 구체적 방법 | 통과 기준 |
|---|----------|-----------|----------|
| 1 | 결과 일치 | VBT 벡터화 vs EDA 결과 비교 | Sharpe/CAGR/MDD 오차 < 1% |
| 2 | 이벤트 체인 완결성 | BarEvent → ... → FillEvent 전체 흐름 | correlation_id로 추적 가능 |
| 3 | 포지션 정확성 | 매 bar마다 포지션/잔고 스냅샷 | VBT 결과와 매칭 |
| 4 | 리스크 규칙 | Stop-Loss/Trailing Stop/Rebalance | PM 규칙이 이벤트 기반에서도 동일 작동 |
| 5 | 멀티에셋 | 8-asset EW cash_sharing 시뮬레이션 | 기존 run_multi() 결과와 일치 |
| 6 | 에지 케이스 | 빈 시그널, 연속 같은 방향, 급등/급락 | 크래시 없이 처리 |

### 7.3 Phase 5-A: Shadow Mode 검증 체크리스트

> **목표:** 실시간 데이터에서 전략이 안정적으로 시그널 생성

| # | 검증 항목 | 구체적 방법 | 통과 기준 |
|---|----------|-----------|----------|
| 1 | 시그널 빈도 | 24시간 시그널 수 로깅 | 백테스트 일평균과 유사 |
| 2 | 시그널 방향 분포 | Long/Short/Flat 비율 분석 | 과도한 편향 없음 |
| 3 | Latency | WebSocket 수신 → 시그널 생성 시간 | P99 < 100ms |
| 4 | 데이터 무결성 | 수신 OHLCV vs Silver 데이터 | 누락/불일치 0건 |
| 5 | 재연결 안정성 | WebSocket 끊김 → 자동 복구 | 5분 내 자동 재연결 |
| 6 | 메모리 안정성 | 72시간 연속 운영 | 메모리 누수 없음 |
| 7 | **기간** | **최소 1~2주** | 다양한 시장 상황 경험 |

### 7.4 Phase 5-B: Paper Trading 검증 체크리스트

> **목표:** 시뮬레이션 체결의 현실성 + PnL 추적 정확성

| # | 검증 항목 | 구체적 방법 | 통과 기준 |
|---|----------|-----------|----------|
| 1 | 슬리피지 모델 | 시뮬레이션 체결가 vs 실제 시장가 | 백테스트 가정과 일치 |
| 2 | 수수료 계산 | CostModel의 fee/funding 적용 | Binance fee tier 정확 반영 |
| 3 | 포지션 추적 | 가상 포지션 변화 로깅 | 매 체결마다 정확한 업데이트 |
| 4 | PnL 정확성 | 일별/주별 PnL 리포트 | 수동 검산과 일치 |
| 5 | 리스크 규칙 발동 | Stop-loss/Trailing stop 트리거 | 실제 작동 확인 |
| 6 | 알림 시스템 | Discord 시그널/체결/에러 알림 | 정상 수신 |
| 7 | Shadow vs Paper 비교 | 동일 기간 결과 교차 검증 | PnL 패턴 유사 |
| 8 | **기간** | **최소 2~4주** | 월간 리밸런싱 1회 이상 경험 |

### 7.5 Phase 5-C: Canary 검증 체크리스트

> **목표:** 실제 거래소 API 연동 안정성 + 실제 슬리피지/수수료 측정

| # | 검증 항목 | 구체적 방법 | 통과 기준 |
|---|----------|-----------|----------|
| 1 | 주문 실행 | Binance Futures 실주문 | 정상 접수/체결 |
| 2 | 멱등성 | client_order_id 중복 방지 | 동일 주문 재전송 시 거부 |
| 3 | 실제 슬리피지 | 주문가 vs 체결가 | Paper 모델과 유사 |
| 4 | 실제 수수료 | 체결 내역의 fee 확인 | CostModel 예측 대비 ±10% |
| 5 | 레버리지 관리 | 실제 leverage 설정 | max_leverage_cap 준수 |
| 6 | Kill Switch | 비정상 시 자동 전량 청산 | 즉시 작동 |
| 7 | Paper vs Canary 비교 | 같은 기간 결과 비교 | PnL 패턴 유사 |
| 8 | **자본 규모** | $100~500 | 수수료 대비 의미 있는 수준 |
| 9 | **기간** | **최소 2~4주** | 안정적이면 자본 점진 증가 |

### 7.6 Phase 6: Go-Live 기준

| 기준 | 임계값 |
|------|-------|
| Paper/Canary 총 기간 | 최소 2개월 |
| 시스템 가동률 | > 99.5% |
| 시그널 일관성 (Shadow vs Backtest) | > 90% |
| 치명적 버그 | 0건 |
| Canary PnL | 양수 (시장 상황 고려하여 유연) |

### 7.7 자본 투입 스케줄 (Phase 5-C → 6)

```
Week 1-2:   $100~500 (최소 단위, 8에셋)
Week 3-4:   전체 자본의 5~10%
Month 2:    전체 자본의 20%
Month 3:    전체 자본의 50%
Month 4+:   전체 자본의 100%
```

매 단계 증액 전 확인: MDD 한도 내, Kill Switch 미발동, 슬리피지 모델 일치.

---

## 8. 프로젝트 구조

### 8.1 새로 생성할 파일

```
src/
├── core/
│   ├── events.py              ← BaseEvent + EventType enum + 이벤트 타입 계층
│   └── event_bus.py           ← async EventBus (bounded queue, backpressure)
├── eda/
│   ├── __init__.py
│   ├── data_feed.py           ← HistoricalDataFeed / LiveDataFeed (Protocol)
│   ├── strategy_engine.py     ← BaseStrategy → 이벤트 기반 래퍼
│   ├── portfolio_manager.py   ← SignalEvent → OrderRequestEvent
│   ├── risk_manager.py        ← 주문 사전 검증 (NEW)
│   ├── oms.py                 ← OMS + ExecutionMode
│   ├── executors.py           ← Backtest/Shadow/Paper/Live Executor
│   └── analytics.py           ← 실시간 메트릭 수집 → BacktestResult
├── models/
│   └── eda.py                 ← EDAConfig, ExecutionMode (Pydantic)
└── cli/
    └── eda.py                 ← Typer CLI (run-eda 커맨드)
```

### 8.2 기존 파일 수정 (최소)

| 파일 | 변경 | 이유 |
|------|------|------|
| `src/strategy/base.py` | 변경 없음 | StrategyEngine이 래핑 |
| `src/portfolio/config.py` | 변경 없음 | EDA PM이 설정 재사용 |
| `src/models/backtest.py` | 변경 없음 | AnalyticsEngine이 동일 스키마 출력 |

---

## 9. 구현 작업 목록

| # | 작업 | 난이도 | 의존성 |
|---|------|--------|--------|
| 1 | `BaseEvent` + 이벤트 타입 정의 (`src/core/events.py`) | 낮음 | - |
| 2 | `EventBus` 구현 (async, bounded queue, event log) | 중간 | #1 |
| 3 | `HistoricalDataFeed` — bar 단위 이벤트 리플레이 | 중간 | #2 |
| 4 | `StrategyEngine` — BaseStrategy 래퍼 (bar-by-bar) | 중간 | #2 |
| 5 | `PortfolioManager` 이벤트 기반 구현 | 높음 | #2 |
| 6 | `RiskManager` 구현 (leverage, stop-loss, position limit) | 높음 | #2 |
| 7 | `OMS` + `ExecutionMode` + Executor 패턴 | 높음 | #2 |
| 8 | `BacktestExecutor` — 즉시 체결 시뮬레이터 | 중간 | #7 |
| 9 | `AnalyticsEngine` — 메트릭 수집 + BacktestResult 생성 | 중간 | #2 |
| 10 | Parity Test — VBT vs EDA 결과 일치 검증 | 높음 | #1~9 |
| 11 | CLI `run-eda` 커맨드 | 낮음 | #10 |

---

## 10. 기술 선택

### 도입

| 기술 | 이유 |
|------|------|
| `asyncio.TaskGroup` (Python 3.13) | 구조화된 동시성, 예외 전파 보장 |
| `asyncio.Queue` (bounded) | in-process backpressure |
| JSONL event log | audit trail + deterministic replay |

### 도입하지 않는 것

| 기술 | 이유 |
|------|------|
| Redis Streams / Kafka | 단일 프로세스에서 과도한 인프라. Phase 7+ 재검토 |
| NautilusTrader | 프레임워크 전환 비용 > 이점. 자체 EDA가 유연성 높음 |
| Event Sourcing DB | EventBus + JSONL 로그로 충분. 규모가 커지면 재검토 |

### 참고 프로젝트

| 프로젝트 | 참고 포인트 |
|----------|-----------|
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | MessageBus 패턴, backtest-live parity 설계 |
| [bubus](https://github.com/browser-use/bubus) | Pydantic 기반 EventBus, WAL persistence |
| [hftbacktest](https://github.com/nkaz001/hftbacktest) | tick-by-tick 시뮬레이션, order book 모델링 |

---

## 11. EDA vs VBT 실전 비교 결과 (2026-02-07)

> 상세 분석: VBT vs EDA 비교는 위 표 참조.

### 통제 환경 Parity (SimpleMomentum)

| Metric | VBT | EDA | Delta |
|--------|:---:|:---:|:-----:|
| Total Return | 2.77% | 2.72% | -0.05pp |
| Trades | 21 | 20 | -1 |

**판정: PASS** — 통제 환경에서 parity 확보.

### 실전 전략 비교 (BTC/USDT 2024-2025)

| Strategy | Engine | Return | Trades | 이슈 |
|----------|--------|-------:|-------:|------|
| TSMOM | VBT / EDA | +47% / **-45%** | 95 / **3** | Signal dedup 과도 |
| Breakout | VBT / EDA | +16% / **+632%** | 23 / 20 | 수익률 이상치 |
| BB-RSI | VBT / EDA | -3% / -11% | 117 / 62 | 가장 유사 |

### Phase 4 → Phase 5 전환 판정: **NOT READY**

**필수 수정:** PM vol-target 리밸런싱, 실전 전략 parity 테스트 강화

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-02-06 | 초기 문서 작성 — EDA 아키텍처 설계, Shadow/Paper/Canary 검증 체크리스트 |
| 2026-02-06 | PM 강화 — Equity 이중 계산 수정, Position SL/Trailing Stop, 매 Bar BalanceUpdate, Parity 결과 추가 |
| 2026-02-07 | EDA vs VBT 실전 비교 — 4 전략 × 3 심볼 비교, parity 분석, 페이퍼 트레이딩 준비도 NOT READY 판정 |
