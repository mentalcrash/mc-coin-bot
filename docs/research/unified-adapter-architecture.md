# 통합 어댑터 아키텍처 리서치: VBT/EDA/Paper/Shadow/Live 코드 통일

> **Research Date**: 2026-02-08
> **Scope**: Nautilus Trader 어댑터 패턴 분석 + MC Coin Bot 통합 아키텍처 설계
> **Objective**: 백테스트 → 페이퍼 → 섀도우 → 카나리 → 라이브까지 코드 변경 최소화

---

## 목차

1. [문제 정의: 현재 아키텍처의 코드 중복](#1-문제-정의)
2. [업계 프레임워크 비교 분석](#2-업계-프레임워크-비교-분석)
3. [핵심 패턴: Ports & Adapters](#3-핵심-패턴-ports--adapters)
4. [MC Coin Bot 현재 구조 분석](#4-mc-coin-bot-현재-구조-분석)
5. [제안: 통합 어댑터 아키텍처](#5-제안-통합-어댑터-아키텍처)
6. [구체적 리팩토링 계획](#6-구체적-리팩토링-계획)
7. [VBT와의 관계 정리](#7-vbt와의-관계-정리)
8. [운영 모드별 배포 전략](#8-운영-모드별-배포-전략)
9. [Trade-offs 및 결론](#9-trade-offs-및-결론)
10. [References](#10-references)

---

## 1. 문제 정의

### 1.1 현재 고통점

MC Coin Bot은 **두 개의 독립적인 백테스팅 시스템**이 병렬로 존재한다:

```
VBT Path (Vectorized)                    EDA Path (Event-Driven)
━━━━━━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━━━━━
BacktestEngine (1206 lines)               8개 컴포넌트 (2300+ lines)
  apply_stop_loss_to_weights (Numba)        PM._on_bar → stop-loss check
  apply_trailing_stop_to_weights (Numba)    PM._update_trailing_stop
  apply_rebalance_threshold_numba           PM._evaluate_rebalance
  _apply_pm_rules_to_weights                RM._on_order_request
  PerformanceAnalyzer                       AnalyticsEngine
```

**문제점:**
1. **리스크 규칙 이중 구현**: SL/TS/rebalance가 VBT(Numba 배열 연산)와 EDA(이벤트 기반 상태 추적)에 각각 구현
2. **결과 불일치**: VBT vs EDA 약 2~4% Sharpe 차이 (체결 타이밍, 상태 관리 방식 차이)
3. **라이브 전환 불가**: VBT는 동기/배치 처리, EDA는 비동기/이벤트 처리 → 라이브에 VBT 사용 불가
4. **테스트 부담 배가**: 동일 로직에 대해 두 세트의 테스트 유지 필요
5. **CLI 분리**: `src/cli/backtest.py` (VBT) vs `src/cli/eda.py` (EDA) 별도 진입점

### 1.2 공유되는 것 vs 중복되는 것

| Layer | 공유 (✅) | 중복 (❌) |
|-------|---------|---------|
| **전략 코드** | ✅ `BaseStrategy.run()` | — |
| **비용 모델** | ✅ `CostModel` | — |
| **설정** | ✅ `PortfolioManagerConfig` | — |
| **결과 모델** | ✅ `PerformanceMetrics`, `TradeRecord` | — |
| **Stop-Loss** | — | ❌ Numba 배열 vs PM 이벤트 |
| **Trailing Stop** | — | ❌ Numba 배열 vs PM ATR 추적 |
| **Rebalance** | — | ❌ Numba threshold vs PM should_rebalance |
| **포지션 추적** | — | ❌ VBT 내부 vs PM Position dataclass |
| **성과 분석** | — | ❌ PerformanceAnalyzer vs AnalyticsEngine |
| **데이터 공급** | — | ❌ 일괄 DataFrame vs HistoricalDataFeed |
| **CLI** | — | ❌ 2개의 분리된 CLI |

### 1.3 목표

```
Before: 5가지 모드에 각각 다른 코드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VBT Backtest    → engine.py (Numba)
EDA Backtest    → 8 EDA modules
Paper Trading   → 미구현
Shadow Trading  → ShadowExecutor만 존재
Live Trading    → 미구현

After: 1개의 EDA 코어 + 교체 가능한 어댑터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDA Core (공통) + BacktestAdapter  = VBT-parity 백테스트
EDA Core (공통) + PaperAdapter     = 실시간 페이퍼 트레이딩
EDA Core (공통) + ShadowAdapter    = 라이브 시그널 비교
EDA Core (공통) + CanaryAdapter    = 소규모 실전
EDA Core (공통) + LiveAdapter      = 프로덕션
VBT Engine (독립) = 빠른 파라미터 스크리닝 (Phase 1 전용)
```

---

## 2. 업계 프레임워크 비교 분석

### 2.1 NautilusTrader — Hexagonal Architecture (가장 관련성 높음)

**핵심 설계:**
```
NautilusKernel (공통 코어)
├── MessageBus       → 이벤트 통신 (Pub/Sub + Req/Rep)
├── DataEngine       → 데이터 정규화 + 배포
├── ExecutionEngine  → 주문 라우팅
├── RiskEngine       → 사전 거래 검증
├── Portfolio        → 포지션/잔고 추적
└── Cache            → 고성능 상태 저장소

BacktestNode          TradingNode
├── TestClock         ├── LiveClock
├── Historical Data   ├── WebSocket Data
└── SimulatedExchange └── Live Exchange API
```

**모드 전환 방법:** 전략 코드는 변경 없이, `BacktestNode` ↔ `TradingNode`만 교체.

**어댑터 인터페이스 (2가지):**
```python
# Data 쪽
class DataClient(Protocol):
    async def _connect(self) -> None: ...
    async def _subscribe(self, command: SubscribeData) -> None: ...

# Execution 쪽
class ExecutionClient(Protocol):
    async def _submit_order(self, command: SubmitOrder) -> None: ...
    async def _cancel_order(self, command: CancelOrder) -> None: ...
```

**핵심 통찰:**
- 전략은 `self.submit_order()`만 호출 — SimulatedExchange로 갈지, Binance API로 갈지 모름
- Clock 추상화: `TestClock` (결정적 시간) vs `LiveClock` (실시간)
- Sandbox 모드: 실시간 데이터 + 가상 체결 (paper trading 완벽 지원)

### 2.2 LEAN (QuantConnect) — Handler-Based DI

**핵심 설계:**
```
Engine.Run()
├── ISetupHandler       → Backtest / Live 초기화
├── IDataFeed           → FileSystemDataFeed / LiveTradingDataFeed
├── ITransactionHandler → BacktestingTxHandler / BrokerageTxHandler
├── IBrokerage          → BacktestingBrokerage / LiveBrokerage
└── IResultHandler      → BacktestingResult / LiveResult
```

**모드 전환:** `config.json`의 `"environment"` 필드로 전환:
```json
{"environment": "backtesting"}  // or "live-paper", "live-interactive"
```

**핵심 통찰:**
- 10년 이상 프로덕션 검증된 가장 성숙한 시스템
- **AlgorithmManager**가 매 TimeSlice마다 동일 파이프라인 실행 (가격 업데이트 → OnData → 체결)
- 백테스트/라이브 모두 동일한 `IAlgorithm` 인터페이스

### 2.3 Backtrader — Store Model

**핵심 설계:**
```
Cerebro (orchestrator)
├── Strategy.next()      → 사용자 로직
├── BackBroker / IBBroker → 교체 가능한 브로커
├── CSVData / IBData      → 교체 가능한 데이터
└── Store (IBStore)       → broker + data 생성 팩토리
```

**핵심 통찰:**
- Store Model이 broker와 data를 한 번에 교체하는 팩토리 패턴
- 라이브 전환이 자연스러우나, vectorized 대비 느림
- 2020년 이후 메인테이너 부재 (레거시)

### 2.4 VectorBT Pro — Callback + External Bridge

**핵심 설계:**
```
Portfolio.from_order_func()
├── order_func (Numba callback per bar)
├── StrateQueue (외부 라이브 브릿지)
└── Context (현재 시뮬레이션 상태)
```

**핵심 통찰:**
- 파라미터 스윕에서 **압도적 속도** (Numba JIT vectorization)
- 라이브 전환은 네이티브가 아님 (StrateQueue 외부 의존)
- `from_order_func`의 event-driven 모드가 있으나 제한적

### 2.5 종합 비교

| Feature | NautilusTrader | LEAN | Backtrader | VectorBT Pro |
|---------|:---:|:---:|:---:|:---:|
| **코드 통일** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| **Backtest 속도** | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| **Crypto 지원** | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ |
| **커스터마이징** | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| **Python 네이티브** | ★★★★☆ | ★☆☆☆☆ (C#) | ★★★★★ | ★★★★★ |
| **유지보수** | ★★★★☆ | ★★★★★ | ★☆☆☆☆ | ★★★★☆ |

**결론:** NautilusTrader의 Ports & Adapters 패턴이 MC Coin Bot에 가장 적합한 참조 모델.

---

## 3. 핵심 패턴: Ports & Adapters

### 3.1 패턴 개요 (Hexagonal Architecture)

```
                    ┌─────────────────────────────────┐
                    │        Application Core          │
                    │  (Business Logic - 변경 없음)     │
                    │                                   │
                    │  Strategy → PM → RM → OMS         │
                    │        EventBus                    │
                    └──────────┬────────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                    │
         ┌────▼────┐    ┌─────▼─────┐    ┌────────▼────────┐
         │  Port:   │    │  Port:    │    │  Port:          │
         │  Data    │    │  Execution│    │  Analytics      │
         │  Feed    │    │  Client   │    │  Output         │
         └────┬────┘    └─────┬─────┘    └────────┬────────┘
              │               │                    │
     ┌────────┴──┐    ┌──────┴─────┐    ┌────────┴────────┐
     │ Adapters: │    │ Adapters:  │    │ Adapters:       │
     │ Historical│    │ Backtest   │    │ JSONL Logger    │
     │ Live WS   │    │ Paper      │    │ Dashboard       │
     │ Replay    │    │ Shadow     │    │ Prometheus      │
     └───────────┘    │ Live       │    └─────────────────┘
                      └────────────┘
```

### 3.2 핵심 원칙

1. **인터페이스 의존**: 코어는 추상 Port에만 의존, 구체 Adapter를 모름
2. **어댑터 교체**: 런타임 설정으로 어댑터만 교체 → 동일 코어 코드
3. **단방향 의존**: Core ← Ports ← Adapters (의존성 역전)
4. **이벤트 기반**: 모든 컴포넌트 간 통신은 이벤트 (BarEvent, SignalEvent, FillEvent 등)

### 3.3 NautilusTrader에서 배운 핵심 추상화

| 추상화 | NautilusTrader | MC Coin Bot 대응 |
|--------|---------------|-----------------|
| **DataClient** | `LiveDataClient` / `BacktestDataClient` | `DataFeed` Protocol |
| **ExecutionClient** | `LiveExecutionClient` / `SimulatedExchange` | `Executor` Protocol |
| **Clock** | `LiveClock` / `TestClock` | 새로 추가 필요 |
| **MessageBus** | MessageBus (Pub/Sub + Req/Rep) | EventBus (이미 있음) |
| **Cache** | In-memory state store | 새로 추가 필요 (선택) |

---

## 4. MC Coin Bot 현재 구조 분석

### 4.1 이미 갖춰진 어댑터 패턴

MC Coin Bot의 EDA는 이미 어댑터 패턴의 **핵심 인프라를 보유**하고 있다:

**1. Executor Protocol** (`src/eda/oms.py`):
```python
@runtime_checkable
class Executor(Protocol):
    async def execute(self, order: OrderRequestEvent) -> FillEvent | None: ...
```
→ NautilusTrader의 `ExecutionClient`와 동일한 역할

**2. BacktestExecutor + ShadowExecutor** (`src/eda/executors.py`):
→ 이미 2개 어댑터 구현체 존재

**3. EventBus** (`src/core/event_bus.py`):
→ NautilusTrader의 MessageBus와 동일 개념 (Pub/Sub, flush, backpressure)

**4. DataFeed 패턴** (`src/eda/data_feed.py`):
→ HistoricalDataFeed, AggregatingDataFeed가 암묵적 인터페이스 따름

### 4.2 누락된 추상화

| 누락 항목 | 필요성 | 비고 |
|----------|--------|------|
| **DataFeed Protocol** | 높음 | 현재 암묵적 → 명시적 Protocol 필요 |
| **LiveDataFeed** | Phase 8 | WebSocket → BarEvent |
| **LiveExecutor** | Phase 8 | CCXT Pro → FillEvent |
| **Clock 추상화** | 중간 | 백테스트 시간 vs 실시간 |
| **Reconciliation** | Phase 8 | 거래소 ↔ 내부 상태 동기화 |
| **통합 CLI** | 낮음 | 2개 CLI 통합 |
| **통합 Metrics** | 중간 | PerformanceAnalyzer와 AnalyticsEngine 통합 |

### 4.3 VBT와 EDA의 구체적 코드 중복 매핑

```
VBT (src/backtest/engine.py)              EDA (src/eda/)
━━━━━━━━━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━
L46-110: apply_stop_loss_to_weights       portfolio_manager.py L300-350: _on_intrabar
         (Numba, 65 lines)                 (stateful position check)

L112-210: apply_trailing_stop_to_weights  portfolio_manager.py L350-400: _update_trailing_stop
          (Numba, 98 lines)                (ATR incremental, peak tracking)

L212-231: apply_rebalance_threshold_numba portfolio_manager.py L280-300: _evaluate_rebalance
          (Numba, 19 lines)                (threshold comparison)

L1066-1120: _apply_pm_rules_to_weights    portfolio_manager.py: 위 3개 조합
            (orchestrator, 52 lines)

L462-617: _create_portfolio_from_orders   oms.py + executors.py: 주문 실행
          (VBT Portfolio 생성)

analyzer.py: PerformanceAnalyzer          analytics.py: AnalyticsEngine
             (VBT portfolio → metrics)     (event stream → metrics)
```

**총 중복 코드:** ~500 lines (리스크 규칙) + ~300 lines (성과 분석) = ~800 lines

---

## 5. 제안: 통합 어댑터 아키텍처

### 5.1 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Strategy Code                           │
│  BaseStrategy.run(df) → (processed_df, StrategySignals)            │
│  ※ VBT와 EDA에서 동일 코드 재사용 (이미 달성)                          │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────┐
│                       EDA Core (변경 없음)                           │
│                                                                     │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Strategy │→│ Portfolio   │→│ Risk         │→│ OMS        │  │
│  │ Engine   │  │ Manager     │  │ Manager      │  │            │  │
│  └──────────┘  └─────────────┘  └──────────────┘  └─────┬──────┘  │
│       ↑              ↑                                    │         │
│       │         EventBus (이벤트 라우팅)                    │         │
│       │              ↑                                    │         │
│  ┌────┴──────┐  ┌───┴────────┐                    ┌──────▼──────┐  │
│  │ Analytics │  │            │                    │             │  │
│  │ Engine    │  │            │                    │             │  │
│  └───────────┘  │            │                    │             │  │
│                 │            │                    │             │  │
│  ═══════════════╪════════════╪════════════════════╪═════════════╪══│
│   Port:         │ DataFeed   │  Port:             │  Executor   │  │
│   Protocol      │ Protocol   │  Protocol          │  Protocol   │  │
└─────────────────┼────────────┼────────────────────┼─────────────┼──┘
                  │            │                    │             │
    ┌─────────────┴──┐   ┌────┴───────────┐   ┌───┴─────────────┴──┐
    │   Adapters:    │   │                │   │   Adapters:        │
    │                │   │                │   │                    │
    │ Historical     │   │   Aggregating  │   │ BacktestExecutor   │
    │ DataFeed       │   │   DataFeed     │   │ ShadowExecutor     │
    │                │   │                │   │ PaperExecutor      │
    │ LiveDataFeed   │   │                │   │ LiveExecutor       │
    │ (Phase 8)      │   │                │   │ DualExecutor       │
    └────────────────┘   └────────────────┘   └────────────────────┘
```

### 5.2 Port 정의 (Protocol)

```python
# src/eda/ports.py (새 파일)

from typing import Protocol, runtime_checkable

@runtime_checkable
class DataFeedPort(Protocol):
    """데이터 공급 포트."""
    async def start(self, bus: EventBus) -> None: ...
    async def stop(self) -> None: ...
    @property
    def bars_emitted(self) -> int: ...

@runtime_checkable
class ExecutorPort(Protocol):
    """주문 실행 포트."""
    async def execute(self, order: OrderRequestEvent) -> FillEvent | None: ...

@runtime_checkable
class MetricsPort(Protocol):
    """성과 분석 포트."""
    def compute(
        self,
        equity_curve: list[EquityPoint],
        trades: list[TradeRecord],
        timeframe: str,
    ) -> PerformanceMetrics: ...
```

### 5.3 어댑터 구현체 (5가지 모드)

| 모드 | DataFeed | Executor | 용도 |
|------|----------|----------|------|
| **EDA Backtest** | `HistoricalDataFeed` | `BacktestExecutor` | 히스토리컬 검증 |
| **EDA Agg Backtest** | `AggregatingDataFeed` | `BacktestExecutor` | 1m 집계 백테스트 |
| **Paper Trading** | `LiveDataFeed` (new) | `PaperExecutor` (new) | 실시간 페이퍼 |
| **Shadow Trading** | `LiveDataFeed` (new) | `DualExecutor` (new) | 라이브 + 시그널 비교 |
| **Live Trading** | `LiveDataFeed` (new) | `LiveExecutor` (new) | 프로덕션 |

### 5.4 Runner 팩토리 패턴

```python
# src/eda/runner.py (확장)

class EDARunner:
    """통합 러너 - 어댑터 조합으로 모든 모드 지원."""

    def __init__(
        self,
        strategy: BaseStrategy,
        feed: DataFeedPort,
        executor: ExecutorPort,
        config: PortfolioManagerConfig,
        initial_capital: float,
        asset_weights: dict[str, float] | None = None,
    ) -> None:
        self._strategy = strategy
        self._feed = feed
        self._executor = executor
        self._config = config
        self._initial_capital = initial_capital
        self._asset_weights = asset_weights

    @classmethod
    def backtest(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        config: PortfolioManagerConfig,
        initial_capital: float = 100_000,
        **kwargs,
    ) -> "EDARunner":
        """백테스트 모드."""
        return cls(
            strategy=strategy,
            feed=HistoricalDataFeed(data),
            executor=BacktestExecutor(config.cost_model),
            config=config,
            initial_capital=initial_capital,
            **kwargs,
        )

    @classmethod
    def backtest_agg(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        initial_capital: float = 100_000,
        **kwargs,
    ) -> "EDARunner":
        """1m 집계 백테스트 모드."""
        return cls(
            strategy=strategy,
            feed=AggregatingDataFeed(data, target_timeframe),
            executor=BacktestExecutor(config.cost_model),
            config=config,
            initial_capital=initial_capital,
            **kwargs,
        )

    @classmethod
    def paper(
        cls,
        strategy: BaseStrategy,
        symbols: list[str],
        config: PortfolioManagerConfig,
        initial_capital: float = 100_000,
        **kwargs,
    ) -> "EDARunner":
        """페이퍼 트레이딩 모드 (실시간 데이터 + 가상 체결)."""
        return cls(
            strategy=strategy,
            feed=LiveDataFeed(symbols),           # Phase 8
            executor=PaperExecutor(config.cost_model),  # Phase 8
            config=config,
            initial_capital=initial_capital,
            **kwargs,
        )

    @classmethod
    def shadow(
        cls,
        strategy: BaseStrategy,
        symbols: list[str],
        config: PortfolioManagerConfig,
        **kwargs,
    ) -> "EDARunner":
        """섀도우 모드 (라이브 + 시그널 로깅 비교)."""
        live_exec = LiveExecutor(...)
        shadow_exec = ShadowExecutor()
        return cls(
            strategy=strategy,
            feed=LiveDataFeed(symbols),
            executor=DualExecutor(primary=live_exec, shadow=shadow_exec),
            config=config,
            **kwargs,
        )
```

### 5.5 새로운 어댑터 구현체 (Phase 8용)

```python
# src/eda/executors.py (확장)

class PaperExecutor:
    """가상 체결 + 슬리피지 시뮬레이션.

    BacktestExecutor와 달리 실시간 시장 가격 기반으로 체결.
    fill 가격에 랜덤 슬리피지 추가 (현실적 시뮬레이션).
    """
    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        price = self._current_market_price[order.symbol]
        slipped_price = price * (1 + random.gauss(0, self._slippage_std))
        return FillEvent(price=slipped_price, fees=..., ...)

class LiveExecutor:
    """CCXT Pro 기반 실제 거래소 주문 실행."""
    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        result = await self._exchange.create_order(
            symbol=order.symbol,
            type="market",
            side=order.side,
            amount=order.quantity,
        )
        return FillEvent(price=result["price"], fees=..., ...)

class DualExecutor:
    """라이브 + 섀도우 동시 실행.

    primary가 실제 체결, shadow는 로깅만.
    두 결과를 비교하여 시그널 품질 검증.
    """
    def __init__(self, primary: ExecutorPort, shadow: ExecutorPort):
        self._primary = primary
        self._shadow = shadow

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        shadow_task = asyncio.create_task(self._shadow.execute(order))
        fill = await self._primary.execute(order)
        await shadow_task  # 비교 로깅
        return fill
```

### 5.6 LiveDataFeed (Phase 8용)

```python
# src/eda/data_feed.py (확장)

class LiveDataFeed:
    """CCXT Pro WebSocket 기반 실시간 데이터 피드.

    NautilusTrader의 LiveMarketDataClient 패턴 참조:
    - WebSocket 구독으로 실시간 OHLCV bar 수신
    - CandleAggregator로 1m → target TF 집계
    - BarEvent로 변환하여 EventBus에 발행
    """

    def __init__(self, symbols: list[str], timeframe: str = "1m"):
        self._symbols = symbols
        self._timeframe = timeframe

    async def start(self, bus: EventBus) -> None:
        async with BinanceClient() as client:
            while True:
                for symbol in self._symbols:
                    ohlcv = await client.watch_ohlcv(symbol, self._timeframe)
                    bar = self._to_bar_event(symbol, ohlcv[-1])
                    await bus.publish(bar)
                    await bus.flush()  # bar-by-bar 동기화

    async def stop(self) -> None:
        self._running = False
```

---

## 6. 구체적 리팩토링 계획

### 6.1 Phase 6-A: Protocol 정의 + 기존 코드 정리 (1~2일)

**목표:** 명시적 Port Protocol 도입, 기존 어댑터가 Protocol을 만족하도록 정리

**작업:**
1. `src/eda/ports.py` 생성 — `DataFeedPort`, `ExecutorPort`, `MetricsPort` Protocol 정의
2. `HistoricalDataFeed`, `AggregatingDataFeed`가 `DataFeedPort` 만족하는지 확인 (이미 만족할 것)
3. `BacktestExecutor`, `ShadowExecutor`가 `ExecutorPort` 만족하는지 확인 (이미 만족)
4. Runner 생성자에서 Protocol 타입 힌트 추가
5. 테스트: 기존 397개 테스트 전부 통과 확인

**변경 파일:**
- `src/eda/ports.py` (새 파일, ~30 lines)
- `src/eda/runner.py` (타입 힌트 추가)
- 기존 코드 변경 최소화 (Protocol은 structural subtyping)

### 6.2 Phase 6-B: Runner 팩토리 메서드 (1일)

**목표:** `EDARunner.backtest()`, `EDARunner.backtest_agg()` 등 팩토리 메서드 추가

**작업:**
1. `EDARunner`에 `@classmethod` 팩토리 메서드 추가
2. 기존 `__init__` 시그니처는 유지 (하위 호환)
3. CLI에서 팩토리 메서드 사용하도록 변경

**변경 파일:**
- `src/eda/runner.py` (~50 lines 추가)
- `src/cli/eda.py` (팩토리 메서드 사용)

### 6.3 Phase 6-C: 통합 Metrics 엔진 (2~3일)

**목표:** PerformanceAnalyzer와 AnalyticsEngine의 메트릭 계산 로직 통합

**작업:**
1. `src/backtest/metrics.py` 생성 — 공통 메트릭 계산 함수
2. `PerformanceAnalyzer`에서 공통 함수 호출하도록 리팩토링
3. `AnalyticsEngine`에서 공통 함수 호출하도록 리팩토링
4. 양쪽의 parity 테스트 추가

**변경 파일:**
- `src/backtest/metrics.py` (새 파일, ~200 lines)
- `src/backtest/analyzer.py` (공통 함수 호출)
- `src/eda/analytics.py` (공통 함수 호출)

### 6.4 Phase 6-D: 통합 CLI (1일)

**목표:** 2개 CLI를 하나로 통합하거나, 공통 인터페이스 제공

**작업:**
1. `src/cli/eda.py`에 `--mode` 옵션 추가 (backtest/backtest-agg/paper/shadow/live)
2. 또는 기존 CLI 유지하면서 `src/cli/unified.py` 추가 (선택)

**변경 파일:**
- `src/cli/eda.py` 또는 새 통합 CLI

### 6.5 Phase 8 (향후): 라이브 어댑터 구현

```
Phase 8-A: LiveDataFeed
├── CCXT Pro WebSocket 연동
├── CandleAggregator 통합
└── Reconnection + error handling

Phase 8-B: LiveExecutor
├── CCXT Pro 주문 실행
├── Rate limiting
└── Error handling + retry

Phase 8-C: Reconciliation
├── Startup: 거래소 상태 → 내부 상태 동기화
├── Continuous: in-flight 주문 모니터링
└── State persistence (Redis/SQLite)

Phase 8-D: PaperExecutor + DualExecutor
├── 가상 체결 + 슬리피지 시뮬
├── Shadow 비교 로깅
└── 카나리 배포 지원
```

---

## 7. VBT와의 관계 정리

### 7.1 VBT의 역할 (변경 없음)

VBT는 **Phase 1: 빠른 파라미터 스크리닝** 전용으로 유지한다.

```
Phase 1: VBT Screening (수초~수분)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
수천 개 파라미터 조합 → 유망 후보 N개 선별
Numba JIT vectorization으로 압도적 속도
look-ahead bias 가능하지만, 스크리닝 목적으로는 충분

Phase 2: EDA Validation (수분~수십분)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
유망 후보 정밀 검증 (bar-by-bar, SL/TS, 비용 모델)
IS/OOS, WFA, CPCV, DSR, PBO 고급 검증
look-ahead bias 구조적 차단

Phase 3: Deployment
━━━━━━━━━━━━━━━━━━
동일 EDA 코드로 Paper → Shadow → Canary → Live
```

### 7.2 VBT 리스크 규칙 중복 처리 전략

**결론: VBT의 Numba 리스크 규칙은 유지한다.**

이유:
1. VBT의 Numba 함수는 vectorized 최적화를 위해 **의도적으로 다른 구현**
2. 두 구현의 "결과 동등성"은 중요하지만, "코드 동일성"은 불필요
3. VBT는 스크리닝 전용이므로, 정확한 parity보다 **상대적 순위 보존**이 중요
4. Parity 테스트로 두 구현의 결과 차이를 지속 모니터링하면 충분

```
VBT Numba 함수 (유지)        EDA 이벤트 핸들러 (유지)
━━━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━━━
apply_stop_loss_to_weights   PM._on_intrabar
apply_trailing_stop_to_weights PM._update_trailing_stop
apply_rebalance_threshold     PM._evaluate_rebalance

                ┌────────────────────┐
                │  Parity Test Suite  │
                │  (결과 비교 테스트)   │
                │  Sharpe 부호 일치    │
                │  거래 수 ±20%       │
                │  수익률 부호 일치     │
                └────────────────────┘
```

### 7.3 통합 전략 코드 흐름

```
                    BaseStrategy.run(df)
                    ┌─────────┴─────────┐
                    │                   │
              VBT Path              EDA Path
              (Phase 1)             (Phase 2+)
                    │                   │
        strategy.run(full_df)     StrategyEngine:
              │                   strategy.run(buffer_df)
              ▼                   extract signals[-1]
        StrategySignals                 │
        (full time series)              ▼
              │                   SignalEvent (per bar)
              ▼                         │
        VBT Portfolio              PM → RM → OMS
        (vectorized)               (event-driven)
              │                         │
              ▼                         ▼
        BacktestResult            PerformanceMetrics
```

---

## 8. 운영 모드별 배포 전략

### 8.1 5단계 배포 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deployment Pipeline                           │
│                                                                  │
│  ① VBT Screening ──→ ② EDA Backtest ──→ ③ Paper Trading        │
│  (파라미터 탐색)       (정밀 검증)          (실시간 가상 체결)      │
│                                                                  │
│              ④ Shadow Trading ──→ ⑤ Live (Canary → Full)        │
│              (라이브 시그널 비교)     (실제 거래)                    │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 각 단계별 상세

| 단계 | DataFeed | Executor | 자본 | 기간 | Gate 조건 |
|------|----------|----------|------|------|----------|
| ① VBT Screening | 일괄 DataFrame | VBT Portfolio | - | 수초 | 유망 후보 N개 |
| ② EDA Backtest | HistoricalDataFeed | BacktestExecutor | - | 수분 | IS/OOS Sharpe > 0, PBO < 0.3 |
| ③ Paper Trading | LiveDataFeed | PaperExecutor | 가상 $100K | 2~4주 | 수익 부호 일치, MDD < 30% |
| ④ Shadow Trading | LiveDataFeed | DualExecutor | 라이브 5% | 1~2주 | 시그널 일치율 > 90% |
| ⑤ Live (Canary) | LiveDataFeed | LiveExecutor | 라이브 10% | 2주 | MDD < 20%, 이상 없음 |
| ⑤ Live (Full) | LiveDataFeed | LiveExecutor | 라이브 100% | - | 지속 모니터링 |

### 8.3 자동 롤백 조건

```python
# 카나리 → 롤백 트리거
ROLLBACK_CONDITIONS = {
    "max_drawdown_pct": 20.0,       # MDD 20% 초과
    "consecutive_losses": 10,        # 연속 손실 10회
    "latency_p99_ms": 5000,         # 주문 지연 5초 초과
    "fill_rejection_rate": 0.1,     # 체결 거절율 10% 초과
    "signal_divergence_pct": 30.0,  # 시그널 불일치 30% 초과
}
```

---

## 9. Trade-offs 및 결론

### 9.1 이 리팩토링에서 하지 않는 것

| 하지 않는 것 | 이유 |
|-------------|------|
| VBT 코드 삭제/교체 | VBT는 Phase 1 스크리닝에서 대체 불가 (속도) |
| NautilusTrader 도입 | 마이그레이션 비용 대비 효과 낮음 ([기존 평가 참조](nautilus-trader-evaluation.md)) |
| VBT-EDA 리스크 규칙 통합 | 의도적 최적화 차이 (Numba vs Event), parity 테스트로 관리 |
| 즉시 라이브 어댑터 구현 | Phase 8에서 점진적 구현 (현재는 Protocol 정의까지) |

### 9.2 즉시 실행 가능한 것 (Phase 6)

| 작업 | 예상 기간 | 효과 |
|------|----------|------|
| Port Protocol 정의 | 1~2일 | 아키텍처 명확화, 향후 확장 기반 |
| Runner 팩토리 메서드 | 1일 | 모드 전환 간소화 |
| 통합 Metrics 엔진 | 2~3일 | 코드 중복 ~300 lines 제거 |
| 통합 CLI | 1일 | UX 개선 |

**총 예상 기간: 5~7일**

### 9.3 핵심 결론

1. **MC Coin Bot의 EDA는 이미 어댑터 패턴의 핵심을 갖추고 있다**
   - `Executor` Protocol, `EventBus`, `DataFeed` 패턴이 NautilusTrader/LEAN과 구조적으로 일치
   - 새로운 아키텍처 도입이 아니라 **명시적 Protocol 정의 + 팩토리 패턴 추가**로 충분

2. **VBT와 EDA의 코드 중복은 의도적이며, 통합보다 parity 테스트가 적절**
   - Numba vectorized (VBT) vs Event-driven stateful (EDA)는 근본적으로 다른 패러다임
   - "동일 결과" 검증이 "동일 코드"보다 중요

3. **라이브 전환 시 코드 변경이 최소화되는 구조**
   - `LiveDataFeed` + `LiveExecutor` 구현체 추가 → Runner 팩토리에서 조합
   - 전략 코드, PM, RM, OMS, Analytics 모두 변경 없음
   - 이것이 NautilusTrader가 달성한 것과 동일한 가치

4. **5단계 배포 파이프라인으로 안전한 프로덕션 진입**
   - VBT → EDA Backtest → Paper → Shadow → Live (Canary → Full)
   - 각 단계별 Gate 조건으로 자동/수동 검증

### 9.4 우선순위 로드맵

```
즉시 (Phase 6, 5~7일)
├── 6-A: Port Protocol 정의
├── 6-B: Runner 팩토리 메서드
├── 6-C: 통합 Metrics 엔진
└── 6-D: 통합 CLI (선택)

다음 (Phase 7)
├── 전략 연구 + 검증 계속
└── Dynamic Slippage Model

향후 (Phase 8, 라이브 준비)
├── 8-A: LiveDataFeed (CCXT Pro WebSocket)
├── 8-B: LiveExecutor (CCXT Pro 주문)
├── 8-C: Reconciliation
├── 8-D: PaperExecutor + DualExecutor
└── 8-E: Canary 배포 인프라
```

---

## 10. References

### NautilusTrader (핵심 참조)
- [Architecture - Ports & Adapters](https://nautilustrader.io/docs/latest/concepts/architecture/)
- [Adapter Development Guide](https://nautilustrader.io/docs/nightly/developer_guide/adapters/)
- [Strategy Interface](https://nautilustrader.io/docs/latest/concepts/strategies/)
- [MessageBus](https://nautilustrader.io/docs/nightly/concepts/message_bus/)
- [Backtesting Concepts](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- [Live Trading](https://nautilustrader.io/docs/latest/concepts/live/)
- [Execution Flow](https://nautilustrader.io/docs/nightly/concepts/execution/)
- [GitHub Repository](https://github.com/nautechsystems/nautilus_trader)

### LEAN (QuantConnect)
- [Algorithm Engine Architecture (DeepWiki)](https://deepwiki.com/QuantConnect/Lean/3-engine-and-execution)
- [GitHub Repository](https://github.com/QuantConnect/Lean)

### Other Frameworks
- [Backtrader - Broker Docs](https://www.backtrader.com/docu/broker/)
- [Backtrader - Live Trading IB](https://www.backtrader.com/docu/live/ib/ib/)
- [VectorBT Pro - Portfolio](https://vectorbt.pro/features/portfolio/)
- [StrateQueue - VBT Live Bridge](https://github.com/StrateQueue/StrateQueue)
- [Zipline-Reloaded](https://github.com/stefan-jansen/zipline-reloaded)

### Architecture Patterns
- [QuantStart - Event-Driven Backtesting Part I](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/)
- [IBKR - Vector vs Event Backtesting](https://www.interactivebrokers.com/campus/ibkr-quant-news/a-practical-breakdown-of-vector-based-vs-event-based-backtesting/)

### MC Coin Bot Internal
- [NautilusTrader 도입 평가](nautilus-trader-evaluation.md) — NautilusTrader 도입 비추천 결론
- [VBT → EDA 전환 가이드](vbt-to-eda-transition-guide.md) — 하이브리드 파이프라인 설계 원칙
