# Implementation Roadmap

전략 확정(8-asset EW TSMOM, Sharpe 2.33) 이후 실거래까지의 구현 로드맵.

> **현재 위치:** Phase 1 완료 (단일에셋 백테스트) → Phase 2 진입 예정

---

## Phase 개요

```
Phase 1          Phase 2              Phase 3           Phase 4          Phase 5          Phase 6
단일에셋          멀티에셋             고급 검증          EDA 시스템       Dry Run          Live
백테스트          백테스트 확장         IS/OOS/WFA/CPCV   이벤트 기반       Paper Trading     실거래
✅ 완료           ← 현재              예정               예정             예정              예정
```

---

## Phase 1: 단일에셋 백테스트 (완료)

- VectorBT 기반 백테스트 엔진
- TSMOM/Breakout/Donchian/BB-RSI 4개 전략
- 파라미터 스윕, QuantStats 리포트
- Numba 최적화 (stop-loss, trailing-stop, rebalance)

---

## Phase 2: 멀티에셋 백테스트 확장

### 2.1 목표

단일 심볼 → 8-asset Equal-Weight 포트폴리오를 **하나의 VectorBT Portfolio**로 통합 백테스트.

### 2.2 현재 아키텍처의 제약

| 컴포넌트 | 현재 (단일) | 필요 (멀티) |
|---------|-----------|-----------|
| `MarketDataSet` | `symbol: str`, `ohlcv: DataFrame` | 심볼별 OHLCV dict |
| `BacktestEngine._execute()` | `close=Series` 단일 심볼 | `close=DataFrame` 멀티 심볼 |
| `BaseStrategy.run()` | 단일 DataFrame → 단일 Signals | 심볼별 독립 실행 후 합성 |
| `StrategySignals` | `pd.Series` 4개 | `pd.DataFrame` 4개 (columns=symbols) |
| `Portfolio` | 단일 자본 | 심볼별 자본 배분 (EW: 1/N) |
| `PerformanceAnalyzer` | 단일 포트폴리오 분석 | 포트폴리오 + 심볼별 분해 분석 |

### 2.3 설계 방향

**원칙: 기존 단일에셋 인터페이스를 깨지 않으면서 멀티에셋 래퍼 추가**

#### A. 데이터 레이어

```python
# 새로 추가 — 멀티 심볼 데이터 컨테이너
@dataclass
class MultiSymbolData:
    symbols: list[str]
    timeframe: str
    start: datetime
    end: datetime
    ohlcv: dict[str, pd.DataFrame]  # symbol → OHLCV DataFrame

    @property
    def close_matrix(self) -> pd.DataFrame:
        """심볼별 close를 DataFrame으로 합성 (VectorBT 입력용)"""
        return pd.DataFrame({s: df["close"] for s, df in self.ohlcv.items()})

    def get_single(self, symbol: str) -> MarketDataSet:
        """기존 단일에셋 인터페이스 호환"""
        ...
```

#### B. 전략 실행 — 심볼별 독립 실행 패턴 (권장)

```python
# BacktestEngine에 추가
def run_multi(self, request: MultiAssetBacktestRequest) -> MultiAssetBacktestResult:
    signals_dict: dict[str, StrategySignals] = {}
    processed_dict: dict[str, pd.DataFrame] = {}

    for symbol in request.data.symbols:
        single_df = request.data.ohlcv[symbol]
        processed, signals = request.strategy.run(single_df)
        signals_dict[symbol] = signals
        processed_dict[symbol] = processed

    # DataFrame으로 합성 (VectorBT 멀티에셋 입력)
    entries_df = pd.DataFrame({s: sig.entries for s, sig in signals_dict.items()})
    exits_df = pd.DataFrame({s: sig.exits for s, sig in signals_dict.items()})

    vbt_portfolio = vbt.Portfolio.from_signals(
        close=request.data.close_matrix,
        entries=entries_df,
        exits=exits_df,
        cash_sharing=True,        # 심볼 간 현금 공유
        init_cash=total_capital,
        group_by=True,            # 전체를 하나의 그룹으로
        ...
    )
```

**이 패턴의 장점:**
- `BaseStrategy` 인터페이스 변경 없음 (단일 df → 단일 signals)
- 각 심볼 독립 실행 → 병렬화 가능
- 기존 단일에셋 백테스트와 100% 호환

#### C. 포트폴리오 배분

```python
class AllocationStrategy(str, Enum):
    EQUAL_WEIGHT = "equal_weight"     # 1/N (확정)
    # 향후 확장 가능: RISK_PARITY, INVERSE_VOL, ...

class MultiAssetPortfolioConfig(BaseModel):
    symbols: list[str]
    allocation: AllocationStrategy = AllocationStrategy.EQUAL_WEIGHT
    total_capital: Decimal
    max_leverage_cap: float = 2.0
    strategy_config: TSMOMConfig  # 또는 BaseStrategyConfig
```

#### D. 분석 확장

```python
class MultiAssetResult(BaseModel):
    portfolio_metrics: PerformanceMetrics   # 포트폴리오 전체
    per_symbol_metrics: dict[str, PerformanceMetrics]  # 심볼별
    correlation_matrix: pd.DataFrame        # 수익률 상관관계
    contribution: dict[str, float]          # 심볼별 수익 기여도
```

### 2.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `MultiSymbolData` 데이터 컨테이너 구현 | 낮음 |
| 2 | `DataService.fetch_multi()` 구현 (Silver 레이어 일괄 로딩) | 낮음 |
| 3 | `MultiAssetBacktestRequest` DTO 구현 | 낮음 |
| 4 | `BacktestEngine.run_multi()` 구현 | 중간 |
| 5 | VectorBT `cash_sharing` + `group_by` 멀티에셋 통합 | 중간 |
| 6 | `MultiAssetResult` + 심볼별 분해 분석 | 중간 |
| 7 | CLI 확장 — `backtest run-multi tsmom --symbols BTC,ETH,...` | 낮음 |
| 8 | 기존 단일에셋 테스트가 깨지지 않는지 검증 | 낮음 |

---

## Phase 3: 고급 검증 (과적합 방지)

### 3.1 왜 필요한가

현재 백테스트는 **전체 기간 단일 테스트** — 과적합(overfitting) 여부를 판단할 수 없음.
전략이 미래에도 작동할 확률을 높이려면 통계적으로 엄밀한 검증이 필수.

### 3.2 검증 방법론 (3단계)

#### Stage 1: In-Sample / Out-of-Sample Split

가장 기본적인 검증. 전체 데이터를 시간순으로 분할.

```
|← ───── In-Sample (70%) ─────→|← ── OOS (30%) ──→|
|  2020-01          2024-04    |  2024-04   2025-12 |
|  파라미터 최적화              |  성과 확인         |
```

**구현:**
```python
class ISOOSConfig(BaseModel):
    train_ratio: float = 0.7     # IS 비율
    min_oos_days: int = 365      # 최소 OOS 기간

class ISOOSResult(BaseModel):
    is_sharpe: float             # In-Sample Sharpe
    oos_sharpe: float            # Out-of-Sample Sharpe
    degradation: float           # (IS - OOS) / IS — 0.3 이하 권장
    oos_positive: bool           # OOS에서도 수익인가?
```

#### Stage 2: Walk-Forward Analysis (WFA)

데이터를 슬라이딩 윈도우로 반복 검증 — IS에서 최적화, OOS에서 테스트, 윈도우를 전진.

```
Window 1: |───IS───|─OOS─|
Window 2:     |───IS───|─OOS─|
Window 3:         |───IS───|─OOS─|
...
→ 모든 OOS 구간을 이어 붙여 "합성 OOS 수익률" 생성
```

**구현:**
```python
class WalkForwardConfig(BaseModel):
    is_window_days: int = 720    # IS 윈도우 (2년)
    oos_window_days: int = 180   # OOS 윈도우 (6개월)
    step_days: int = 90          # 전진 폭 (3개월)
    min_windows: int = 4         # 최소 윈도우 수

class WalkForwardResult(BaseModel):
    windows: list[WFWindowResult]       # 각 윈도우 결과
    combined_oos_sharpe: float          # 합성 OOS Sharpe
    efficiency_ratio: float             # OOS Sharpe / IS Sharpe 평균
    consistency: float                  # OOS 수익 윈도우 비율 (>60% 권장)
```

#### Stage 3: Combinatorial Purged Cross-Validation (CPCV)

2019년 Marcos López de Prado가 제안. Walk-Forward의 한계(단일 경로 의존성)를 극복.

```
데이터를 N개 블록으로 분할 → C(N,k) 조합으로 train/test 생성
각 분할에서:
  1. Purging: train/test 경계의 겹치는 샘플 제거
  2. Embargo: purge 후 추가 gap 적용 (자기상관 차단)
→ 수백~수천 개의 독립적 백테스트 경로 생성
→ Sharpe 분포로 과적합 확률(PBO) 추정
```

**구현:**
```python
class CPCVConfig(BaseModel):
    n_splits: int = 6            # 데이터 블록 수
    n_test_splits: int = 2       # 테스트에 사용할 블록 수
    embargo_pct: float = 0.01    # 엠바고 비율 (전체 데이터의 1%)
    purge_window: int = 5        # Purge 윈도우 (거래일)

class CPCVResult(BaseModel):
    pbo: float                   # Probability of Backtest Overfitting (< 0.5 권장)
    sharpe_distribution: list[float]  # 경로별 Sharpe 분포
    mean_oos_sharpe: float
    median_oos_sharpe: float
    deflated_sharpe: float       # Deflated Sharpe Ratio (다중 테스트 보정)
```

### 3.3 검증 판정 기준

| 기준 | 통과 | 주의 | 실패 |
|------|------|------|------|
| IS/OOS 성과 열화 | < 30% | 30~50% | > 50% |
| WFA OOS Sharpe | > 1.0 | 0.5~1.0 | < 0.5 |
| WFA Consistency | > 70% | 50~70% | < 50% |
| CPCV PBO | < 0.30 | 0.30~0.50 | > 0.50 |
| Deflated Sharpe | > 1.0 | 0.5~1.0 | < 0.5 |

### 3.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `src/backtest/validation/` 모듈 확장 | 중간 |
| 2 | IS/OOS Split 검증기 구현 | 낮음 |
| 3 | Walk-Forward Analyzer 구현 | 중간 |
| 4 | CPCV 검증기 구현 (purge + embargo) | 높음 |
| 5 | PBO (Probability of Backtest Overfitting) 계산 | 높음 |
| 6 | Deflated Sharpe Ratio 구현 | 중간 |
| 7 | CLI 확장 — `backtest validate tsmom --method wfa` | 낮음 |
| 8 | 검증 리포트 생성 (결과 요약 + 시각화) | 중간 |

---

## Phase 4: EDA 시스템 (이벤트 기반 아키텍처)

### 4.1 왜 이벤트 기반인가

현재 백테스트는 **벡터화 방식** — 전체 데이터를 한 번에 처리. 빠르지만 실거래와 구조가 다름.
EDA 시스템은 **실거래와 동일한 코드 경로**로 백테스트를 실행하여 backtest-live parity를 보장.

```
                    Vectorized Backtest          EDA Backtest
─────────────────────────────────────────────────────────────
데이터              전체 DataFrame 한번에          바(bar) 단위 이벤트
시그널 생성          전체 Series 한번에            각 바마다 개별 생성
주문 체결           VectorBT 내부 시뮬레이션       OMS 시뮬레이터
리스크 체크          사후 분석                     실시간 체크 (사전)
코드 재사용          별도 라이브 구현 필요          라이브와 동일 코드
속도               매우 빠름 (Numba)              느림 (이벤트 오버헤드)
용도               파라미터 탐색, 빠른 실험        최종 검증, 라이브 전환
```

### 4.2 아키텍처 설계

```
┌─────────────────────────────────────────────────────┐
│                    EventBus                          │
│  publish(event) → [handler1, handler2, ...]          │
└──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
       │      │      │      │      │      │
  ┌────▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼───┐ ┌▼───┐ ┌▼────────┐
  │Market │ │Strat │ │ PM  │ │ RM │ │OMS │ │Analytics│
  │Data   │ │Engine│ │     │ │    │ │    │ │Engine   │
  │Feed   │ │      │ │     │ │    │ │    │ │         │
  └───────┘ └──────┘ └─────┘ └────┘ └────┘ └─────────┘
```

#### 이벤트 타입 정의

```python
# 시장 데이터 이벤트
class BarEvent(BaseEvent):
    symbol: str
    timeframe: str
    ohlcv: dict[str, float]       # open, high, low, close, volume
    timestamp: datetime

# 전략 시그널 이벤트
class SignalEvent(BaseEvent):
    symbol: str
    direction: Direction           # LONG / SHORT / FLAT
    strength: float                # 레버리지 강도
    strategy_name: str
    timestamp: datetime

# 포트폴리오 이벤트
class OrderRequestEvent(BaseEvent):
    symbol: str
    side: OrderSide                # BUY / SELL
    quantity: Decimal
    order_type: OrderType          # MARKET / LIMIT
    client_order_id: str           # 멱등성 키

# 주문 체결 이벤트
class FillEvent(BaseEvent):
    symbol: str
    side: OrderSide
    fill_price: Decimal
    fill_quantity: Decimal
    commission: Decimal
    timestamp: datetime
```

#### 실행 모드 통합

```python
class ExecutionMode(IntEnum):
    BACKTEST = 0     # 히스토리컬 데이터 리플레이
    PAPER = 1        # 실시간 데이터 + 가상 주문 (거래소 paper API)
    SHADOW = 2       # 실시간 데이터 + 로깅만 (주문 발송 X)
    LIVE = 3         # 실거래

class OrderExecutor(Protocol):
    """OMS가 모드별로 다른 Executor를 주입받음"""
    async def submit(self, order: OrderRequestEvent) -> FillEvent: ...

class BacktestExecutor:     # ExecutionMode.BACKTEST
    """히스토리컬 데이터 기반 즉시 체결 시뮬레이션"""

class PaperExecutor:        # ExecutionMode.PAPER
    """거래소 Testnet API 사용 (Binance Testnet)"""

class ShadowExecutor:       # ExecutionMode.SHADOW
    """주문을 보내지 않고 로깅만 수행, 이후 실제 시장과 비교"""

class LiveExecutor:         # ExecutionMode.LIVE
    """실제 거래소 API로 주문"""
```

#### EventBus 구현

```python
class EventBus:
    """타입 안전한 Pub/Sub 이벤트 버스"""

    def __init__(self) -> None:
        self._handlers: dict[type[BaseEvent], list[EventHandler]] = defaultdict(list)
        self._event_log: list[BaseEvent] = []  # Event Sourcing용

    def subscribe(self, event_type: type[E], handler: Callable[[E], Awaitable[None]]) -> None:
        self._handlers[event_type].append(handler)

    async def publish(self, event: BaseEvent) -> None:
        self._event_log.append(event)  # 감사 로그
        for handler in self._handlers.get(type(event), []):
            await handler(event)
```

#### EDA 백테스트 실행 흐름

```python
async def run_eda_backtest(config: EDABacktestConfig) -> BacktestResult:
    bus = EventBus()

    # 컴포넌트 연결
    data_feed = HistoricalDataFeed(config.data, bus)      # BarEvent 발행
    strategy = StrategyEngine(config.strategy, bus)        # BarEvent → SignalEvent
    pm = PortfolioManager(config.portfolio, bus)           # SignalEvent → OrderRequestEvent
    rm = RiskManager(config.risk, bus)                     # OrderRequestEvent 검증
    oms = OrderManagementSystem(BacktestExecutor(), bus)   # OrderRequestEvent → FillEvent
    analytics = AnalyticsEngine(bus)                       # 모든 이벤트 수집

    # 데이터 리플레이
    await data_feed.replay()  # 바 하나씩 BarEvent 발행

    return analytics.generate_result()
```

### 4.3 Vectorized vs EDA 백테스트 병행 전략

둘 다 유지. 용도가 다름:

| 용도 | 엔진 | 이유 |
|------|------|------|
| 파라미터 탐색/스윕 | Vectorized (VectorBT) | 속도 (수백~수천 배 빠름) |
| IS/OOS, WFA, CPCV | Vectorized (VectorBT) | 대량 반복 필요 |
| 최종 검증 | EDA | 실거래와 동일 코드 경로 |
| Paper/Shadow/Live | EDA | 실시간 이벤트 처리 |

### 4.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `BaseEvent` + 이벤트 타입 정의 (`src/core/events.py`) | 낮음 |
| 2 | `EventBus` 구현 (async, type-safe, event log) | 중간 |
| 3 | `HistoricalDataFeed` — 바 단위 이벤트 리플레이 | 중간 |
| 4 | `StrategyEngine` — BaseStrategy 래퍼 (bar-by-bar) | 중간 |
| 5 | `PortfolioManager` 이벤트 기반 리팩토링 | 높음 |
| 6 | `RiskManager` 구현 (포지션 사이즈, 레버리지 체크) | 높음 |
| 7 | `OMS` + `ExecutionMode` + Executor 패턴 구현 | 높음 |
| 8 | `BacktestExecutor` — 즉시 체결 시뮬레이터 | 중간 |
| 9 | EDA 백테스트 결과가 Vectorized와 일치하는지 검증 | 높음 |
| 10 | 통합 테스트 — 전체 이벤트 파이프라인 동작 확인 | 중간 |

---

## Phase 5: Dry Run (Paper Trading)

### 5.1 3단계 실거래 전 검증

```
Stage 1: Shadow Mode (2~4주)
  └─ 실시간 데이터 수신 + 시그널 생성 + 로깅 (주문 X)
  └─ 목적: 시그널이 백테스트와 일관되는지 확인

Stage 2: Paper Trading (1~3개월)
  └─ Binance Testnet으로 가상 주문 실행
  └─ 목적: OMS/PM/RM 파이프라인 end-to-end 검증

Stage 3: Canary Live (1~2개월)
  └─ 최소 자본(포트폴리오의 5~10%)으로 실거래
  └─ 목적: 실제 슬리피지, 체결 지연, API 안정성 확인
```

### 5.2 Shadow Mode 상세

```python
class ShadowExecutor:
    """주문을 보내지 않고, 시장가로 체결되었을 것이라 가정하여 기록"""

    async def submit(self, order: OrderRequestEvent) -> FillEvent:
        # 실제 시장 데이터의 현재가로 가상 체결
        current_price = await self.market_data.get_price(order.symbol)
        slippage = self._estimate_slippage(order)

        fill = FillEvent(
            symbol=order.symbol,
            fill_price=current_price + slippage,
            fill_quantity=order.quantity,
            commission=self._calc_commission(order),
            is_simulated=True,
        )

        # 로깅 — 나중에 실제 시장과 비교
        self.shadow_log.record(order, fill)
        return fill
```

### 5.3 Paper Trading 체크리스트

| 체크 항목 | 기준 |
|----------|------|
| WebSocket 연결 안정성 | 24시간 무중단 (자동 재연결) |
| 시그널 일관성 | Shadow vs 백테스트 시그널 일치율 > 95% |
| 주문 실행 성공률 | > 99% (Testnet 기준) |
| 리밸런싱 정확도 | 목표 비중 대비 오차 < 2% |
| 리스크 체크 동작 | max_leverage, system_stop_loss 정상 트리거 |
| 에러 핸들링 | API 오류, 타임아웃, rate limit 정상 처리 |
| 알림 시스템 | Discord 알림 정상 수신 |
| 로그 완결성 | 모든 이벤트가 감사 로그에 기록 |

### 5.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `ShadowExecutor` 구현 | 중간 |
| 2 | `PaperExecutor` — Binance Testnet 연동 | 중간 |
| 3 | WebSocket 실시간 데이터 피드 (`ccxt.pro`) | 높음 |
| 4 | 자동 재연결 + heartbeat 모니터링 | 중간 |
| 5 | Shadow vs Backtest 시그널 비교 도구 | 중간 |
| 6 | Paper Trading 성과 리포트 생성 | 낮음 |
| 7 | Discord 알림 통합 (주문, 에러, 일간 리포트) | 낮음 |

---

## Phase 6: Live Trading

### 6.1 Go-Live 기준

| 기준 | 임계값 |
|------|-------|
| Paper Trading 기간 | 최소 1개월 |
| Paper Sharpe | > 0 (시장 상황에 따라 유연하게) |
| 시스템 가동률 | > 99.5% |
| 시그널 일관성 | Shadow vs Backtest > 90% |
| 치명적 버그 | 0건 |

### 6.2 Canary Deployment 전략

```
Week 1-2:  전체 자본의 5%로 실거래 (8에셋 × 최소 단위)
Week 3-4:  문제 없으면 10%로 증액
Month 2:   20%로 증액
Month 3+:  50% → 100% 점진적 확대
```

### 6.3 운영 안전장치

```python
# 시스템 레벨 안전장치
class LiveSafetyConfig(BaseModel):
    max_daily_loss_pct: float = 0.05     # 일 최대 손실 5% → 전 포지션 청산
    max_drawdown_pct: float = 0.15       # MDD 15% → 시스템 중단
    max_order_size_usd: Decimal           # 단일 주문 최대 금액
    max_open_positions: int = 8           # 최대 동시 포지션 수
    kill_switch: bool = False             # 긴급 중단 스위치
    allowed_symbols: list[str]            # 화이트리스트
```

### 6.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `LiveExecutor` — 실거래 주문 실행 | 높음 |
| 2 | 안전장치 시스템 (`LiveSafetyConfig`) | 높음 |
| 3 | Kill switch (Discord 명령 또는 API) | 중간 |
| 4 | Canary 자본 관리 로직 | 중간 |
| 5 | 장애 복구 — 미체결 주문 정리, 포지션 동기화 | 높음 |

---

## Phase 7: 모니터링 & 분석

### 7.1 모니터링 스택

```
┌─────────────────────────────────────────────┐
│              Streamlit Dashboard              │
│  전략 성과 분석, PnL 곡선, 파라미터 탐색       │
│  (개발/리서치 단계부터 사용)                    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Prometheus + Grafana               │
│  시스템 헬스, 주문 latency, API 상태           │
│  (Paper Trading 단계부터 사용)                 │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Discord Alerts                   │
│  주문 체결, 에러, 일간 리포트, Kill Switch      │
│  (Shadow 단계부터 사용)                        │
└─────────────────────────────────────────────┘
```

### 7.2 메트릭 체계

#### A. 전략 성과 메트릭 (Streamlit)

| 메트릭 | 갱신 주기 | 용도 |
|--------|---------|------|
| 일간/주간/월간 PnL | 일봉 마감 시 | 수익성 추적 |
| Rolling Sharpe (30d, 90d) | 일간 | 성과 안정성 |
| Current Drawdown | 실시간 | 리스크 모니터링 |
| 심볼별 수익 기여도 | 일간 | 포트폴리오 분해 |
| 백테스트 vs 실거래 괴리 | 주간 | 전략 유효성 확인 |
| 시그널 빈도 | 일간 | 전략 행동 변화 감지 |

#### B. 시스템 운영 메트릭 (Grafana)

| 메트릭 | 알림 임계값 | 용도 |
|--------|-----------|------|
| WebSocket 연결 상태 | 연결 끊김 > 30초 | 데이터 피드 안정성 |
| 주문 응답 시간 (p50, p99) | p99 > 5초 | 체결 품질 |
| API Rate Limit 사용률 | > 80% | 요청 최적화 |
| 프로세스 메모리 | > 2GB | 메모리 누수 감지 |
| 이벤트 처리 지연 | > 1초 | 파이프라인 병목 |

#### C. Discord 알림 체계

| 이벤트 | 알림 레벨 | 내용 |
|--------|---------|------|
| 주문 체결 | INFO | 심볼, 방향, 수량, 가격 |
| 리밸런싱 | INFO | 변경된 포지션 목록 |
| 일간 리포트 | INFO | PnL, 포지션, Sharpe |
| 시스템 에러 | WARN | 에러 메시지, 스택트레이스 |
| 안전장치 트리거 | CRITICAL | max_loss, kill_switch |
| 연결 끊김 | CRITICAL | WebSocket, API 장애 |

### 7.3 작업 목록

| # | 작업 | 도입 시점 |
|---|------|---------|
| 1 | Streamlit 대시보드 — 백테스트 결과 시각화 | Phase 2부터 |
| 2 | Streamlit — 멀티에셋 포트폴리오 분석 뷰 | Phase 2 |
| 3 | Prometheus 메트릭 노출 (`prometheus_client`) | Phase 5 |
| 4 | Grafana 대시보드 템플릿 | Phase 5 |
| 5 | Discord 알림 시스템 확장 | Phase 5 |
| 6 | 백테스트 vs 실거래 괴리 분석 도구 | Phase 6 |
| 7 | 일간/주간 성과 자동 리포트 | Phase 6 |

---

## 기술 스택 결정 (2026년 기준 트렌드 반영)

### 유지하는 것

| 기술 | 이유 |
|------|------|
| **VectorBT** | 파라미터 스윕/벡터화 백테스트에서 대체 불가 성능 |
| **Pydantic V2** | 타입 안전성, Config/DTO 표준 |
| **CCXT Pro** | 거래소 통합 (WebSocket + REST) |
| **Loguru** | 구조화 로깅 |
| **Numba** | Numba-optimized 전략 함수 (이미 구현됨) |

### 새로 도입하는 것

| 기술 | Phase | 이유 |
|------|-------|------|
| **asyncio TaskGroup** | Phase 4 | 구조적 동시성 (Python 3.13 기본) |
| **Streamlit** | Phase 2 | Python-native 대시보드, 빠른 개발 |
| **Prometheus + Grafana** | Phase 5 | 프로덕션 시스템 모니터링 표준 |
| **Polars** (선택적) | Phase 2+ | Silver 데이터 로딩 10x 가속, `.to_pandas()` 호환 |

### 도입하지 않는 것

| 기술 | 이유 |
|------|------|
| **NautilusTrader** | 프레임워크 전환 비용 > 이점. 자체 EDA가 유연성 높음 |
| **MLflow** | 현 단계에서 과도한 인프라. 전략 수가 적음 (1개) |
| **Event Sourcing DB** | EventBus + 로컬 이벤트 로그로 충분. 규모가 커지면 재검토 |
| **Free-threaded Python** | 2026년 기준 NumPy/pandas 생태계 미성숙. 2027+ 재검토 |

---

## 전체 타임라인 (예상)

```
Phase 2: 멀티에셋 백테스트 ─────────┐
Phase 3: 고급 검증 (CPCV/WFA) ──────┤  ← 병렬 가능
Phase 7-A: Streamlit 대시보드 ──────┘

Phase 4: EDA 시스템 ────────────────── ← Phase 2,3 완료 후

Phase 5: Dry Run ───────────────────── ← Phase 4 완료 후
Phase 7-B: Grafana 모니터링 ────────── ← Phase 5와 병행

Phase 6: Live Trading ─────────────── ← Phase 5 검증 통과 후
Phase 7-C: 운영 분석 도구 ──────────── ← Phase 6과 병행
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-02-06 | 초기 로드맵 작성 — Phase 2~7 설계 |
