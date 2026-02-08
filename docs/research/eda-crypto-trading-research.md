# EDA 기반 암호화폐 자동매매 시스템 종합 리서치

> **작성일**: 2026-02-07
> **목적**: Event-Driven Architecture 기반 코인 자동매매 시스템 구축 시 필수 고려사항, 주요 실수, 필수 기능에 대한 전문 리서치
> **대상 시스템**: MC Coin Bot (Python 3.13, CCXT Pro, VectorBT, Pydantic V2)

---

## 목차

1. [EDA 아키텍처 설계](#1-eda-아키텍처-설계)
2. [핵심 인프라 패턴](#2-핵심-인프라-패턴)
3. [리스크 관리 시스템](#3-리스크-관리-시스템)
4. [주문 관리 시스템 (OMS)](#4-주문-관리-시스템-oms)
5. [거래소 연결 및 데이터 인프라](#5-거래소-연결-및-데이터-인프라)
6. [백테스팅 함정과 검증 프레임워크](#6-백테스팅-함정과-검증-프레임워크)
7. [Crypto 특화 전략 고려사항](#7-crypto-특화-전략-고려사항)
8. [라이브 트레이딩 전환](#8-라이브-트레이딩-전환)
9. [모니터링 및 운영](#9-모니터링-및-운영)
10. [흔한 실수 Top 20](#10-흔한-실수-top-20)
11. [필수 기능 체크리스트](#11-필수-기능-체크리스트)
12. [참고 자료](#12-참고-자료)

---

## 1. EDA 아키텍처 설계

### 1.1 Event Bus 패턴 선택

| 패턴 | 특성 | 적합 시나리오 |
|------|------|---------------|
| **Publish/Subscribe** | 1:N broadcast, 느슨한 결합 | Market data → 여러 전략 동시 전달 |
| **Message Queue** | 1:1 point-to-point, 정확히 한 번 처리 | 주문 실행 (중복 방지) |
| **In-process async Queue** | 최저 latency, 단일 프로세스 | 백테스트, 소규모 라이브 |
| **Redis Streams / Kafka** | 프로세스 간 통신, 영속성 | 분산 환경, 멀티 전략 |
| **LMAX Disruptor** | Lock-free ring buffer, 초저지연 | HFT급 시스템 |

**권장**: 단일 프로세스에서는 `asyncio.Queue` 기반 in-process Pub/Sub이 최적. 분산 환경 전환 시 Redis Streams로 교체.

### 1.2 Event Sourcing vs Event-Driven Architecture

두 개념은 **완전히 직교(orthogonal)**하며 독립적으로 또는 함께 사용 가능.

| 구분 | Event-Driven Architecture | Event Sourcing |
|------|--------------------------|----------------|
| **목적** | 컴포넌트 간 비동기 통신 | 상태 변경의 불변 이력 저장 |
| **핵심** | 이벤트로 서비스 간 decoupling | 이벤트 시퀀스로 상태 재구성 |
| **트레이딩 적용** | 실시간 시그널-주문 흐름 (필수) | 감사 추적, 백테스트 리플레이 (보완적) |

**권장**: EDA를 주요 패턴으로 사용하되, JSONL audit log로 Event Sourcing의 기본적인 역할을 수행.

### 1.3 Backpressure 관리

고빈도 데이터 스트림에서 downstream 컴포넌트의 처리 속도가 upstream을 따라잡지 못할 때 **backpressure** 제어가 필수.

**2-Tier Backpressure 정책 (권장)**:

```
NEVER_DROP_EVENTS (await put — 큐 여유까지 대기):
  - SIGNAL, FILL, ORDER_REQUEST, ORDER_ACK, ORDER_REJECTED
  - POSITION_UPDATE, BALANCE_UPDATE, CIRCUIT_BREAKER

DROPPABLE_EVENTS (put_nowait — 큐 가득 시 드롭):
  - BAR, HEARTBEAT, RISK_ALERT
```

**원칙**: "시장 데이터는 stale해도 되지만, 주문/체결은 절대 유실 불가"

추가 전략:
- **Bounded Queue**: 큐 크기 제한으로 backpressure 감지 지점 설정
- **Dead Letter Queue**: handler 에러 시 실패 이벤트를 별도 저장하여 후분석
- **Circuit Breaker**: 연속 실패 시 일시 중단하여 연쇄 장애 방지

### 1.4 컴포넌트 Decoupling 원칙

**핵심**: "Strategy decides **what** to do, Execution handles **how** to do it reliably."

```
Market Data Ingestion
    → Strategy Engine (시그널만 생성, 실행 로직 없음)
    → Portfolio Manager (포지션/잔고 상태 관리, 주문 생성)
    → Risk Manager (pre-trade 검증, 시스템 stop-loss)
    → Order Management System (멱등성 보장, Executor 라우팅)
    → Executor (백테스트/라이브 교체 가능)
```

**Decoupling 메커니즘**:
- 모든 컴포넌트가 EventBus를 통해서만 통신
- `register(bus)` 패턴으로 동적 구독
- Executor는 Protocol(인터페이스)로 정의되어 구현체 교체 가능
- Events는 flat models (상속 없이 독립적 이벤트 타입)

**Anti-Pattern 주의**:
- **Distributed Monolith**: microservice처럼 보이지만 tightly coupled
- **Shared Database**: 여러 서비스가 같은 DB를 직접 읽고 쓰기
- **Temporal Coupling**: 서비스 간 실행 순서 의존

### 1.5 상태 관리

**무상태 전략 / 유상태 실행 (Stateless Strategy / Stateful Execution)**:

```python
# 무상태: StrategyEngine은 bar 데이터로 시그널만 생성
signal = strategy.generate_signal(bar_data)

# 유상태: PM이 Position dataclass로 포지션 상태 관리
@dataclass
class Position:
    symbol: str
    direction: Direction
    size: float
    avg_entry_price: float
    # mark-to-market, ATR, peak/trough 등
```

**원칙**:
- PM이 유일한 상태 소유자 (Single Source of Truth)
- Equity, leverage, drawdown은 기본 상태에서 실시간 계산 (파생 상태)
- `flush()` 메커니즘으로 bar-by-bar 동기 처리 보장

---

## 2. 핵심 인프라 패턴

### 2.1 Event Ordering & Causality

분산 시스템에서 이벤트 순서 보장은 핵심 과제.

**인과 관계 추적**: 모든 이벤트에 `correlation_id`를 포함하여 `Bar → Signal → OrderRequest → Fill` 체인 추적.

**Bar-by-bar flush 패턴 (핵심)**:
```python
async def flush(self) -> None:
    """큐의 모든 이벤트가 처리될 때까지 대기."""
    await self._queue.join()
```

DataFeed가 각 bar 이후 `flush()`를 호출하여 이벤트 체인이 완료된 후에야 다음 bar를 발행. 이는 bar-by-bar 인과적 순서를 보장하는 핵심 메커니즘.

**flush 없이 발생하는 문제**: BAR 이벤트가 DROPPABLE(put_nowait)이므로 DataFeed가 모든 bar를 한 번에 큐에 넣으면, 파생 이벤트가 마지막 bar 가격으로만 체결됨.

### 2.2 멱등성 (Idempotency)

"동일 요청을 여러 번 보내도 결과가 동일"해야 하는 원칙. 네트워크 장애 시 재전송으로 인한 이중 체결 방지.

```python
# client_order_id 기반 중복 방지 (업계 표준)
if order.client_order_id in self._processed_orders:
    logger.warning("Duplicate order ignored")
    return
self._processed_orders.add(order.client_order_id)
```

**주의**: `_processed_orders` set이 무한히 커질 수 있음 → TTL 기반 만료 또는 sliding window 적용 필요.

### 2.3 Circuit Breaker 패턴

사전 정의된 임계값 초과 시 트레이딩을 일시 중단하는 안전장치.

**3단계 방어 체계 (권장)**:

| Level | 트리거 | 대응 |
|-------|--------|------|
| **Position SL** | 개별 포지션 손절 | 해당 포지션 청산 |
| **Trailing Stop** | Peak 대비 ATR*mult 하락 | 해당 포지션 청산 + 재진입 방지 |
| **System CB** | Peak equity 대비 전체 drawdown | 전량 청산 + 거래 중단 |

**Cascade 방어**:
- `_stopped_this_bar` guard: stop-loss 발동 후 같은 bar에서 재진입 방지
- `validated` flag: RM과 OMS 사이의 무한루프 방지
- CB 발동 시 OMS가 RM을 **우회**하여 직접 청산 (CB가 RM 검증 자체를 무력화하므로)

### 2.4 CEP (Complex Event Processing)

여러 이벤트 스트림에서 실시간 패턴을 감지하는 기술:
- 실시간 거래량 vs 과거 거래량 비교
- 섹터/인덱스 움직임 대비 개별 종목 이탈 감지
- Multi-timeframe 교차 분석

**적용 가능 영역**: CandleAggregator가 이미 1m → target TF 집계라는 기본적인 CEP를 수행. 향후 cross-asset correlation 감지 등에 확장 가능.

---

## 3. 리스크 관리 시스템

### 3.1 Position Sizing 알고리즘

| 알고리즘 | 수식 | Crypto 적합도 | 비고 |
|----------|------|---------------|------|
| **Fixed Fractional** | `risk = account * R%` | 높음 (baseline) | 단순, 안정적 |
| **Kelly Criterion** | `f* = (bp - q) / b` | 위험 (full Kelly) | 50%+ drawdown 가능 |
| **Fractional Kelly** | `f = k * f*` (k=0.1~0.25) | 중간 | Kelly + drawdown 제한 |
| **Volatility Targeting** | `size = target_vol / asset_vol` | **매우 높음** | 변동성 정규화, 자동 regime 적응 |
| **ATR-based** | `size = R$ / (ATR * mult)` | 높음 | 시장 변동성 적응 |

**권장**: Volatility Targeting을 기본으로 사용. Per-trade risk은 equity의 1~2%.

**Crypto에서의 Kelly 주의점**:
- Full Kelly는 crypto에서 50% 이상 drawdown 가능 → 10~25% Kelly만 사용
- Win rate/payoff ratio 추정 오류가 크면 Kelly가 오히려 해로움
- Trending market에서는 Kelly 우위, choppy market에서는 Fixed Fractional이 안전

### 3.2 Multi-Level Stop-Loss 전략

| 유형 | 설명 | 구현 우선순위 |
|------|------|---------------|
| **Fixed % SL** | Entry price 기준 고정 % | 필수 (구현 완료) |
| **ATR Trailing Stop** | Peak - ATR*mult 동적 추적 | 필수 (구현 완료) |
| **Time-based Stop** | N봉 경과 후 미수익 시 청산 | P1 |
| **Break-even Stop** | 수익 도달 시 SL을 entry로 이동 | P2 |
| **Layered/Partial Exit** | 단계별 부분 청산 (33%/33%/34%) | P2 |

**ATR Trailing Stop 핵심 파라미터**:
- ATR multiplier 3.0x가 MDD 방어에 최적 (MDD -21.9% → -17.5%)
- ATR period 14가 일봉에서 표준
- Intrabar(1m) 모니터링으로 정밀도 향상

### 3.3 Multi-Level Circuit Breaker (권장 확장)

현재 Level 3(전량 청산)만 구현. 점진적 리스크 축소를 위해 확장 필요:

```
Level 1 (5% drawdown):  WARNING + 포지션 크기 50% 축소
Level 2 (8% drawdown):  CRITICAL + 신규 진입 금지
Level 3 (10% drawdown): EMERGENCY + 전량 청산 + cooldown 30분
```

### 3.4 Daily Loss Limit (필수 추가)

인트라데이 안전장치로, 일일 최대 손실을 제한:

```python
daily_pnl = current_equity - start_of_day_equity
if daily_pnl < -daily_loss_limit:  # 예: -3~5%
    trigger_daily_circuit_breaker()
```

### 3.5 Correlation-Based Risk

**Crypto 상관관계의 특성**:
- 평상시: BTC-altcoin 평균 잔여 상관관계 ~0.29 (분산 효과 존재)
- 위기 시: 상관관계 급격히 1에 수렴 (분산 효과 소멸, herding behavior)

**권장**:
```python
# Rolling correlation 모니터링
corr_matrix = returns_df.rolling(30).corr()
avg_pairwise_corr = corr_matrix.mean().mean()

# 상관관계 급등 시 포지션 규모 축소
if avg_pairwise_corr > 0.7:
    risk_scalar *= 0.5
```

### 3.6 Leverage 관리 및 Margin 모니터링

**Binance Futures 특화 리스크**:

| 리스크 | 설명 | 대응 |
|--------|------|------|
| **Liquidation** | Maintenance margin 미달 시 강제 청산 | Margin ratio 모니터링 필수 |
| **ADL** | Insurance fund 부족 시 수익 포지션 강제 청산 | ADL indicator 모니터링 |
| **Funding Rate** | 8시간마다 long/short 간 자금 이전 | FundingRateEvent 구현 |
| **Mark vs Last Price** | Liquidation은 mark price 기준 | Mark price 기반 계산 |

**안전 마진 권장**:
```
MAX_MARGIN_USAGE_RATIO = 0.6    # 마진의 60%만 사용
MARGIN_WARNING_RATIO = 0.5      # 50% 초과 시 경고
MARGIN_CRITICAL_RATIO = 0.7     # 70% 초과 시 포지션 축소
Initial max leverage = 1.5x     # 검증 후 2.0x로 확대
```

### 3.7 Funding Rate 리스크

Perpetual futures에서 funding rate은 숨겨진 비용/수익:
- 극단적 강세장: long funding rate 0.1%/8h = **연간 ~109%** 비용
- 이를 무시하면 backtest에서 과대 수익 추정
- 연간 funding cost 추산: `funding_rate * 3 * 365`

**Alert 기준**: Funding rate > 0.05%/8h (연간 ~54%) 시 포지션 방향 재검토.

### 3.8 핵심 리스크 파라미터 요약

| 파라미터 | 권장 초기값 | 근거 |
|----------|------------|------|
| Per-trade risk | 1~2% of equity | 업계 표준, ruin 확률 최소화 |
| Max leverage (aggregate) | 1.5x (초기), 2.0x (검증 후) | Crypto 변동성 고려 |
| System stop-loss (MDD) | 10% | Level 3 CB |
| Daily loss limit | 3~5% | 인트라데이 안전장치 |
| Margin usage ratio | max 60% | Liquidation buffer |
| Trailing stop ATR mult | 3.0x | MDD -17.5% (최적) |
| Rebalance threshold | 10% | 거래비용 절감 |
| Funding rate alert | > 0.05%/8h | 연간 ~54% 비용 |
| Heartbeat interval | 30초 | 장애 감지 시간 |
| Cooldown after CB | 30분 | 급변동 안정화 대기 |
| Order TTL | 5분 | Stale order 방지 |

---

## 4. 주문 관리 시스템 (OMS)

### 4.1 필수 기능 목록

| 기능 | 설명 | 우선순위 |
|------|------|----------|
| **Idempotency** | client_order_id 기반 중복 방지 | P0 (구현 완료) |
| **Executor Routing** | 백테스트/라이브 교체 가능 | P0 (구현 완료) |
| **CB 전량 청산** | Circuit breaker 시 모든 포지션 청산 | P0 (구현 완료) |
| **Order State Machine** | PENDING → PARTIAL → FILLED/CANCELLED | P0 |
| **Partial Fill Handling** | 부분 체결 상태 관리 | P1 |
| **Order Reconciliation** | 거래소 vs 내부 포지션 일치 확인 | P1 |
| **Rate Limiting** | Token Bucket 기반 API 쿼터 관리 | P1 |
| **Retry Logic** | API 장애 시 exponential backoff | P1 |
| **Order Timeout** | Stale order 자동 취소 | P1 |

### 4.2 Order State Machine

라이브 트레이딩에서 주문은 복잡한 생명주기를 가짐:

```
PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED
                  ↘ REJECTED
                  ↘ CANCELLED
                  ↘ EXPIRED
```

### 4.3 Partial Fill 처리

```python
# 라이브에서의 Partial fill 시나리오
OrderRequest(qty=1.0) → Fill(qty=0.3) → Fill(qty=0.5) → Fill(qty=0.2)

# PM은 각 partial fill마다 position 업데이트
# OMS는 remaining quantity 추적
```

### 4.4 API Rate Limiting

**Binance Futures Rate Limit**:
- REST API: 1200 weight/min (주문은 각 1~5 weight)
- WebSocket: 10 msg/sec
- Order limit: 300/min per account

```python
# Token Bucket 알고리즘 권장
class RateLimiter:
    def __init__(self, max_tokens: int, refill_rate: float):
        self._tokens = max_tokens
        self._max = max_tokens
        self._refill_rate = refill_rate  # tokens/sec

    async def acquire(self, weight: int = 1) -> None:
        while self._tokens < weight:
            await asyncio.sleep(0.1)
        self._tokens -= weight
```

**추가 권장**: 20~30% headroom 유지 (limit의 70~80%만 사용), 비상 시 주문 실행 우선.

---

## 5. 거래소 연결 및 데이터 인프라

### 5.1 WebSocket vs REST API

| 항목 | WebSocket | REST API |
|------|-----------|----------|
| **Latency** | 10~100ms (실시간) | 100~500ms (polling) |
| **Rate Limit** | 연결 수 제한만 | Weight 기반 제한 |
| **적합 용도** | Market data, order book | Order execution, 스냅샷 복구 |
| **데이터 보장** | 유실 가능 (disconnect 시) | 요청-응답 보장 |

**권장**: "WebSocket for data, REST for execution"

### 5.2 Reconnection 전략

```python
# Exponential backoff 패턴
RECONNECT_DELAYS = [1, 2, 4, 8, 16, 30, 60]  # 초

async def watch_with_reconnect(exchange, symbol: str):
    attempt = 0
    while True:
        try:
            data = await exchange.watch_order_book(symbol)
            attempt = 0  # 성공 시 리셋
            yield data
        except ccxt.NetworkError:
            delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
            await asyncio.sleep(delay)
            attempt += 1
```

**Binance WebSocket 요구사항**:
- 서버가 3분마다 ping frame 전송, 10분 이내 pong 미응답 시 연결 종료
- 연결당 최대 구독 수 제한 있음
- 재연결 후 REST API로 snapshot 복구 필수

### 5.3 CCXT Pro 사용 시 주의사항

- `watchX()` 중단 시 `unWatchX()` 호출 필수 (백그라운드 스트림 유지됨)
- 캐싱 레이어 기본 1000 entries 제한
- Exchange-specific 호환성 이슈 존재
- `adjustForTimeDifference: True` 설정으로 시간 동기화

### 5.4 Data Pipeline (Medallion Architecture)

| Layer | 역할 | 포맷 |
|-------|------|------|
| **Bronze** | 원본 데이터 (raw, 변환 없음) | Parquet |
| **Silver** | 정제 데이터 (gap filling, dedup) | Parquet |
| **Gold** | 분석 데이터 (feature, signal) | DataFrame |

**라이브 확장 권장**:
- Real-time ingestion: Kafka/Redis Streams로 Bronze 실시간 적재
- Data versioning: DVC 또는 LakeFS로 데이터 lineage 추적

### 5.5 Data Quality Validation

**필수 검증 항목**:
- NaN/Inf 검사
- `high < low` 검사
- Timestamp 연속성 검증 (gap detection)
- Volume 이상치 탐지 (평균 대비 10x 이상)
- Price spike 탐지 (이전 bar 대비 급변)
- Stale data 탐지 (동일 OHLC 반복)
- Cross-exchange 가격 일관성 검증

### 5.6 흔한 연결 장애 및 해결책

| 장애 | 원인 | 해결책 |
|------|------|--------|
| WebSocket 끊김 | 급변동 시 거래소 부하 | Exponential backoff + state recovery |
| Rate limit 초과 | API 과다 호출 | Token bucket + 20% headroom |
| IP ban (HTTP 418) | 반복 위반 | Adaptive throttling |
| REST/WS 데이터 불일치 | 전파 지연 | 주기적 reconciliation loop |
| Timestamp 오차 | 로컬 시계 drift | `adjustForTimeDifference: True` + NTP |
| Exchange maintenance | 예정/비예정 점검 | Health monitor + 알림 + 안전 모드 |

---

## 6. 백테스팅 함정과 검증 프레임워크

### 6.1 Look-Ahead Bias (미래 정보 편향)

**정의**: 의사결정 시점에 사용 불가능했던 미래 정보를 backtesting에서 사용하는 오류.

**경고 신호**:
- 지나치게 매끄러운 equity curve
- 연환산 수익률 > 12% (비정상적으로 높음)
- Sharpe Ratio > 1.5 (의심 구간)

**방지 전략**:
1. **Event-driven backtesting** (bar-by-bar 처리) — EDA 아키텍처가 구조적으로 방지
2. **Point-in-time 데이터** — 당시 알 수 있었던 정보만 사용
3. **코드 리뷰 체크리스트**: 모든 feature 계산에서 `shift()` 누락 여부 확인

### 6.2 Survivorship Bias (생존 편향)

Crypto에서 특히 심각. 수천 개 token이 매년 launch되고 delist/rug-pull됨.

**해결책**:
- 과거 시점의 전체 market universe snapshot 유지
- Delist된 asset을 historical dataset에 포함
- Universe 구성 시 look-ahead 방지 (당시 top-N만 선택)

### 6.3 Overfitting / Curve Fitting

**경고 지표**:

| Metric | 현실적 범위 | Overfitting 의심 |
|--------|------------|------------------|
| Profit Factor | 1.5 ~ 2.0 | > 3.0 |
| Sharpe Ratio | 0.5 ~ 1.5 | > 3.0 |
| Sortino Ratio | 1.0 ~ 2.0 | > 3.0 |
| Win Rate | 40% ~ 60% | > 80% |

**방지 원칙**:
1. **파라미터 최소화**: rule of thumb: 데이터 포인트 / 파라미터 > 252
2. **80/20 train/test split**: IS 80% 학습, OOS 20% 검증
3. **다중 자산/시간프레임 검증**: 하나의 전략이 여러 asset에서 작동하는지 확인
4. **단순 전략 선호**: 복잡한 규칙은 noise에 적합할 확률이 높음

### 6.4 Transaction Cost Modeling

**현실적 비용 구조 (Crypto)**:

| Component | Conservative | Moderate | Optimistic |
|-----------|-------------|----------|------------|
| Round-trip 비용 | 0.5% | 0.3% | 0.15% |
| Maker fee | 0.1% | 0.02% | 0.01% |
| Taker fee | 0.2% | 0.05% | 0.03% |
| Funding rate (perp) | 0.01%/8h | 0.005%/8h | 0.003%/8h |

**Slippage — Square Root Market Impact Law**:
```
Impact = sigma * sqrt(Q / V) * pi
```
- `sigma`: 일일 변동성 (crypto 기본 ~2%)
- `Q`: 주문 크기
- `V`: 일평균 거래량
- `pi`: permanent impact 계수 (~0.1)

**핵심**: Slippage는 비선형 — 주문 크기에 비례가 아닌 제곱근에 비례. 항상 conservative 가정으로 backtest.

### 6.5 Walk-Forward Analysis (WFA)

Traditional train/test split의 한계를 극복하는 rolling validation:

1. Training window (6개월)에서 파라미터 최적화
2. Test window (1개월)에서 OOS 성능 평가
3. Window를 앞으로 slide하여 반복
4. 모든 OOS 구간을 연결하여 최종 평가

**Walk-Forward Efficiency (WFE)**:
```
WFE = Average Test Sharpe / Average Train Sharpe
```

| WFE | 해석 |
|-----|------|
| > 0.5 | Healthy (robust) |
| 0.3 ~ 0.5 | Acceptable (주의 필요) |
| < 0.3 | Poor/Overfit (재검토) |

**Crypto 특수**: Sliding window가 expanding window보다 적합 (regime shift 빈번). Window 크기는 최소 1 market cycle (1~2년).

### 6.6 CPCV (Combinatorially Purged Cross-Validation)

Marcos Lopez de Prado가 개발한 금융 시계열 전용 cross-validation:

1. **Purging**: training sample 중 test set과 정보가 겹치는 sample 제거 (look-ahead bias 방지)
2. **Embargo**: test set 경계에 buffer 구간 설정 (추가 안전장치)
3. **Combinatorial paths**: N개 group에서 k개 test group을 선택하는 모든 조합 생성

**핵심 강점**: 다양한 market regime 조합에 대해 파라미터 검증. 단일 값이 아닌 **분포**로 평가.

### 6.7 PBO & DSR

**PBO (Probability of Backtest Overfitting)**:
- IS에서 최적으로 선택된 전략이 OOS에서 median 이하 성능을 보일 확률
- CSCV 방법으로 추정

| PBO | 해석 |
|-----|------|
| < 0.1 | 전략 선택 과정이 robust |
| 0.1~0.3 | 약간의 overfitting 우려 |
| 0.3~0.5 | 상당한 overfitting 위험 |
| > 0.5 | 전략 선택 과정 자체가 unreliable |

**DSR (Deflated Sharpe Ratio)**:
- Selection bias (multiple testing)와 non-normal return 분포를 보정
- 시도한 전략의 수가 많을수록 DSR 감소
- 진정한 statistical significance vs statistical fluke 분리

### 6.8 Monte Carlo Simulation

**주요 방법**:
1. **Trade shuffling**: 거래 순서 랜덤 재배치 → drawdown 분포 추정
2. **Trade skipping**: 일부 거래 랜덤 제거 (실전 미체결 시뮬레이션)
3. **Parameter perturbation**: 파라미터 ±10~20% 변형 → 민감도 분석
4. **Bootstrap resampling**: 거래 결과 복원 추출 → confidence interval

**권장**: 최소 1,000회 이상 시뮬레이션. 95th percentile의 worst-case drawdown을 "expected MDD"로 사용.

---

## 7. Crypto 특화 전략 고려사항

### 7.1 24/7 Market 특성

기존 금융시장과의 근본적 차이:
- Market close가 없음 → overnight gap 분석 불가
- 아시아/유럽/미국 세션별 유동성 변화
- 새벽 3시에 market-moving news 발생 가능
- 주말/공휴일에도 거래 지속

**전략적 시사점**:
- 24시간 리스크 모니터링 필수 → 자동화된 kill-switch
- Daily candle의 "시작/끝" 기준을 UTC 00:00으로 표준화
- 저유동성 시간대 (UTC 10:00~14:00) 진입 회피 고려

### 7.2 Extreme Volatility Regimes

**Crypto 변동성의 특수성**:
- BTC 일일 변동성: 평균 ~2%, 극단 시 >10%
- 뚜렷한 volatility regime: high-vol, moderate, calm
- FOMC 발표, 규제 뉴스 등 매크로 이벤트에 극도로 민감

**Regime Detection (HMM 기반)**:
- Hidden Markov Model로 low-vol / high-vol 두 상태 식별
- 전략별 regime 적합성:
  - **Momentum**: bull/고vol에서 강, bear 전환기 약
  - **Mean-reversion**: sideways/저vol에서 강
  - **Volatility targeting**: 모든 regime에서 안정적 (자체 regime filter 역할)

### 7.3 Liquidity Fragmentation

- DEX/CEX 비율: spot 23%, futures 9%로 사상 최고치
- 동일 asset이 50개 이상 venue에서 거래
- 단일 exchange 데이터로 backtest하면 실제 유동성 과대 평가

### 7.4 Funding Rate Arbitrage

**기본 메커니즘**: Spot long + perp short = delta-neutral + funding 수취

**3대 Arbitrage 경로**:
1. **Hyperliquid ↔ CEX**: 가장 liquid, spread 작지만 빈번한 기회
2. **CEX ↔ L2 DEX**: 가장 exploitable, AMM이 fair value에서 20~50bps 이탈
3. **Hyperliquid ↔ L2 DEX**: "juiciest" — Hyperliquid은 fair price 준수, AMM은 미준수

### 7.5 Strategy Degradation 감지

**전략이 "죽는" 이유**:
- **Overfitting decay**: 매년 Sharpe decay가 약 5%p씩 증가
- **Arbitrage crowding**: 알려진 anomaly에 자본이 몰리면서 수익 잠식

**모니터링 지표**:

| 지표 | 경고 임계값 |
|------|------------|
| Rolling Sharpe | IS Sharpe의 50% 이하 |
| Win rate trend | 지속적 하락 |
| Execution cost deviation | > 15% 이탈 |
| Live MDD vs IS MDD | Live > IS MDD |

**전략 중단 기준**:
1. Live drawdown이 training period의 maximum drawdown에 도달
2. Rolling Sharpe가 지속적으로 0 이하
3. 3~6개월 연속 OOS 성능 부진

---

## 8. 라이브 트레이딩 전환

### 8.1 Backtest-to-Live Gap 원인과 대책

| Gap 원인 | 설명 | 대책 |
|----------|------|------|
| Slippage 과소추정 | OHLCV는 intra-candle 정보 미포함 | Square root impact law 적용 |
| Latency | Signal → 주문 도달 시간 지연 | Conservative fill assumption (next-bar open) |
| Partial fill | 대량 주문 부분 체결 | Fill probability model |
| Data quality | Backtest vs live feed 차이 | 동일 source 사용 |
| Market impact | 자신의 주문이 시장가격 영향 | Volume-aware sizing |

### 8.2 단계적 배포 아키텍처

```
Stage 0: Backtest     (Historical data, simulated execution)
    ↓
Stage 1: Shadow Mode  (Live data, simulated execution, 실전과 병렬)
    ↓
Stage 2: Paper Trade  (Live data, exchange paper API)
    ↓
Stage 3: Canary       (Live, 최소 자본 1~5%)
    ↓
Stage 4: Full Deploy  (Live, 목표 자본)
```

**Shadow Mode 핵심**: 두 개의 동일 시스템을 병렬 운영. Shadow는 실전과 동일한 신호를 생성하되 주문은 shadow ledger에 기록. 두 시스템의 outcome 실시간 비교.

### 8.3 Gradual Capital Deployment

| 단계 | 자본 비율 | 기간 | 승급 조건 |
|------|----------|------|-----------|
| Paper trading | 0% | 3~6개월 | Rule compliance 확인 |
| Micro live | 1~5% | 1~3개월 | Paper와 일관된 결과 |
| Small live | 10~20% | 3~6개월 | Sharpe > 0, MDD < 예상 |
| Medium live | 50% | 3~6개월 | 전 단계 성능 유지 |
| Full deployment | 100% | 지속 | 지속적 모니터링 |

**승급 기준**:
- Return > 0
- MDD < backtest MDD × 1.5
- Execution cost < 예상 × 1.2
- Win rate > backtest의 80%
- 최소 50회 이상 거래 샘플

**핵심 원칙**: Live 결과가 paper test에 의미 있게 뒤처지면 즉시 중단 후 리뷰. 절대로 loss 만회를 위해 자본을 늘리지 않음.

### 8.4 Performance Attribution (Live)

```
Total Return = Known Premia + Residual Alpha + Costs + Slippage + Timing
```

**모니터링 대시보드 필수 항목**:
- P&L breakdown (gross vs net)
- Slippage distribution (per trade)
- Fill rate 및 partial fill 비율
- Latency histogram
- Risk metrics (VaR, CVaR, leverage)

---

## 9. 모니터링 및 운영

### 9.1 Health Check 계층

```
Level 1 (매 10초): System Health
  - CPU/Memory/Disk 사용률
  - 프로세스 alive 확인
  - EventBus queue depth

Level 2 (매 30초): Exchange Health
  - WebSocket 연결 상태
  - 마지막 market data 수신 시각
  - API rate limit 잔여량

Level 3 (매 1분): Trading Health
  - 미체결 주문 수 / 체결률
  - 포지션 크기 / 마진 사용률
  - P&L / drawdown 상태
```

### 9.2 모니터링 Metrics

| Category | Metric | Alert Threshold |
|----------|--------|----------------|
| Latency | Order-to-fill latency | > 5s |
| Latency | Market data lag | > 3s stale |
| Fill | Fill rate (체결률) | < 80% |
| Fill | Slippage vs expected | > 0.1% |
| Risk | Unrealized P&L | drawdown > 5% |
| Risk | Position exposure | > max limit |
| System | EventBus queue depth | > 80% capacity |
| System | Handler error count | > 0/min |
| Exchange | Rate limit usage | > 70% |
| Exchange | WebSocket reconnects | > 3/hour |

### 9.3 Alerting 전략

| 심각도 | 채널 | 예시 |
|--------|------|------|
| **CRITICAL** | Discord + SMS | CB 발동, 거래소 연결 실패, liquidation 위험 |
| **WARNING** | Discord | Drawdown > 3%, high latency, rate limit 80% |
| **INFO** | Discord (별도 채널) | 주문 체결, 포지션 변경, 일일 리포트 |
| **DEBUG** | 로그 파일만 | 모든 이벤트 상세 |

### 9.4 Logging Best Practices

**Structured Logging (JSONL)**:
```python
logger.add(
    "logs/trading_{time}.jsonl",
    serialize=True,         # 자동 JSON 변환
    rotation="100 MB",      # 파일 크기 제한
    retention="30 days",    # 보관 기간
    compression="gz",       # 압축
    enqueue=True,           # 비동기 쓰기 (성능)
)
```

**Audit Trail 필수 기록**:
- 모든 market data 수신 (timestamp, symbol, OHLCV)
- 모든 signal 생성 (strategy, direction, size)
- 모든 order 발행/체결/취소 (order_id, fill_price, slippage)
- 모든 risk check 결과 (pass/reject, reason)
- 시스템 상태 변경 (circuit breaker, reconnection)

### 9.5 Kill Switch (Emergency Stop)

자동 트리거 외에 수동 개입 수단이 필수:

```python
# 1. 파일 기반 kill switch (가장 단순)
KILL_SWITCH_FILE = "EMERGENCY_STOP"
if Path(KILL_SWITCH_FILE).exists():
    trigger_emergency_stop()

# 2. Discord/Telegram 명령어 기반

# 3. HTTP endpoint 기반 (FastAPI)

# 4. Heartbeat watchdog
if time_since_last_heartbeat > MAX_HEARTBEAT_INTERVAL:
    trigger_emergency_stop("Heartbeat timeout")
```

---

## 10. 흔한 실수 Top 20

### 아키텍처 실수

| # | 실수 | 결과 | 방지책 |
|---|------|------|--------|
| 1 | **Monolithic 설계** | 전략 변경 시 전체 시스템 재배포 | EDA + 컴포넌트 분리 |
| 2 | **Event storm 미방지** | 연쇄 이벤트로 시스템 마비 | `_stopped_this_bar` guard, validated flag |
| 3 | **flush 없이 bar 발행** | 모든 파생 이벤트가 마지막 bar 가격으로 체결 | `await bus.flush()` 필수 |
| 4 | **상태 동기화 미스** | PM과 실제 거래소 포지션 불일치 | Reconciliation loop |
| 5 | **Event 상속 사용** | pyright 호환 문제, schema evolution 어려움 | Flat event models |

### 리스크 관리 실수

| # | 실수 | 결과 | 방지책 |
|---|------|------|--------|
| 6 | **Kill switch 없음** | 장애 시 포지션 방치 | 파일/HTTP/watchdog 기반 emergency stop |
| 7 | **Daily loss limit 없음** | 하루에 전체 자본 손실 가능 | 일일 손실 한도 (3~5%) |
| 8 | **Single-level CB** | 전량 청산 또는 무대응 양극단 | 3-level circuit breaker |
| 9 | **Funding rate 무시** | 수익이 funding 비용에 잠식 | Funding cost 모니터링 |
| 10 | **Over-leveraging** | Liquidation → 전체 자본 소멸 | Margin ratio 60% 이하 유지 |

### 백테스팅 실수

| # | 실수 | 결과 | 방지책 |
|---|------|------|--------|
| 11 | **Look-ahead bias** | Backtest에서만 수익 | EDA bar-by-bar 처리 |
| 12 | **Survivorship bias** | 과거 성과 과대 평가 | Delist asset 포함 |
| 13 | **Slippage 과소추정** | Live에서 수익 급감 | Square root impact law |
| 14 | **Overfitting** | OOS에서 성능 급락 | WFA WFE > 0.5, PBO < 0.3 |
| 15 | **Flat fee model** | 실제 비용 과소 추정 | Volume-aware slippage + funding |

### 운영 실수

| # | 실수 | 결과 | 방지책 |
|---|------|------|--------|
| 16 | **WebSocket 재연결 미처리** | 급변동 시 데이터 유실 | Exponential backoff + state recovery |
| 17 | **Rate limit 초과** | API ban (2분~3일) | Token bucket + 20% headroom |
| 18 | **Reconciliation 없음** | 내부 상태와 거래소 상태 불일치 | 주기적 snapshot 비교 |
| 19 | **Paper trading 건너뜀** | 예상치 못한 live 문제 | 최소 3개월 paper trading |
| 20 | **Loss 만회 위해 자본 증가** | 더 큰 손실로 이어짐 | 단계적 자본 배포 원칙 고수 |

---

## 11. 필수 기능 체크리스트

### Phase 5 전환 전 필수 (P0)

```
[ ] Multi-level Circuit Breaker (Level 1/2/3 단계화)
[ ] Daily Loss Limit (일일 최대 손실 한도)
[ ] Slippage Model (Volume-aware, square root impact law)
[ ] Order State Machine (PENDING/PARTIAL/FILLED/CANCELLED)
[ ] Manual Kill Switch (파일/HTTP/Discord 기반)
[ ] Heartbeat Watchdog (컴포넌트 헬스체크 + 자동 정지)
[ ] LiveDataFeed (WebSocket 기반, HistoricalDataFeed와 동일 인터페이스)
[ ] LiveExecutor (REST API order execution)
```

### Shadow/Paper 단계 (P1)

```
[ ] Partial Fill Handling (부분 체결 상태 관리)
[ ] Rate Limiter (Token Bucket 기반 API 쿼터)
[ ] Order Reconciliation (거래소 vs 내부 포지션 일치 확인)
[ ] Funding Rate 모니터링 (FundingRateEvent + 비용 추적)
[ ] Margin Ratio 모니터링 (Liquidation 방지)
[ ] Time-based Stop (N봉 미수익 시 청산)
[ ] Reconnection Handler (Exponential backoff + state recovery)
[ ] Discord Alerting (CRITICAL/WARNING/INFO 3단계)
[ ] Strategy Degradation Monitor (Rolling Sharpe, MDI)
```

### Canary/Live 단계 (P2)

```
[ ] Correlation Matrix 모니터링 (위기 시 포지션 축소)
[ ] Regime Detection (HMM 기반 vol regime → 전략 가중치 조절)
[ ] Layered/Partial Exit (단계별 부분 청산)
[ ] Break-even Stop (수익 도달 시 SL을 entry로 이동)
[ ] Order Book Depth 분석 (Market impact 사전 추정)
[ ] Shadow Mode 아키텍처 (Live + shadow 병렬)
[ ] Monte Carlo Robustness Test (worst-case MDD 추정)
[ ] Performance Attribution Dashboard
[ ] Wash Trading Detection (자가 거래 패턴 감시)
[ ] Audit Trail 무결성 검증 (hash chain)
```

### 라이브 전환 최종 체크리스트

```
[ ] WFA WFE > 0.5 확인
[ ] PBO < 0.3 확인
[ ] Monte Carlo 95th percentile MDD 수용 가능
[ ] Transaction cost model에 conservative 가정 적용
[ ] Paper trading 3개월 이상 완료
[ ] Paper vs backtest 결과 비교 → 유의미한 차이 없음
[ ] Micro live (1~5% 자본) 1개월 이상 완료
[ ] Kill-switch 및 circuit breaker 테스트 완료
[ ] 24/7 monitoring alert 설정 완료
[ ] Degradation detection 지표 대시보드 구축
```

---

## 12. 참고 자료

### EDA Architecture
- [Event-Driven Architecture for Trading Systems](https://www.thefullstack.co.in/event-driven-architecture-trading-systems/)
- [Mastering Event-Driven Architecture: Financial Trading Platform](https://solutionsarchitecture.medium.com/)
- [Event Sourcing Pattern - Microsoft Azure](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing)
- [Event-Driven FSM for Distributed Trading System](https://www.quantisan.com/event-driven-finite-state-machine-for-a-distributed-trading-system/)
- [What do you mean by "Event-Driven"? - Martin Fowler](https://martinfowler.com/articles/201701-event-driven.html)
- [AAT: Asynchronous Algorithmic Trading in Python](https://github.com/AsyncAlgoTrading/aat)

### Risk Management
- [Crypto Risk Management Strategies - Changelly](https://changelly.com/blog/risk-management-in-crypto-trading/)
- [Risk Analysis of Crypto Assets - Two Sigma](https://www.twosigma.com/articles/risk-analysis-of-crypto-assets/)
- [Trading System Kill Switch - NYIF](https://www.nyif.com/articles/trading-system-kill-switch-panacea-or-pandoras-box)
- [Risk Before Returns: Position Sizing Frameworks](https://medium.com/@ildiveliu/risk-before-returns-position-sizing-frameworks-fixed-fractional-atr-based-kelly-lite-4513f770a82a)
- [Kelly Criterion for Crypto Traders](https://medium.com/@tmapendembe_28659/kelly-criterion-for-crypto-traders)

### Exchange Connectivity
- [CCXT Pro Manual](https://github.com/ccxt/ccxt/wiki/ccxt.pro.manual)
- [Hummingbot Architecture](https://hummingbot.org/blog/hummingbot-architecture---part-1/)
- [Navigating API Rate Limits in Crypto Trading](https://www.weex.com/news/detail/navigating-api-rate-limits-in-crypto-trading-essential-strategies-for-developers-and-traders-204698)
- [Production-Grade Python Logging with Loguru](https://www.dash0.com/guides/python-logging-with-loguru)

### Backtesting & Validation
- [PBO Paper (SSRN - Bailey et al.)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [Deflated Sharpe Ratio Paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [CPCV Explained](https://quantoisseur.com/2019/11/05/combinatorial-purged-cross-validation-explained/)
- [Walk-Forward Analysis Deep Dive - IBKR](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)
- [Realistic Backtesting Methodology](https://www.hyper-quant.tech/research/realistic-backtesting-methodology)
- [Common Pitfalls in Backtesting](https://medium.com/funny-ai-quant/ai-algorithmic-trading-common-pitfalls-in-backtesting)

### Crypto-Specific
- [Regime Switching Forecasting for Cryptocurrencies](https://link.springer.com/article/10.1007/s42521-024-00123-2)
- [HMM Market Regime Detection](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Funding Rate Arbitrage CEX vs DEX](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- [When Trading Systems Break Down](https://talkmarkets.com/content/stocks--equities/when-trading-systems-break-down-causes-of-decay-and-stop-criteria?post=524322)
- [Strategy Decay Research](https://www.tandfonline.com/doi/full/10.1080/14697688.2022.2098810)

### Production & Monitoring
- [Algorithmic Trading System Architecture - TuringFinance](http://www.turingfinance.com/algorithmic-trading-system-architecture-post/)
- [High-Frequency Trading Infrastructure](https://dysnix.com/blog/high-frequency-trading-infrastructure)
- [Lessons Learned: WebSocket at Scale - DraftKings](https://medium.com/draftkings-engineering/lessons-learned-websocketapi-at-scale-604617a54cdb)
- [Circuit Breakers in Crypto Trading](https://www.lcx.com/circuit-breakers-in-crypto-trading-explained/)
- [Why Most Trading Bots Fail](https://medium.com/@FelosInsights/why-most-trading-bots-fail-and-what-we-do-differently)
- [Lessons from Algo Trading Failures](https://www.luxalgo.com/blog/lessons-from-algo-trading-failures/)

### Notable Trading Failures (교훈)
- **Knight Capital (2012)**: 알고리즘 오작동 → 45분 만에 $4.4억 손실
- **Ethereum Flash Crash (2017)**: GDAX에서 cascade 매도 → ETH $0.10 폭락
- **2010 Flash Crash**: 알고리즘 봇 연쇄 매도 → Dow 1000pt 급락
