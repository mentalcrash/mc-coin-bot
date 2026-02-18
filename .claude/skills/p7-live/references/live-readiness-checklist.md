# Live Readiness Checklist: 7항목 상세 검증 패턴

> EDA 시스템이 라이브 환경(Paper/Live Trading)에서 안전하게 작동하는지 검증하는 코드 레벨 체크리스트.
> VBT-EDA Parity와 별개로, EDA 코드 자체의 라이브 준비 상태를 점검한다.

---

## L1: EventBus Flush 패턴

### 목적

DataFeed가 발행한 BarEvent의 파생 이벤트(Signal → Order → Fill → Balance)가
**모두 처리된 후** 다음 bar를 진행하는지 확인.

### 위험

flush 없이 연속 bar 발행 → 모든 파생 이벤트가 **마지막 bar 가격**으로 체결.
예: 100개 bar를 한 번에 queue에 넣으면, signal/fill이 bar-100의 가격으로만 실행.

### 검증 코드

```bash
# HistoricalDataFeed: TF bar 완성 시 flush
grep -n "flush" src/eda/data_feed.py

# LiveDataFeed: TF bar 완성 시 flush
grep -n "flush" src/eda/live_data_feed.py
```

### PASS 조건

- `HistoricalDataFeed`: TF candle 완성 직후 `await self._bus.flush()` 호출
- `LiveDataFeed`: TF candle 완성 직후 `await self._bus.flush()` 호출
- flush 호출이 bar 발행과 같은 async context에 있음

### 일반적 패턴

```python
# 올바른 패턴 (data_feed.py)
async def _emit_completed_bar(self, ...):
    await self._bus.publish(bar_event)
    if is_tf_complete:
        await self._bus.flush()  # 이벤트 체인 완료 보장
```

### FAIL 시 영향

- **백테스트**: 성과 왜곡 (마지막 bar 가격으로만 체결)
- **라이브**: 동일 문제 발생 → 실시간 가격이 아닌 마지막 처리 가격으로 체결

---

## L2: Executor Bar Handler 순서

### 목적

`executor_bar_handler`가 BAR 이벤트 핸들러 중 **첫 번째**로 등록되어,
이전 bar에서 생성된 pending 주문이 현재 bar에서 체결(fill) → drain되어야
Strategy/PM이 최신 포지션 정보로 동작.

### 위험

순서 잘못 → Strategy가 이전 fill을 모르는 상태에서 시그널 생성 →
PM이 잘못된 포지션 기준으로 리밸런싱.

### 검증 코드

```bash
# Runner에서 BAR handler 등록 순서 확인
grep -n "subscribe.*BAR\|subscribe.*BarEvent\|EventType.BAR" \
  src/eda/runner.py src/eda/live_runner.py
```

### PASS 조건

- `executor_bar_handler` 등록이 `strategy_engine.on_bar` 및 `pm.on_bar`보다 **앞에** 위치
- 등록 순서: executor → strategy → PM → analytics (또는 동등)

### 일반적 패턴

```python
# 올바른 순서 (runner.py)
bus.subscribe(EventType.BAR, executor_bar_handler)  # 1st: fill pending
bus.subscribe(EventType.BAR, strategy_engine.on_bar)  # 2nd: generate signals
bus.subscribe(EventType.BAR, pm.on_bar)  # 3rd: rebalance
```

### FAIL 시 영향

- **백테스트**: 거래 수/수익률 왜곡
- **라이브**: 동일 문제 + 실주문 시 위험

---

## L3: Deferred Execution 일관성

### 목적

BacktestExecutor와 LiveExecutor가 동일한 deferred execution 패턴을 따르는지 확인.
백테스트에서 검증된 실행 로직이 라이브에서도 동일하게 작동해야 함.

### 패턴

| 주문 유형 | price 필드 | 체결 시점 |
|----------|-----------|----------|
| **일반 시그널** | None | 다음 TF bar open (deferred) |
| **SL/TS** | order.price 설정 | 즉시 (immediate) |
| **CB 청산** | pos.last_price | 즉시 (immediate) |

### 검증 코드

```bash
# BacktestExecutor: deferred vs immediate 분기
grep -n "pending\|_pending_orders\|fill_pending\|drain_fills" src/eda/executors.py

# LiveExecutor: 동일 패턴 확인
grep -n "price.*None\|reduceOnly\|_resolve_position_side" src/eda/executors.py
```

### PASS 조건

- BacktestExecutor: price=None → pending, price set → 즉시 fill
- LiveExecutor: price=None → market order (다음 bar), price set → limit/즉시
- 두 Executor가 동일한 조건 분기 로직을 가짐

### FAIL 시 영향

- **라이브**: SL/TS가 deferred되어 다음 bar까지 청산 지연 → MDD 급증

---

## L4: PM Batch Mode 동작

### 목적

멀티에셋 환경에서 PM의 batch processing이 올바르게 작동하는지 확인.
단일에셋이면 N/A.

### 검증 코드

```bash
# flush_pending_signals 호출 확인
grep -n "flush_pending\|_batch_mode\|_pending_signals" \
  src/eda/runner.py src/eda/live_runner.py src/eda/portfolio_manager.py
```

### PASS 조건

- `len(asset_weights) > 1`일 때 `_batch_mode=True` 자동 활성화
- Runner에서 `pm.flush_pending_signals()` 호출 존재
- 동일 equity snapshot으로 모든 심볼 주문 일괄 실행
- effective_threshold = rebalance_threshold × asset_weight 스케일링

### 단일에셋 N/A 조건

- `asset_weights`가 None 또는 단일 심볼 → batch mode 비활성 → L4 = N/A

### FAIL 시 영향

- **라이브**: 시그널 버퍼에 남아 미실행 → 포지션 미조정 → 리스크 노출

---

## L5: Position Reconciler 연동

### 목적

LiveRunner에서 PositionReconciler가 주기적으로 실행되어
PM의 내부 포지션과 거래소 실제 포지션 간 drift를 감지하는지 확인.

### 검증 코드

```bash
# LiveRunner에서 reconciliation 구현
grep -n "reconcil\|_periodic\|drift\|_DRIFT_THRESHOLD" \
  src/eda/live_runner.py src/eda/reconciler.py
```

### PASS 조건

- `PositionReconciler` 클래스 존재 (`src/eda/reconciler.py`)
- `_DRIFT_THRESHOLD` 정의 (기본 5%)
- LiveRunner에서 `_periodic_reconciliation()` 주기적 실행 (기본 60초)
- 자동 수정 없음 (safety-first) — 경고만 발행

### FAIL 시 영향

- **라이브**: PM과 거래소 간 포지션 drift 감지 불가 → 잘못된 사이징 → 과도한 레버리지

---

## L6: Graceful Shutdown

### 목적

LiveRunner에서 SIGTERM/SIGINT 수신 시 안전하게 종료되는지 확인.
미체결 주문, 열린 포지션, 부분 캔들이 적절히 처리되어야 함.

### 검증 코드

```bash
# Signal handler 등록
grep -n "SIGTERM\|SIGINT\|signal\.signal\|add_signal_handler" src/eda/live_runner.py

# Task cancellation + cleanup
grep -n "cancel\|gather.*return_exceptions\|shutdown\|_wait_shutdown" src/eda/live_runner.py

# Partial candle flush
grep -n "flush_all\|flush.*partial" src/eda/live_data_feed.py src/eda/candle_aggregator.py
```

### PASS 조건

1. SIGTERM/SIGINT signal handler 등록
1. `_shutdown_event.set()` → stream tasks cancel
1. `asyncio.gather(return_exceptions=True)` 로 task 정리
1. LiveDataFeed `flush_all()` 호출 (partial candle flush)
1. 로그에 shutdown 상태 기록

### 일반적 패턴

```python
# 올바른 패턴 (live_runner.py)
async def _wait_shutdown(self):
    await self._shutdown_event.wait()
    for task in self._stream_tasks:
        task.cancel()
    await asyncio.gather(*self._stream_tasks, return_exceptions=True)
```

### FAIL 시 영향

- **라이브**: 프로세스 강제 종료 시 미체결 주문 orphan → 의도치 않은 포지션
- WebSocket 연결 미정리 → 리소스 leak

---

## L7: Circuit Breaker Close 즉시성

### 목적

CircuitBreaker (시스템 최대 DD 발동) 시 전량 청산 주문이
**deferred 되지 않고 즉시 체결**되는지 확인.

### 위험

CB 청산 주문에 price=None → pending_orders에 추가 → 다음 bar까지 청산 지연 →
급락 시 추가 손실.

### 검증 코드

```bash
# OMS의 circuit breaker 처리
grep -n "circuit\|close_all\|CircuitBreaker" src/eda/oms.py

# PM의 CB 관련 주문
grep -n "last_price\|close.*order\|CircuitBreaker" src/eda/portfolio_manager.py

# CB 주문에 price 설정 확인
grep -A5 "circuit\|close_all" src/eda/oms.py
```

### PASS 조건

- CB 청산 주문 생성 시 `price=pos.last_price` 설정
- price 설정 → BacktestExecutor에서 즉시 체결 (deferred 아님)
- LiveExecutor에서도 market order로 즉시 실행

### FAIL 시 영향

- **라이브**: 시스템 최대 DD 발동 시에도 청산이 다음 bar까지 지연 → 파산 위험

---

## 종합 판정 매트릭스

| 항목 | Critical | 판정 기준 |
|------|:--------:|----------|
| **L1** EventBus Flush | **Yes** | flush 누락 = 실행 로직 전체 결함 |
| **L2** Handler 순서 | **Yes** | 순서 오류 = 시그널/체결 불일치 |
| **L3** Deferred Execution | **Yes** | 패턴 불일치 = SL/TS 지연 위험 |
| **L4** PM Batch Mode | No (단일에셋 N/A) | 멀티에셋 전용, 단일에셋은 검증 불필요 |
| **L5** Reconciler | No | 경고 전용, 자동 수정 없으므로 중요도 중간 |
| **L6** Graceful Shutdown | **Yes** | shutdown 미처리 = 포지션 orphan 위험 |
| **L7** Circuit Breaker | **Yes** | CB 즉시성 = 파산 방어 최후 수단 |

### 판정 규칙

| 결과 | 조건 |
|------|------|
| **PASS** | 7/7 전 항목 PASS (또는 N/A 포함) |
| **WARNING** | L4 또는 L5 1개 FAIL (non-critical) |
| **FAIL** | L1, L2, L3, L6, L7 중 1개라도 FAIL |

---

## EDA 시스템 핵심 파일 맵

| 파일 | 역할 | L 항목 |
|------|------|--------|
| `src/eda/runner.py` | 백테스트 Runner (컴포넌트 조립) | L2, L4 |
| `src/eda/live_runner.py` | 라이브 Runner (WebSocket + 실행) | L2, L4, L5, L6 |
| `src/eda/data_feed.py` | 히스토리컬 DataFeed | L1 |
| `src/eda/live_data_feed.py` | 라이브 DataFeed (WebSocket) | L1, L6 |
| `src/eda/executors.py` | BacktestExecutor + LiveExecutor | L3, L7 |
| `src/eda/portfolio_manager.py` | PM (포지션/시그널 관리) | L4, L7 |
| `src/eda/oms.py` | OMS (주문 라우팅) | L7 |
| `src/eda/reconciler.py` | 포지션 교차검증 | L5 |
| `src/eda/candle_aggregator.py` | 1m→TF 캔들 집계 | L1 |
| `src/core/event_bus.py` | EventBus (비동기 큐) | L1 |
